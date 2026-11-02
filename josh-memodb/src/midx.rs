//! Keeps a multi-pack index over an objects directory's packs.
//!
//! A gitoxide object store has a fixed number of slots, sized when it opens: by default 1.1 times
//! the pack indices on disk then, and at least 32. Every pack index takes a slot, but a
//! multi-pack index takes one slot for all the packs it covers (gix-odb lists the indices it
//! covers under it, not on their own). josh writes a pack for every flush, so a store that stays
//! open, as the proxy's do, fills up unless the packs are covered. [`maintain`] rewrites the
//! multi-pack index once more than [`MAX_UNCOVERED_PACKS`] packs are outside it, which bounds the
//! slots a store needs without rewriting any pack data.
//!
//! Writers follow git's protocol for `objects/pack/multi-pack-index`: write
//! `multi-pack-index.lock`, created exclusively, then rename it into place. A writer that finds
//! the lock taken skips, since whoever holds it is writing a multi-pack index already. As with
//! git, a lock left behind by a crash holds off every writer until someone removes it; packs
//! stay readable meanwhile, only the bound on slots is lost.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Mutex, MutexGuard};

/// Packs that may sit outside the multi-pack index before [`maintain`] rewrites it. Together
/// with the multi-pack index itself this stays well under the 32 slots a store opened on a young
/// cache gets, leaving room for packs that `git` writes between two calls.
pub const MAX_UNCOVERED_PACKS: usize = 8;

/// Held while this process writes a multi-pack index, and by maintenance that deletes packs
/// (see [`exclusive`]), so a multi-pack index never lists a pack a concurrent repack removes.
static MAINTENANCE: Mutex<()> = Mutex::new(());

/// Hold off [`maintain`] in this process, for example while a repack deletes packs.
pub fn exclusive() -> MutexGuard<'static, ()> {
    MAINTENANCE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Rewrite `objects_dir/pack/multi-pack-index` to cover every pack there, if more than
/// [`MAX_UNCOVERED_PACKS`] packs are outside it or it lists a pack that is gone. Returns whether
/// it wrote one. Skips, returning `false`, while another writer holds the lock or [`exclusive`]
/// is held.
pub fn maintain(objects_dir: &Path) -> anyhow::Result<bool> {
    let _maintenance = match MAINTENANCE.try_lock() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
    };
    let pack_dir = objects_dir.join("pack");
    let indices = pack_indices(&pack_dir)?;
    let midx_path = pack_dir.join("multi-pack-index");
    let covered: Vec<PathBuf> = match gix_pack::multi_index::File::at(&midx_path, None) {
        Ok(midx) => midx.index_names().to_vec(),
        Err(_) => Vec::new(),
    };
    let uncovered = indices
        .iter()
        .filter(|path| !covered.iter().any(|name| path.ends_with(name)))
        .count();
    let stale = covered
        .iter()
        .any(|name| !indices.iter().any(|path| path.ends_with(name)));
    if uncovered <= MAX_UNCOVERED_PACKS && !stale {
        return Ok(false);
    }

    let lock_path = pack_dir.join("multi-pack-index.lock");
    let lock = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
    {
        Ok(lock) => lock,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let written = (|| -> anyhow::Result<()> {
        let mut out = std::io::BufWriter::new(lock);
        gix_pack::multi_index::write_from_index_paths(
            indices,
            &mut out,
            &mut gix_features::progress::Discard,
            &AtomicBool::new(false),
            gix_pack::multi_index::write::Options {
                object_hash: gix_hash::Kind::Sha1,
            },
        )?;
        out.into_inner().map_err(|error| error.into_error())?;
        std::fs::rename(&lock_path, &midx_path)?;
        Ok(())
    })();
    if written.is_err() {
        std::fs::remove_file(&lock_path).ok();
    }
    written.map(|()| true)
}

/// The pack indices in `pack_dir` whose pack is present, as gix-odb counts them.
fn pack_indices(pack_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let entries = match std::fs::read_dir(pack_dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut indices = Vec::new();
    for entry in entries {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "idx") && path.with_extension("pack").is_file()
        {
            indices.push(path);
        }
    }
    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Once a flush returns, a store with one slot more than [`MAX_UNCOVERED_PACKS`] reads every
    /// object, however many packs the flushes wrote.
    #[test]
    fn flushes_keep_packs_covered() {
        let dir = tempfile::tempdir().unwrap();
        let repo = gix::init_bare(dir.path()).unwrap();
        let store = crate::MemOdb::new(None, crate::pack::objects_dir(&repo));
        let ids: Vec<_> = (0..3 * MAX_UNCOVERED_PACKS as u8)
            .map(|i| {
                let id = store.write(gix_object::Kind::Blob, &[i]);
                store.flush().unwrap();
                id
            })
            .collect();

        let slots = gix_odb::store::init::Slots::Given(MAX_UNCOVERED_PACKS as u16 + 1);
        let options = gix::open::Options::isolated().object_store_slots(slots);
        let disk = gix::open_opts(dir.path(), options).unwrap();
        for id in ids {
            disk.find_object(id).unwrap();
        }
    }
}
