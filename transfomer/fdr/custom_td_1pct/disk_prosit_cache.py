#!/usr/bin/env python3
"""Disk-backed Prosit prediction cache with the PrositPredictor interface.

The stock PrositPredictor loads every cached (peptide, charge) prediction into
Python objects (tens of GB for AT/HEK).  This variant converts the same
``prosit_*_cache.tsv.gz`` (+ ``.seg_*`` files) into a SQLite file next to the
cache, one row per key:

    pep TEXT, charge INTEGER, mz BLOB(float32), it BLOB(float32), annos BLOB

Lookups are point queries; memory holds only the key set (a few hundred MB per
million keys) plus a bounded LRU of recently used predictions.  The stored
peaks/annotations are produced with the same ``_sorted_peaks`` call as the
stock loader, so scores are identical.

Building the DB is one-time per cache file (a few minutes per million keys).
"""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TF = _HERE.parents[1]
for _p in (str(_TF), str(_TF / "ds"), str(_TF / "scripts"),
           str(_TF / "fdr"), str(_HERE / "prosit"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import step4_prosit_score as p4  # noqa: E402


class DiskPrositPredictor:
    def __init__(self, cache_path, batch_size: int = 1500,
                 mods_mode: str = "reanchor", db_path=None,
                 lru_size: int = 200_000):
        if cache_path is None:
            raise SystemExit("DiskPrositPredictor requires a cache path")
        self.cache_path = Path(cache_path)
        self.batch_size = batch_size
        self.mods_mode = mods_mode
        self.db_path = (Path(db_path) if db_path else
                        Path(str(self.cache_path) + ".sqlite"))
        self._lru: OrderedDict = OrderedDict()
        self._lru_size = lru_size
        self._engine = None

        if not self.db_path.exists() or self._stale():
            self._build()

        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA cache_size=-1048576")   # 1 GiB page cache
        self.conn.execute("PRAGMA mmap_size=8589934592")  # 8 GiB mmap window
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._keys: set[tuple[str, int]] = set()
        for pep, ch in self.conn.execute("SELECT pep, charge FROM peaks"):
            self._keys.add((pep, int(ch)))
        print(f"[prosit-disk] {self.db_path.name}: "
              f"{len(self._keys):,} keys indexed", flush=True)

    # ------------------------------------------------------------------ build
    def _cache_files(self) -> list[Path]:
        files = [self.cache_path] if self.cache_path.exists() else []
        files += sorted(self.cache_path.parent.glob(
            self.cache_path.name + ".seg_*.tsv.gz"))
        return files

    def _stale(self) -> bool:
        newest = 0.0
        for p in self._cache_files():
            newest = max(newest, p.stat().st_mtime)
        return self.db_path.stat().st_mtime < newest

    def _iter_rows(self):
        # Byte-identical to PrositPredictor._load_cache: split on b"\n" only,
        # so a CRLF cache keeps its trailing "\r" in the annotation field too.
        for path in self._cache_files():
            with gzip.open(path, "rb") as fh:
                for bline in fh:
                    if bline.startswith(b"peptide\t"):
                        continue
                    parts = bline.rstrip(b"\n").split(b"\t")
                    if len(parts) < 4:
                        continue
                    try:
                        ch = int(parts[1])
                    except ValueError:
                        continue
                    yield (parts[0].decode("utf-8", "replace"), ch,
                           float(parts[2]), float(parts[3]),
                           parts[4].decode("utf-8", "replace")
                           if len(parts) > 4 else "")

    def _build(self):
        t0 = time.time()
        tmp = Path(str(self.db_path) + ".tmp")
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(str(tmp))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("CREATE TABLE peaks ("
                     "pep TEXT, charge INTEGER, mz BLOB, it BLOB, annos BLOB, "
                     "PRIMARY KEY (pep, charge)) WITHOUT ROWID")

        cur = None
        mz_l: list[float] = []
        it_l: list[float] = []
        ann_l: list[str] = []
        batch: list[tuple] = []
        n_frag = 0
        n_keys = 0

        def flush_key():
            nonlocal cur, mz_l, it_l, ann_l, n_keys
            if cur is None:
                return
            arr = np.asarray(list(zip(mz_l, it_l)), dtype=np.float32)
            arr = p4._sorted_peaks(arr)
            ann = ann_l if (ann_l and ann_l[0]) else None
            batch.append((cur[0], cur[1],
                          arr[:, 0].tobytes(), arr[:, 1].tobytes(),
                          json.dumps(ann) if ann is not None else None))
            n_keys += 1
            cur, mz_l, it_l, ann_l = None, [], [], []

        for pep, ch, mz, it, anno in self._iter_rows():
            n_frag += 1
            key = (pep, ch)
            if key != cur:
                flush_key()
                cur = key
            mz_l.append(mz)
            it_l.append(it)
            ann_l.append(anno)
            if len(batch) >= 20_000:
                conn.executemany("INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?)",
                                 batch)
                conn.commit()
                batch.clear()
            if n_frag % 5_000_000 == 0:
                print(f"[prosit-disk] {self.db_path.name}: {n_frag / 1e6:.0f}M "
                      f"fragments, {n_keys:,} keys, "
                      f"{time.time() - t0:.0f}s", flush=True)
        flush_key()
        if batch:
            conn.executemany("INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?)",
                             batch)
        conn.commit()
        conn.close()
        if self.db_path.exists():
            self.db_path.unlink()
        tmp.replace(self.db_path)
        print(f"[prosit-disk] built {self.db_path.name}: {n_keys:,} keys, "
              f"{n_frag / 1e6:.1f}M fragments in {time.time() - t0:.0f}s "
              f"({self.db_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    # --------------------------------------------------------------- interface
    def _lru_put(self, key, value):
        self._lru[key] = value
        self._lru.move_to_end(key)
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)

    def get(self, peptide: str, charge: int):
        key = (peptide, int(charge))
        hit = self._lru.get(key)
        if hit is not None:
            self._lru.move_to_end(key)
            return hit
        row = self.conn.execute(
            "SELECT mz, it, annos FROM peaks WHERE pep=? AND charge=?",
            key).fetchone()
        if row is None:
            val = (np.empty((0, 2), dtype=np.float32), None)
        else:
            mz = np.frombuffer(row[0], dtype=np.float32)
            it = np.frombuffer(row[1], dtype=np.float32)
            arr = np.column_stack([mz, it]).astype(np.float32, copy=False)
            annos = json.loads(row[2]) if row[2] else None
            val = (arr, annos)
        self._lru_put(key, val)
        return val

    def prefetch(self, peptides, charge: int):
        """Load many keys for one charge with a single IN query (chunked).

        This turns ~300 per-candidate SELECTs into 1-2 statements per spectrum,
        which is where most of the sqlite overhead came from.
        """
        ch = int(charge)
        todo = [p for p in peptides if (p, ch) not in self._lru]
        if not todo:
            return
        for i in range(0, len(todo), 800):
            chunk = todo[i:i + 800]
            q = ("SELECT pep, mz, it, annos FROM peaks WHERE charge=? AND pep IN ("
                 + ",".join("?" * len(chunk)) + ")")
            for pep, mz_b, it_b, ann_b in self.conn.execute(q, [ch, *chunk]):
                mz = np.frombuffer(mz_b, dtype=np.float32)
                it = np.frombuffer(it_b, dtype=np.float32)
                arr = np.column_stack([mz, it]).astype(np.float32, copy=False)
                annos = json.loads(ann_b) if ann_b else None
                self._lru_put((pep, ch), (arr, annos))

    def predict_missing(self, needed: set[tuple[str, int]]):
        missing = needed - self._keys
        if not missing:
            # batch-load every present key of this spectrum in 1-2 queries
            by_ch: dict[int, list[str]] = {}
            for pep, ch in needed:
                if (pep, ch) not in self._lru:
                    by_ch.setdefault(ch, []).append(pep)
            for ch, peps in by_ch.items():
                self.prefetch(peps, ch)
            return
        print(f"[prosit-disk] predicting {len(missing)} missing peptides "
              f"(index has {len(self._keys):,})", flush=True)
        by_ch = {}
        for pep, ch in needed - missing:
            if (pep, ch) not in self._lru:
                by_ch.setdefault(ch, []).append(pep)
        for ch, peps in by_ch.items():
            self.prefetch(peps, ch)
        if self._engine is None:
            self._engine = p4.PrositPredictor(
                cache_path=None, batch_size=self.batch_size,
                mods_mode=self.mods_mode)
        self._engine.predict_missing(missing)
        rows = []
        for (pep, ch), (arr, annos) in self._engine.cache.items():
            rows.append((pep, int(ch),
                         arr[:, 0].astype(np.float32, copy=False).tobytes(),
                         arr[:, 1].astype(np.float32, copy=False).tobytes(),
                         json.dumps(annos) if annos is not None else None))
            self._keys.add((pep, int(ch)))
            self._lru_put((pep, int(ch)), (arr, annos))
        if rows:
            with self.conn:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO peaks VALUES (?,?,?,?,?)", rows)
        self._engine.cache.clear()
        self._engine._pending.clear()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Build a SQLite index for a Prosit prediction cache")
    ap.add_argument("cache", help="prosit_*_cache.tsv.gz")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    DiskPrositPredictor(a.cache, db_path=a.db)
    print("done", flush=True)
