#!/usr/bin/env python3
"""Batched Prosit prefetch for a custom target/decoy candidate TSV.

Collects the unique (backbone, charge) keys needed by prosit_fdr_score_custom.py
and predicts them through Koina in chunks.  Charges come directly from the
candidates TSV (column 2), so no MGF pass is needed.

Sharding: each shard writes its OWN cache path (safe for parallel processes):

    python prefetch_custom_prosit.py --candidates ... --labels T \
        --cache <shard_dir>/_t_shard0.tsv.gz --shard-index 0 --num-shards 8

After all shards finish, concatenate their `*.seg_*.tsv.gz` files into one base
cache; PrositPredictor._load_cache reads base + segments (see run_custom_prosit.py).

"""
from __future__ import annotations

import argparse
import csv
import gzip
import sys
import zlib
from pathlib import Path

# Self-contained Prosit implementation under this 0918 directory.
_HERE = Path(__file__).resolve().parent
_TF = _HERE.parents[1]                              # v20_0918/transfomer
for _p in (str(_TF), str(_TF / "ds"), str(_TF / "scripts"),
           str(_TF / "fdr"), str(_HERE / "prosit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import step4_prosit_score as p4  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", required=True, help="candidates_<name>_td.tsv.gz")
    ap.add_argument("--cache", required=True, help="THIS SHARD's own cache path")
    ap.add_argument("--labels", choices=["T", "D", "TD"], default="TD")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=p4.PROSIT_MAX_LEN)
    ap.add_argument("--limit-keys", type=int, default=None,
                    help="only the first N sorted keys (smoke test)")
    ap.add_argument("--count-only", action="store_true")
    return ap.parse_args()


def collect_keys(path: Path, labels: str, max_len: int) -> set[tuple[str, int]]:
    needed: set[tuple[str, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd, None)
        for row in rd:
            if not row or len(row) < 7:
                continue
            seq, lab = row[5], row[6]
            if labels != "TD" and lab != labels:
                continue
            try:
                ch = int(row[2])
            except ValueError:
                continue
            bb = p4.strip_mods(seq)
            if ch > 0 and len(bb) <= max_len:
                needed.add((bb, ch))
    return needed


def main() -> int:
    a = parse_args()
    if not (0 <= a.shard_index < a.num_shards):
        raise SystemExit("shard-index must be in [0, num-shards)")
    needed = collect_keys(Path(a.candidates), a.labels, a.max_len)
    if a.limit_keys:
        needed = set(sorted(needed)[:a.limit_keys])
    mine = {k for k in needed if zlib.crc32(
        f"{k[0]}\t{k[1]}".encode("utf-8")) % a.num_shards == a.shard_index}
    print(f"labels={a.labels} shard {a.shard_index}/{a.num_shards}: "
          f"{len(mine):,} of {len(needed):,} keys", flush=True)
    if a.count_only:
        return 0

    pred = p4.PrositPredictor(cache_path=Path(a.cache),
                              batch_size=a.batch_size, mods_mode="reanchor")
    already = len(mine & set(pred.cache))
    print(f"  cache already has {already:,}; predicting {len(mine) - already:,}",
          flush=True)
    pred.predict_missing(mine)
    pred.save_cache()
    print(f"  done: cache has {len(pred.cache):,} -> {a.cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
