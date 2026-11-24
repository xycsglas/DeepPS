#!/usr/bin/env python3
"""One-command Prosit 1% FDR for a custom MGF + target/decoy candidate TSV.

Pipeline:
  1. prefetch_custom_prosit.py  (N shards for T and D, Koina, resumable)
  2. merge shard segment caches into prosit_target_cache.tsv.gz / ..._decoy...
  3. prosit_fdr_score_custom.py (cached scoring on every spectrum)
  4. step4_fdr_curve.py         ((D+1)/T q-value counting)

The heavy network part is step 1; run it on the server with `nohup`.
Each shard is independent and resumable; the merge is a plain concatenation.

Usage:
  python run_custom_prosit.py --name AT \
    --mgf ".../AT.mgf" --candidates "<work>/AT/data/candidates_AT_td.tsv.gz" \
    --work "<work>" --jobs 8

Smoke test: --limit-keys 2000 --max-scans 500.
"""
from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDR_DIR = HERE.parent


def run(cmd: list[str], env: dict) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def merge_shards(shard_dir: Path, tag: str, out: Path) -> int:
    files = sorted(p for p in shard_dir.iterdir()
                   if p.is_file() and p.name.startswith(f"_{tag}_shard")
                   and ".tsv.gz" in p.name)   # exclude per-shard .log files
    if not files:
        raise SystemExit(f"no shard caches for {tag} in {shard_dir}")
    n = 0
    with gzip.open(out, "wb", compresslevel=1) as fo:
        for path in files:
            with gzip.open(path, "rb") as fi:
                shutil.copyfileobj(fi, fo)
            n += 1
    print(f"[merge] {tag}: {n} shard files -> {out} "
          f"({out.stat().st_size / 1e9:.2f} GB)", flush=True)
    return n


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--mgf", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--jobs", type=int, default=8, help="shards per label")
    ap.add_argument("--batch-size", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--rel", default="0,0.05,0.10,0.20")
    ap.add_argument("--fdr-targets", default="0.01")
    ap.add_argument("--unique-branches",
                    default="f_frag_rel000,f_frag_rel005,"
                            "f_frag_rel010,f_frag_rel020",
                    help="comma-separated branches for unique-peptide export "
                         "(empty disables the post-step4 export)")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--limit-keys", type=int, default=None)
    ap.add_argument("--skip-prefetch", action="store_true")
    ap.add_argument("--prefetch-only", action="store_true",
                    help="run prefetch + merge, then stop (score later with --skip-prefetch)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="score only; missing keys are predicted per spectrum (slow)")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    root = Path(a.work) / a.name
    data = root / "data"
    out = root / "output"
    shard_dir = data / "prosit_shards"
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    py = sys.executable
    target_cache = data / "prosit_target_cache.tsv.gz"
    decoy_cache = data / "prosit_decoy_cache.tsv.gz"

    if not a.skip_prefetch and not a.no_prefetch:
        procs = []
        for tag, labels in (("t", "T"), ("d", "D")):
            for k in range(a.jobs):
                cmd = [py, str(HERE / "prefetch_custom_prosit.py"),
                       "--candidates", a.candidates,
                       "--labels", labels,
                       "--cache", str(shard_dir / f"_{tag}_shard{k}.tsv.gz"),
                       "--shard-index", str(k), "--num-shards", str(a.jobs),
                       "--batch-size", str(a.batch_size),
                       "--max-len", str(a.max_len)]
                if a.limit_keys:
                    cmd += ["--limit-keys", str(a.limit_keys)]
                log = open(shard_dir / f"_{tag}_shard{k}.log", "w", encoding="utf-8")
                print("$ " + " ".join(cmd), flush=True)
                procs.append((subprocess.Popen(cmd, env=env, stdout=log,
                                               stderr=subprocess.STDOUT), log))
        failed = 0
        for p, log in procs:
            if p.wait() != 0:
                failed += 1
            log.close()
        if failed:
            raise SystemExit(f"{failed} prefetch shard(s) failed; see {shard_dir}")

    # Merge shard caches whenever they exist -- including --skip-prefetch, where
    # the merged files may be missing or stale.  The shards are authoritative.
    have_shards = any(p.is_file() and ".tsv.gz" in p.name
                      for p in shard_dir.iterdir())
    if have_shards:
        for p in list(data.glob("prosit_target_cache.tsv.gz*")) + \
                 list(data.glob("prosit_decoy_cache.tsv.gz*")):
            p.unlink()
        merge_shards(shard_dir, "t", target_cache)
        merge_shards(shard_dir, "d", decoy_cache)
    else:
        print("[run] no shard caches; assuming merged caches already exist")

    if a.prefetch_only:
        print(f"\n[prefetch-only] {a.name}: caches ready -> {data}")
        return 0

    cmd = [py, str(HERE / "prosit_fdr_score_custom.py"),
           "--species", a.name, "--mgf", a.mgf,
           "--candidates", a.candidates, "--out-dir", str(out),
           "--cache", str(target_cache), "--decoy-cache", str(decoy_cache),
           "--rel-intensities", a.rel, "--batch-size", str(a.batch_size)]
    if a.max_scans:
        cmd += ["--max-scans", str(a.max_scans)]
    run(cmd, env)

    results = out / f"results_prositfdr_{a.name}.csv"
    run([py, str(FDR_DIR / "step4_fdr_curve.py"),
         "--species", a.name, "--out-dir", str(out),
         "--results", str(results), "--fdr-targets", a.fdr_targets], env)

    for branch in [b.strip() for b in a.unique_branches.split(",") if b.strip()]:
        run([py, str(HERE / "unique_peptides_fdr_csv.py"),
             "--results", str(results), "--branch", branch,
             "--name", "prosit", "--out-dir", str(out / "unique")], env)

    print(f"\n[done] {a.name}: {results}")
    print(f"[done] summary -> {out / f'summary_fdr_{a.name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
