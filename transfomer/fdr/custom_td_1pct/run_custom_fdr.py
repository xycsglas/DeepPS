#!/usr/bin/env python3
"""One-command V20_0918 1% FDR for a custom MGF + target-decoy peptide FASTA.

Pipeline:
  1. convert_td_fasta.py     plain TD FASTA (>DECOY_) -> MASS=/LABEL=/SEQ= format
  2. step2_match_custom.py   precursor-mass matching (±--ppm) -> candidates TSV.gz
  3. step3_score_custom.py   V20 (39x46x6) candidate scoring -> results_step3_<name>.csv
  4. step4_fdr_curve.py      (D+1)/T q-value counting -> summary_fdr_<name>.json

Usage (run in v20_0918/transfomer/fdr/custom_td_1pct):
  python run_custom_fdr.py \
    --name AT \
    --mgf "E:/夸克/download/AT/A.thaliana_PXD012708.filtered_minpeak64.mgf" \
    --fasta "E:/夸克/download/AT/Ath_target_decoy_peptide.fasta" \
    --work "E:/data/custom_td_work" \
    --checkpoint /home/wangyuhang/seq2mass/runs/v200918_massivekb46/best_model.pt \
    --gpu 0

Smoke test: add --max-scans 2000.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDR_DIR = HERE.parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="dataset name, e.g. AT/CE/EC/HEK/HELA")
    ap.add_argument("--mgf", required=True)
    ap.add_argument("--fasta", required=True, help="target-decoy peptide FASTA (>DECOY_ decoys)")
    ap.add_argument("--work", required=True, help="work root; outputs go to <work>/<name>/")
    ap.add_argument("--checkpoint", required=True, help="V20_0918 best_model.pt")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--cys-mod", type=float, default=57.021,
                    help="fixed C mass shift applied in the converted FASTA (0 disables)")
    ap.add_argument("--variable-mods", default="M:15.995",
                    help="variable mods for V20 variant enumeration "
                         "(e.g. M:15.995 or M:15.995,STY:79.96633)")
    ap.add_argument("--rel", default="0.05,0.10,0.20")
    ap.add_argument("--branches", default=None,
                    help="step3 --branches string (default: step3's built-in set)")
    ap.add_argument("--fdr-targets", default="0.01")
    ap.add_argument("--unique-branches",
                    default="d_f_exp_rel005,d_f_exp_rel010,d_f_exp_rel020,"
                            "d_f_exp_mby_rel005,d_f_exp_mby_rel010,d_f_exp_mby_rel020",
                    help="comma-separated branches for unique-peptide export "
                         "(empty disables the post-step4 export)")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--force-match", action="store_true")
    ap.add_argument("--force-score", action="store_true")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    root = Path(a.work) / a.name
    data = root / "data"
    out = root / "output"
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    td_fasta = data / f"{a.name}_td.fasta"
    candidates = data / f"candidates_{a.name}_td.tsv.gz"
    results = out / f"results_step3_{a.name}.csv"
    py = sys.executable

    if a.force_convert or not td_fasta.exists():
        run([py, str(HERE / "convert_td_fasta.py"),
             "--fasta", a.fasta, "--out", str(td_fasta),
             "--engine-out", str(data / f"{a.name}_td_engine.fasta"),
             "--min-len", str(a.min_len), "--max-len", str(a.max_len),
             "--cys-mod", str(a.cys_mod),
             "--variable-mods", a.variable_mods,
             "--stats-json", str(root / "convert_stats.json")])
    else:
        print(f"[run] reuse {td_fasta}")

    if a.force_match or not candidates.exists():
        run([py, str(HERE / "step2_match_custom.py"),
             "--species", a.name, "--mgf", a.mgf, "--fasta", str(td_fasta),
             "--out-candidates", str(candidates), "--out-dir", str(out),
             "--ppm", str(a.ppm)]
            + (["--max-scans", str(a.max_scans)] if a.max_scans else []))
    else:
        print(f"[run] reuse {candidates}")

    cmd = [py, str(HERE / "step3_score_custom.py"),
           "--species", a.name, "--mgf", a.mgf, "--candidates", str(candidates),
           "--checkpoint", a.checkpoint, "--out-dir", str(out),
           "--gpu", str(a.gpu), "--batch-size", str(a.batch_size),
           "--rel", a.rel]
    if a.branches:
        cmd += ["--branches", a.branches]
    if a.max_scans:
        cmd += ["--max-scans", str(a.max_scans)]
    if a.force_score or not results.exists():
        run(cmd)
    else:
        print(f"[run] reuse {results}")

    run([py, str(FDR_DIR / "step4_fdr_curve.py"),
         "--species", a.name, "--out-dir", str(out),
         "--results", str(results), "--fdr-targets", a.fdr_targets])

    for branch in [b.strip() for b in a.unique_branches.split(",") if b.strip()]:
        run([py, str(HERE / "unique_peptides_fdr_csv.py"),
             "--results", str(results), "--branch", branch,
             "--name", "v20", "--out-dir", str(out / "unique")])

    print(f"\n[done] {a.name}: results -> {results}")
    print(f"[done] summary -> {out / f'summary_fdr_{a.name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
