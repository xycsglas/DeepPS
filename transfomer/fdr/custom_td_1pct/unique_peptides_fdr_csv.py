#!/usr/bin/env python3
"""Unique peptides from a V20/Prosit 1% FDR result CSV.

The FDR ranking is identical to fdr/step4_fdr_curve.py: per branch, sort by
fused score DESC (ties: smaller ordinal first), fdr_dt=(n_decoy+1)/n_target,
q_dt = suffix-min.  Spectra with q_dt <= --fdr are accepted; targets
(label != 'D') contribute their per-branch top-1 sequence.

Unique sets are reported for peptides whose backbone (mods stripped) is
<= --max-len aa.  Both backbone and normalized annotated forms are written.

Usage:
  python unique_peptides_fdr_csv.py \
    --results <out>/results_step3_AT.csv --branch d_f_exp_mby_rel005 \
    --name v20 --out-dir <out>/unique

Outputs (in --out-dir):
  unique_<branch>_backbone_<name>.tsv    backbone, n_spectra, annotated
  unique_<branch>_annotated_<name>.tsv   annotated, backbone, n_spectra
  unique_<branch>_summary_<name>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

FLANK_RE = re.compile(r"^[A-Z-]*\.(.*?)\.[A-Z-]*$")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
INLINE_MOD_RE = re.compile(r"([A-Z])([+-]\d+(?:\.\d+)?)")

CANON = {
    57.021: 57.021, 57.02: 57.021, 57.02146: 57.021,
    15.995: 15.995, 15.9949: 15.995, 15.994915: 15.995,
    79.966: 79.966, 79.96633: 79.966, 79.966331: 79.966,
    0.984: 0.984, 0.984016: 0.984,
    42.011: 42.011, 42.010565: 42.011,
    43.006: 43.006, 43.005814: 43.006,
    -17.027: -17.027, -17.026549: -17.027,
}


def strip_flanks(seq: str) -> str:
    m = FLANK_RE.match(seq or "")
    return m.group(1) if m else (seq or "")


def backbone(seq: str) -> str:
    core = strip_flanks(seq)
    core = BRACKET_RE.sub("", core)
    core = re.sub(r"\([^)]*\)", "", core)
    core = NUM_RE.sub("", core)
    return "".join(ch for ch in core if ch.isalpha())


def canon_delta(raw) -> str:
    d = float(raw)
    for k, v in CANON.items():
        if abs(d - k) < 0.002:
            return f"{v:+.3f}"
    return f"{d:+.3f}"


def normalize_annotated(seq: str) -> str:
    s = strip_flanks(seq)
    s = re.sub(r"\(([+-]?\d+(?:\.\d+)?)\)", lambda m: canon_delta(m.group(1)), s)
    s = re.sub(r"\[([+-]?\d+(?:\.\d+)?)\]", lambda m: canon_delta(m.group(1)), s)
    s = INLINE_MOD_RE.sub(lambda m: f"{m.group(1)}{canon_delta(m.group(2))}", s)
    return s


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--branch", required=True,
                    help="branch key(s), comma-separated, e.g. "
                         "d_f_exp_mby_rel005,d_f_exp_rel010")
    ap.add_argument("--name", required=True, help="method name for output files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=30,
                    help="unique-peptide length cap (peptides only)")
    ap.add_argument("--row-max-len", type=int, default=None,
                    help="also drop spectra whose branch top-1 backbone exceeds "
                         "this BEFORE the FDR ranking (nine-species <=30 pool "
                         "convention; no GT is needed)")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    branches = [b.strip() for b in a.branch.split(",") if b.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(a.results, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        raw_rows = list(reader)
    ordinal_by_row = []
    for i, r in enumerate(raw_rows):
        try:
            ordinal_by_row.append(int(r.get("ordinal") or i))
        except ValueError:
            ordinal_by_row.append(i)

    for branch in branches:
        fcol, lcol, scol = (f"{branch}_fused", f"{branch}_label",
                            f"{branch}_seq")
        missing = [c for c in (fcol, lcol, scol) if c not in header]
        if missing:
            raise SystemExit(f"missing columns {missing} in {a.results}")

        rows = []
        n_len_dropped = 0
        for i, r in enumerate(raw_rows):
            seq = (r.get(scol) or "").strip()
            if a.row_max_len is not None and len(backbone(seq)) > a.row_max_len:
                n_len_dropped += 1
                continue
            try:
                fused = float(r.get(fcol) or "-inf")
            except ValueError:
                fused = float("-inf")
            rows.append({"ordinal": ordinal_by_row[i],
                         "label": (r.get(lcol) or "").strip(),
                         "fused": fused,
                         "seq": seq})

        ordered = sorted(rows, key=lambda r: (r["fused"], -r["ordinal"]),
                         reverse=True)
        fdr = []
        cum_t = cum_d = 0
        for r in ordered:
            if r["label"] == "D":
                cum_d += 1
            else:
                cum_t += 1
            fdr.append(((cum_d + 1) / cum_t) if cum_t else float("inf"))
        q = [0.0] * len(fdr)
        best = float("inf")
        for i in range(len(fdr) - 1, -1, -1):
            best = min(best, fdr[i])
            q[i] = best

        accepted = [r for r, qq in zip(ordered, q) if qq <= a.fdr]
        targets = [r for r in accepted if r["label"] != "D"]

        bb_spectra: dict[str, int] = defaultdict(int)
        ann_by_bb: dict[str, set] = defaultdict(set)
        ann_spectra: dict[str, int] = defaultdict(int)
        ann_backbone: dict[str, str] = {}
        n_over_len = 0
        for r in targets:
            bb = backbone(r["seq"])
            if not bb:
                continue
            if len(bb) > a.max_len:
                n_over_len += 1
                continue
            ann = normalize_annotated(r["seq"])
            bb_spectra[bb] += 1
            ann_by_bb[bb].add(ann)
            ann_spectra[ann] += 1
            ann_backbone.setdefault(ann, bb)

        with open(out / f"unique_{branch}_backbone_{a.name}.tsv", "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["backbone", "n_spectra", "annotated"])
            for bb in sorted(bb_spectra):
                w.writerow([bb, bb_spectra[bb],
                            ";".join(sorted(ann_by_bb[bb]))])
        with open(out / f"unique_{branch}_annotated_{a.name}.tsv", "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["annotated", "backbone", "n_spectra"])
            for ann in sorted(ann_spectra):
                w.writerow([ann, ann_backbone[ann], ann_spectra[ann]])

        summary = {
            "method": a.name, "results": str(a.results), "branch": branch,
            "fdr": a.fdr, "max_len": a.max_len,
            "score_column": fcol,
            "threshold": (round(float(accepted[-1]["fused"]), 6)
                          if accepted else None),
            "row_max_len": a.row_max_len,
            "n_rows_total": len(raw_rows),
            "n_rows_len_dropped": n_len_dropped,
            "n_rows": len(rows), "n_accepted": len(accepted),
            "n_target": len(targets), "n_decoy": len(accepted) - len(targets),
            "n_target_over_max_len": n_over_len,
            "n_unique_backbone": len(bb_spectra),
            "n_unique_annotated": len(ann_spectra),
        }
        (out / f"unique_{branch}_summary_{a.name}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"[unique] {a.name}/{branch}: target={len(targets)} "
              f"(>{a.max_len}aa dropped={n_over_len}) "
              f"uniq_backbone={len(bb_spectra)} "
              f"uniq_annotated={len(ann_spectra)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
