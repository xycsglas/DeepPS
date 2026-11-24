#!/usr/bin/env python3
"""Turn a search-engine PSM table into the unique peptides passing 1% FDR.

Same FDR rule as fdr_full/engine_fdr.py: best PSM per spectrum by score
(lower better), sort ascending, accept the largest prefix with
``(n_decoy + 1) / n_target <= fdr``.  This script additionally keeps the
peptide of every accepted target PSM and emits the unique sets.

Two peptide definitions are written:
  backbone  mods stripped, only letters (used for cross-method comparison)
  annotated normalized modification annotation (C+57.021, M+15.995, ...)

Spectrum key: MS-GF+ needs ``--scan SpecId`` (``index=N``) because multi-RAW
MGFs reuse ``ScanNum``; MSFragger uses ``scannum`` (unique after
make_unique_scans.py).

Usage:
  python unique_peptides.py --engine msgfplus \
      --tsv E:/custom_td_work/AT/msgfplus/AT.tsv --decoy-token DECOY_ \
      --out-dir E:/custom_td_work/unique/AT --name msgfplus

Outputs (--out-dir):
  <name>_unique_backbone.tsv    backbone, n_spectra, annotated
  <name>_unique_annotated.tsv   annotated, backbone, n_spectra
  <name>_summary.json
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
FRAG_MOD_RE = re.compile(r"(\d+)([A-Z])\(([+-]?\d+(?:\.\d+)?)\)")
NTERM_MOD_RE = re.compile(r"N-term\(([+-]?\d+(?:\.\d+)?)\)")

CANON = {
    57.021: 57.021, 57.02: 57.021, 57.02146: 57.021,
    15.995: 15.995, 15.9949: 15.995, 15.994915: 15.995,
    79.966: 79.966, 79.96633: 79.966, 79.966331: 79.966,
    0.984: 0.984, 0.984016: 0.984,
    42.011: 42.011, 42.010565: 42.011,
    43.006: 43.006, 43.005814: 43.006,
    -17.027: -17.027, -17.026549: -17.027,
}

ENGINE = {
    "msgfplus": {"score": "SpecEValue", "scan": "SpecId",
                 "peptide": "Peptide", "protein": "Protein"},
    "msfragger": {"score": "expectscore", "scan": "scannum",
                  "peptide": "peptide", "protein": "proteins"},
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


def fragpipe_annotated(pep: str, modinfo: str) -> str:
    """MSFragger peptide + `N M(15.9949), ...` -> inline annotated sequence."""
    if not modinfo:
        return pep
    chars = list(pep)
    for pos, _res, delta in FRAG_MOD_RE.findall(modinfo):
        i = int(pos) - 1
        if 0 <= i < len(chars):
            chars[i] = f"{chars[i]}{canon_delta(delta)}"
    prefix = "".join(canon_delta(m.group(1))
                     for m in NTERM_MOD_RE.finditer(modinfo))
    return prefix + "".join(chars)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", choices=list(ENGINE), required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--decoy-token", default="DECOY_")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--score", default=None)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--peptide", default=None)
    ap.add_argument("--protein", default=None)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    cfg = ENGINE[a.engine]
    score_col = a.score or cfg["score"]
    scan_col = a.scan or cfg["scan"]
    pep_col = a.peptide or cfg["peptide"]
    prot_col = a.protein or cfg["protein"]

    best: dict[str, dict] = {}
    n_rows = 0
    with open(a.tsv, encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if score_col not in (reader.fieldnames or []):
            raise SystemExit(f"missing column {score_col!r}; have {reader.fieldnames}")
        for r in reader:
            pep = r.get(pep_col) or ""
            if len(backbone(pep)) > a.max_len:
                continue
            try:
                score = float(r.get(score_col) or "")
            except ValueError:
                continue
            n_rows += 1
            key = r.get(scan_col) or str(n_rows)
            decoy = a.decoy_token in (r.get(prot_col) or "")
            cur = best.get(key)
            if cur is None or score < cur["score"]:
                best[key] = {"score": score, "decoy": decoy,
                             "pep": pep, "row": r}

    psms = sorted(best.values(), key=lambda x: x["score"])
    cum_t = cum_d = 0
    cut = -1
    for i, p in enumerate(psms):
        if p["decoy"]:
            cum_d += 1
        else:
            cum_t += 1
        if (cum_d + 1) / max(cum_t, 1) <= a.fdr:
            cut = i
    accepted = psms[:cut + 1] if cut >= 0 else []
    targets = [p for p in accepted if not p["decoy"]]
    threshold = psms[cut]["score"] if cut >= 0 else None

    bb_spectra: dict[str, list] = defaultdict(list)
    ann_spectra: dict[str, list] = defaultdict(list)
    ann_by_bb: dict[str, set] = defaultdict(set)
    for p in targets:
        raw = p["pep"]
        if a.engine == "msfragger":
            row = p["row"]
            raw = (row.get("modified_peptide") or row.get("Modified Peptide")
                   or fragpipe_annotated(
                       p["pep"],
                       row.get("modification_info")
                       or row.get("Assigned Modifications") or ""))
        bb = backbone(raw)
        ann = normalize_annotated(raw)
        if not bb:
            continue
        bb_spectra[bb].append(ann)
        ann_spectra[ann].append(bb)
        ann_by_bb[bb].add(ann)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.tsv).stem

    with open(out / f"{name}_unique_backbone.tsv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["backbone", "n_spectra", "annotated"])
        for bb in sorted(bb_spectra):
            w.writerow([bb, len(bb_spectra[bb]),
                        ";".join(sorted(ann_by_bb[bb]))])

    with open(out / f"{name}_unique_annotated.tsv", "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["annotated", "backbone", "n_spectra"])
        for ann in sorted(ann_spectra):
            w.writerow([ann, ann_spectra[ann][0], len(ann_spectra[ann])])

    summary = {
        "engine": a.engine, "tsv": str(a.tsv), "fdr": a.fdr,
        "max_len": a.max_len, "score_col": score_col,
        "scan_col": scan_col, "decoy_token": a.decoy_token,
        "n_psm_rows": n_rows, "n_spectra": len(psms),
        "n_accepted": len(accepted), "n_target": len(targets),
        "n_decoy": len(accepted) - len(targets),
        "threshold": threshold,
        "n_unique_backbone": len(bb_spectra),
        "n_unique_annotated": len(ann_spectra),
    }
    (out / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"[unique] {name}: target_spectra={len(targets)} "
          f"unique_backbone={len(bb_spectra)} "
          f"unique_annotated={len(ann_spectra)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
