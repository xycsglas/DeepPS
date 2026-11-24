#!/usr/bin/env python3
"""Four-method 1% FDR peptide intersection (<=30aa).

Methods
-------
* V20_0918 : results_step3_<sp>.csv, branch d_f_exp_mby_rel005
* Prosit   : results_prositfdr_<sp>.csv, branch f_frag_rel000
* MS-GF+   : <sp>.tsv, SpecEValue, decoy = "rev_" in Protein
* MSFragger: <sp>_fixed.tsv, expectscore, decoy = "rev_" in proteins

FDR conventions
---------------
V20/Prosit: rows with GT backbone <=30aa; per branch sort by fused DESC
(tie: smaller ordinal first), fdr_dt=(D+1)/T, q = suffix-min; accept q<=0.01.
MS-GF+/MSFragger: rows with backbone <=30aa; best PSM per scan by score
(lower better); sort ascending; accept the largest prefix with (D+1)/T<=0.01
(exactly fdr_full/engine_fdr.py).  These reproduce the published counts.

Outputs (in --out-dir)
----------------------
intersection_backbone_<sp>.csv   one row per backbone sequence
intersection_annotated_<sp>.csv  one row per normalized annotated sequence
unique_<sp>.csv                  method-unique peptides
summary_<sp>.csv                 metric,value
summary_all_species.csv          one row per species
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

SPECIES = ["mouse", "human", "yeast", "bacillus", "clambacteria",
           "honeybee", "mmazei", "ricebean", "tomato"]
METHODS = ["msfragger", "msgfplus", "prosit", "v20"]

FLANK_RE = re.compile(r"^[A-Z-]*\.(.*?)\.[A-Z-]*$")
BRACKET_RE = re.compile(r"\[[^\]]*\]")
NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
PAREN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)")
FRAG_MOD_RE = re.compile(r"(\d+)([A-Z])\(([+-]?\d+(?:\.\d+)?)\)")
NTERM_MOD_RE = re.compile(r"N-term\(([+-]?\d+(?:\.\d+)?)\)")
INLINE_MOD_RE = re.compile(r"([A-Z])([+-]\d+(?:\.\d+)?)")

CANON = {
    57.021: 57.021, 57.02: 57.021, 57.02146: 57.021,
    15.995: 15.995, 15.9949: 15.995, 15.994915: 15.995,
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
    # C(+57.02) / C[+57.02] / [+43.006]N-term -> inline +X
    s = re.sub(r"\(([+-]?\d+(?:\.\d+)?)\)", lambda m: canon_delta(m.group(1)), s)
    s = re.sub(r"\[([+-]?\d+(?:\.\d+)?)\]", lambda m: canon_delta(m.group(1)), s)
    # canonicalise already-inline X+57.021 etc.
    s = INLINE_MOD_RE.sub(lambda m: f"{m.group(1)}{canon_delta(m.group(2))}", s)
    return s


def fragpipe_annotated(peptide: str, modinfo: str) -> str:
    """FragPipe peptide + modification_info -> inline annotated sequence."""
    if not modinfo:
        return peptide
    chars = list(peptide)
    for pos, _res, delta in FRAG_MOD_RE.findall(modinfo):
        i = int(pos) - 1
        if 0 <= i < len(chars):
            chars[i] = f"{chars[i]}{canon_delta(delta)}"
    prefix = "".join(canon_delta(m.group(1)) for m in NTERM_MOD_RE.finditer(modinfo))
    return prefix + "".join(chars)


def accepted_v20_prosit(path: Path, branch: str, gt_by_ord: dict[int, str],
                        max_len: int, fdr_target: float):
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            try:
                ordinal = int(r.get("ordinal") or i)
            except ValueError:
                ordinal = i
            gt = (r.get("gt_seq") or gt_by_ord.get(ordinal) or "").strip()
            if len(backbone(gt)) > max_len:
                continue
            try:
                fused = float(r.get(f"{branch}_fused") or "-inf")
            except ValueError:
                fused = float("-inf")
            rows.append({
                "ordinal": ordinal,
                "label": (r.get(f"{branch}_label") or "").strip(),
                "fused": fused,
                "seq": (r.get(f"{branch}_seq") or "").strip(),
                "gt": gt,
            })

    ordered = sorted(rows, key=lambda r: (r["fused"], -r["ordinal"]), reverse=True)
    fdr_arr, cum_t, cum_d = [], 0, 0
    for r in ordered:
        if r["label"] == "D":
            cum_d += 1
        else:
            cum_t += 1
        fdr_arr.append(((cum_d + 1) / cum_t) if cum_t else float("inf"))
    q = [0.0] * len(fdr_arr)
    best = float("inf")
    for i in range(len(fdr_arr) - 1, -1, -1):
        best = min(best, fdr_arr[i])
        q[i] = best

    accepted = [r for r, qq in zip(ordered, q) if qq <= fdr_target]
    targets = [r for r in accepted if r["label"] != "D"]
    return targets, {
        "n_rows_le30": len(rows), "n_accepted": len(accepted),
        "n_target": len(targets),
        "n_decoy": len(accepted) - len(targets),
    }


def accepted_engine(path: Path, engine: str, max_len: int, fdr_target: float):
    cfg = {
        "msfragger": ("expectscore", "scannum", "peptide", "proteins"),
        "msgfplus": ("SpecEValue", "ScanNum", "Peptide", "Protein"),
    }[engine]
    score_col, scan_col, pep_col, prot_col = cfg
    best: dict[str, dict] = {}
    n_rows = 0
    with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            pep = r.get(pep_col) or ""
            if len(backbone(pep)) > max_len:
                continue
            try:
                score = float(r.get(score_col) or "")
            except ValueError:
                continue
            n_rows += 1
            key = r.get(scan_col) or str(n_rows)
            decoy = "rev_" in (r.get(prot_col) or "")
            cur = best.get(key)
            if cur is None or score < cur["score"]:
                best[key] = {"score": score, "decoy": decoy, "pep": pep, "row": r}

    psms = sorted(best.values(), key=lambda x: x["score"])
    cum_t = cum_d = 0
    cut = -1
    for i, p in enumerate(psms):
        if p["decoy"]:
            cum_d += 1
        else:
            cum_t += 1
        if (cum_d + 1) / max(cum_t, 1) <= fdr_target:
            cut = i
    accepted = psms[:cut + 1] if cut >= 0 else []
    targets = [p for p in accepted if not p["decoy"]]
    return targets, {
        "n_rows_le30": n_rows, "n_spectra": len(psms),
        "n_accepted": len(accepted), "n_target": len(targets),
        "n_decoy": len(accepted) - len(targets),
    }


def add_sets(sets: dict, ann_maps: dict, method: str, targets, annotate):
    """annotate(target) -> (backbone, normalized annotated)."""
    for p in targets:
        bb, ann = annotate(p)
        if not bb:
            continue
        sets[method].add(bb)
        ann_maps[method][bb].add(ann)


def write_intersection(path: Path, sets: dict, ann_maps: dict, available: list[str]):
    union = set()
    for m in available:
        union |= sets[m]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone"] + [f"in_{m}" for m in METHODS] + ["n_methods"]
                   + [f"annotated_{m}" for m in METHODS])
        for bb in sorted(union):
            flags = []
            anns = []
            n = 0
            for m in METHODS:
                if m not in available:
                    flags.append("")
                    anns.append("")
                    continue
                hit = int(bb in sets[m])
                flags.append(hit)
                n += hit
                anns.append(";".join(sorted(ann_maps[m].get(bb, ()))))
            w.writerow([bb] + flags + [n] + anns)


def write_annotated(path: Path, ann_sets: dict, available: list[str]):
    union = set()
    for m in available:
        union |= ann_sets[m]
    rows = []
    for ann in union:
        bb = backbone(ann)
        flags = [int(ann in ann_sets[m]) if m in available else "" for m in METHODS]
        n = sum(f for f in flags if isinstance(f, int))
        rows.append((bb, ann, flags, n))
    rows.sort(key=lambda x: (x[0], x[1]))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "annotated"] + [f"in_{m}" for m in METHODS] + ["n_methods"])
        for bb, ann, flags, n in rows:
            w.writerow([bb, ann] + flags + [n])


def write_unique(path: Path, sets: dict, ann_maps: dict, available: list[str]):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "backbone", "annotated"])
        for m in available:
            others = set()
            for o in available:
                if o != m:
                    others |= sets[o]
            for bb in sorted(sets[m] - others):
                w.writerow([m, bb, ";".join(sorted(ann_maps[m].get(bb, ())))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=r"C:\Users\admin\Desktop\v20_0918_39466_最新数据\1%FDR")
    ap.add_argument("--msgf-dir", default=r"C:\Users\admin\Desktop\FDR1%\results\MSGF+")
    ap.add_argument("--fragger-dir", default=r"C:\Users\admin\Desktop\FDR1%\results\MSFragger")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-len", type=int, default=30)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--v20-branch", default="d_f_exp_mby_rel005")
    ap.add_argument("--prosit-branch", default="f_frag_rel000")
    ap.add_argument("--species", default=",".join(SPECIES))
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out_dir) if args.out_dir else base / "in"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for sp in [s.strip() for s in args.species.split(",") if s.strip()]:
        v20_path = base / "result_all_9speciesFDR1%_v200918" / f"results_step3_{sp}.csv"
        prosit_path = base / "prosit" / sp / f"results_prositfdr_{sp}.csv"
        msgf_path = Path(args.msgf_dir) / f"{sp}.tsv"
        fragger_path = Path(args.fragger_dir) / f"{sp}_fixed.tsv"
        if not fragger_path.exists():
            fragger_path = Path(args.fragger_dir) / f"{sp}10000_fixed.tsv"

        gt_by_ord: dict[int, str] = {}
        if v20_path.exists():
            with open(v20_path, encoding="utf-8", newline="") as fh:
                for i, r in enumerate(csv.DictReader(fh)):
                    try:
                        o = int(r.get("ordinal") or i)
                    except ValueError:
                        o = i
                    gt = (r.get("gt_seq") or "").strip()
                    if gt:
                        gt_by_ord[o] = gt

        sets = {m: set() for m in METHODS}
        ann_maps = {m: defaultdict(set) for m in METHODS}
        stats = {}
        available = []

        if fragger_path.exists():
            targets, stats["msfragger"] = accepted_engine(fragger_path, "msfragger", args.max_len, args.fdr)
            add_sets(sets, ann_maps, "msfragger", targets,
                     lambda p: (backbone(p["pep"]), normalize_annotated(
                         fragpipe_annotated(p["pep"], (p["row"].get("modification_info") or "").strip()))))
            available.append("msfragger")
        else:
            print(f"{sp}: MSFragger results missing -> 3-method intersection", flush=True)

        if msgf_path.exists():
            targets, stats["msgfplus"] = accepted_engine(msgf_path, "msgfplus", args.max_len, args.fdr)
            add_sets(sets, ann_maps, "msgfplus", targets,
                     lambda p: (backbone(p["pep"]), normalize_annotated(p["pep"])))
            available.append("msgfplus")

        if prosit_path.exists() and gt_by_ord:
            targets, stats["prosit"] = accepted_v20_prosit(
                prosit_path, args.prosit_branch, gt_by_ord, args.max_len, args.fdr)
            add_sets(sets, ann_maps, "prosit", targets,
                     lambda p: (backbone(p["seq"]), normalize_annotated(p["seq"])))
            available.append("prosit")

        if v20_path.exists():
            targets, stats["v20"] = accepted_v20_prosit(
                v20_path, args.v20_branch, gt_by_ord, args.max_len, args.fdr)
            add_sets(sets, ann_maps, "v20", targets,
                     lambda p: (backbone(p["seq"]), normalize_annotated(p["seq"])))
            available.append("v20")

        write_intersection(out_dir / f"intersection_backbone_{sp}.csv",
                           sets, ann_maps, available)
        ann_sets = {m: {a for bbs in ann_maps[m].values() for a in bbs}
                    for m in METHODS}
        write_annotated(out_dir / f"intersection_annotated_{sp}.csv",
                        ann_sets, available)
        write_unique(out_dir / f"unique_{sp}.csv", sets, ann_maps, available)

        union = set().union(*[sets[m] for m in available]) if available else set()
        summ = {"n_available_methods": len(available)}
        for m in METHODS:
            # Final 1%FDR identification count = unique peptides after dedup.
            # Report both peptide definitions: backbone (mods stripped) and
            # normalized annotated (C57/Mox etc. kept).
            n_bb = len(sets[m]) if m in available else ""
            n_ann = len(ann_sets[m]) if m in available else ""
            n_spec = stats.get(m, {}).get("n_target", "") if m in available else ""
            summ[f"n_unique_backbone_{m}"] = n_bb
            summ[f"n_unique_annotated_{m}"] = n_ann
            summ[f"n_accepted_target_spectra_{m}"] = n_spec
            if isinstance(n_ann, int) and n_ann:
                summ[f"spectra_per_unique_annotated_{m}"] = round(n_spec / n_ann, 3)
            summ[f"missing_in_{m}"] = (len(union - sets[m]) if m in available else "")
        summ["n_backbone_union"] = len(union)
        for k in range(len(available), 0, -1):
            summ[f"in_{k}_methods"] = sum(
                1 for bb in union
                if sum(1 for m in available if bb in sets[m]) == k)

        ann_union = set().union(*[ann_sets[m] for m in available]) if available else set()
        summ["n_annotated_union"] = len(ann_union)
        for k in range(len(available), 0, -1):
            summ[f"in_{k}_methods_annotated"] = sum(
                1 for a in ann_union
                if sum(1 for m in available if a in ann_sets[m]) == k)
        for a in range(len(METHODS)):
            for b in range(a + 1, len(METHODS)):
                ma, mb = METHODS[a], METHODS[b]
                if ma in available and mb in available:
                    summ[f"pair_{ma}_{mb}"] = len(sets[ma] & sets[mb])
        for m in available:
            for k, v in stats.get(m, {}).items():
                summ[f"{m}_{k}"] = v

        with open(out_dir / f"summary_{sp}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "value"])
            for k, v in summ.items():
                w.writerow([k, v])

        all_rows.append({"species": sp, "available": ";".join(available), **summ})
        print(f"{sp}: " + " ".join(
            f"{m}={len(sets[m]) if m in available else 'NA'}" for m in METHODS)
            + f" union={len(union)}", flush=True)

    keys = ["species", "available", "n_available_methods",
            "n_unique_backbone_msfragger", "n_unique_backbone_msgfplus",
            "n_unique_backbone_prosit", "n_unique_backbone_v20",
            "n_unique_annotated_msfragger", "n_unique_annotated_msgfplus",
            "n_unique_annotated_prosit", "n_unique_annotated_v20",
            "n_accepted_target_spectra_msfragger", "n_accepted_target_spectra_msgfplus",
            "n_accepted_target_spectra_prosit", "n_accepted_target_spectra_v20",
            "n_backbone_union", "in_4_methods", "in_3_methods",
            "in_2_methods", "in_1_methods",
            "n_annotated_union", "in_4_methods_annotated",
            "in_3_methods_annotated", "in_2_methods_annotated",
            "in_1_methods_annotated"]
    with open(out_dir / "summary_all_species.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
