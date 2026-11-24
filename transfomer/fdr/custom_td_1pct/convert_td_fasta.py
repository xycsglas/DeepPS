#!/usr/bin/env python3
"""Convert a peptide-level target-decoy FASTA to the v20_0918 FDR formats.

Input (DeepSearch-style benchmark FASTAs):
    >sp|...|PROT
    PEPTIDEK
    >DECOY_sp|...|PROT
    REVERSEDPEPTIDEK
* decoys are marked by a `DECOY_` header prefix; sequences are plain, 7..32 aa.

Outputs
-------
1. V20 TD FASTA (`--out`) -- *annotated modification variants*, exactly like
   fdr/step1_build_td_fasta.py:
       >TD_TGT_i MASS=<neutral> LABEL=T SEQ=<annotated>
       <annotated>
   Fixed C+57.021 on every C; variable mods (default M+15.995, up to
   --max-variable-mods) are enumerated as separate variants.  Decoys are the
   input FASTA's `DECOY_` records, annotated the same way; no decoys are
   generated, and decoy variants identical to a target variant are dropped.

2. Engine FASTA (`--engine-out`) -- PLAIN sequences with unique headers for
   MS-GF+/MSFragger:
       >TGT_i / >DECOY_j
   The engines apply C57/M themselves from their mod configuration, so no inline
   modifications here (feeding annotated sequences would break parsing).

Processing: length 7..32, standard residues only, dedup, decoy-target
collision removal (at the annotated level for V20; at the plain level for the
engine FASTA).

Usage:
  python convert_td_fasta.py --fasta input.fasta \
    --out work/<name>/data/<name>_td.fasta \
    --engine-out work/<name>/data/<name>_td_engine.fasta \
    --stats-json work/<name>/convert_stats.json
  # HeLa: add phospho
  --variable-mods "M:15.995,STY:79.96633"
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FDR_DIR = HERE.parent
TRANSFORMER_DIR = FDR_DIR.parent
DS_DIR = TRANSFORMER_DIR / "ds"
for _p in (str(DS_DIR), str(FDR_DIR), str(TRANSFORMER_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ds_common as dc                                        # noqa: E402
import fdr_common as fc                                       # noqa: E402
from step2_build_candidate_fasta import enumerate_variants    # noqa: E402

STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
WHITESPACE_RE = re.compile(r"\s+")
FIXED_MODS = {"C": ["57.021"]}


def parse_mods(spec: str) -> dict[str, list[str]]:
    """'M:15.995,STY:79.96633' -> {'M':['15.995'],'S':[...],'T':[...],'Y':[...]}"""
    out: dict[str, list[str]] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        res, delta = item.split(":", 1)
        if res.upper() == "NTERM":
            out.setdefault("NTERM", []).append(delta)
        else:
            for r in res:
                out.setdefault(r.upper(), []).append(delta)
    return out


def iter_records(path: Path):
    header = None
    parts: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].strip()
                parts = []
            elif header is not None:
                parts.append(line.strip())
    if header is not None:
        yield header, "".join(parts)


def clean_seq(raw: str) -> str:
    return WHITESPACE_RE.sub("", raw).upper().rstrip("*")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", default=None,
                    help="V20 annotated TD FASTA (required unless --engine-only)")
    ap.add_argument("--engine-only", action="store_true",
                    help="only write the plain engine FASTA; skip variant enumeration")
    ap.add_argument("--engine-out", default=None,
                    help="PLAIN engine TD FASTA with unique >TGT_i / >DECOY_j headers")
    ap.add_argument("--min-len", type=int, default=7)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--variable-mods", default="M:15.995",
                    help="comma list RES:delta (e.g. M:15.995,STY:79.96633); empty disables")
    ap.add_argument("--max-variable-mods", type=int, default=3)
    ap.add_argument("--cys-mod", type=float, default=57.021,
                    help="fixed C mass shift (0 disables; default 57.021)")
    ap.add_argument("--stats-json", default=None)
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    if not a.engine_only and not a.out:
        raise SystemExit("--out is required unless --engine-only")
    if a.engine_only and not a.engine_out:
        raise SystemExit("--engine-out is required with --engine-only")
    src = Path(a.fasta)
    out = Path(a.out) if a.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    variable = parse_mods(a.variable_mods)
    fixed_mods = {"C": [f"{a.cys_mod:g}"]} if a.cys_mod else {}
    stats = {
        "input": str(src), "output": str(out) if out else None,
        "engine_out": a.engine_out, "engine_only": a.engine_only,
        "fixed": fixed_mods, "variable": variable,
        "max_variable_mods": a.max_variable_mods,
        "min_len": a.min_len, "max_len": a.max_len,
        "records_T": 0, "records_D": 0,
        "dropped_length": 0, "dropped_residue": 0,
        "plain_target": 0, "plain_decoy": 0,
        "variant_target": 0, "variant_decoy": 0,
        "dup_target_variant": 0, "dup_decoy_variant": 0,
        "decoy_collision": 0,
    }

    # pass 1: plain targets + annotated target variants
    print("[convert] pass 1: target variants", flush=True)
    plain_targets: set[str] = set()
    target_variants: set[str] = set()
    valid_t = 0
    for header, raw in iter_records(src):
        if header.startswith("DECOY_"):
            stats["records_D"] += 1
            continue
        stats["records_T"] += 1
        seq = clean_seq(raw)
        if not (a.min_len <= len(seq) <= a.max_len):
            stats["dropped_length"] += 1
            continue
        if any(ch not in STANDARD for ch in seq):
            stats["dropped_residue"] += 1
            continue
        valid_t += 1
        plain_targets.add(seq)
        if not a.engine_only:
            for anno in enumerate_variants(seq, fixed_mods, variable,
                                           a.max_variable_mods):
                target_variants.add(anno)
    stats["plain_target"] = len(plain_targets)
    stats["variant_target"] = len(target_variants)
    stats["dup_target_variant"] = valid_t - len(target_variants)  # variants collapse
    print(f"[convert] plain targets={len(plain_targets):,} "
          f"annotated variants={len(target_variants):,}", flush=True)

    # pass 2: write V20 variants + engine plain FASTA
    print("[convert] pass 2: writing", flush=True)
    engine_fh = open(a.engine_out, "w", encoding="utf-8", newline="\n") \
        if a.engine_out else None
    decoy_variants: set[str] = set()
    all_target_variants = set(target_variants)   # frozen for collision checks
    engine_decoy: set[str] = set()
    n_t = n_d = n_et = n_ed = 0
    seen_engine_t: set[str] = set()
    try:
        with (open(out, "w", encoding="utf-8", newline="\n")
              if out is not None else contextlib.nullcontext()) as fh:
            for header, raw in iter_records(src):
                is_decoy = header.startswith("DECOY_")
                seq = clean_seq(raw)
                if not (a.min_len <= len(seq) <= a.max_len):
                    continue
                if any(ch not in STANDARD for ch in seq):
                    continue
                if not is_decoy:
                    # engine: one plain target record per base peptide
                    if engine_fh is not None and seq not in seen_engine_t:
                        seen_engine_t.add(seq)
                        engine_fh.write(f">TGT_{n_et}\n{seq}\n")
                        n_et += 1
                    if fh is not None:
                        for anno in enumerate_variants(seq, fixed_mods, variable,
                                                       a.max_variable_mods):
                            if anno not in target_variants:
                                continue
                            target_variants.remove(anno)
                            mass = fc.neutral_mass(anno)
                            fh.write(f">TD_TGT_{n_t} MASS={mass:.6f} LABEL=T SEQ={anno}\n{anno}\n")
                            n_t += 1
                else:
                    # engine: one reversed-plain decoy record (collision-checked)
                    if engine_fh is not None and seq not in plain_targets \
                            and seq not in engine_decoy:
                        engine_decoy.add(seq)
                        engine_fh.write(f">DECOY_{n_ed}\n{seq}\n")
                        n_ed += 1
                    if fh is not None:
                        for anno in enumerate_variants(seq, fixed_mods, variable,
                                                       a.max_variable_mods):
                            if anno in all_target_variants or anno in decoy_variants:
                                stats["decoy_collision"] += 1
                                continue
                            decoy_variants.add(anno)
                            mass = fc.neutral_mass(anno)
                            fh.write(f">TD_DEC_{n_d} MASS={mass:.6f} LABEL=D SEQ={anno}\n{anno}\n")
                            n_d += 1
    finally:
        if engine_fh is not None:
            engine_fh.close()

    stats["variant_target"] = n_t
    stats["variant_decoy"] = n_d
    stats["plain_target"] = n_et
    stats["plain_decoy"] = n_ed
    if not a.engine_only:
        print(f"[convert] V20: {n_t:,} target + {n_d:,} decoy annotated variants",
              flush=True)
    print(f"[convert] engine: {n_et:,} target + {n_ed:,} decoy plain records", flush=True)
    print(f"[convert] dropped: length={stats['dropped_length']:,} "
          f"residue={stats['dropped_residue']:,} collisions={stats['decoy_collision']:,}",
          flush=True)

    if a.stats_json:
        Path(a.stats_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.stats_json).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
