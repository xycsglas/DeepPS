#!/usr/bin/env python3
"""Rewrite MGF TITLE fields into MSFragger/FragPipe-compatible form.

Multi-RAW MGFs reuse scan numbers per file, so MSFragger aborts with
"scans with duplicated scan numbers".  MSFragger splits TITLE on '.' into
``file.scan.scan.charge`` and ignores SCANS, so the only reliable fix is to
rewrite TITLE with a globally unique scan number.

Same rule as the nine-species v1-search ``fix_mgf_title.py``:
``TITLE=<name>.<scan>.<scan>.<charge>`` with ``scan`` = 1-based spectrum
position, so ``msfragger_scan - 1 == ordinal``.  All other fields (SCANS,
CHARGE, PEPMASS, SEQ, peaks) are kept untouched, and the input file is never
modified.

Usage:
  python make_unique_scans.py --in CE_full.mgf --out CE_msfragger.mgf
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def extract_charge(meta: dict[str, str]) -> str:
    m = re.search(r"(\d+)", meta.get("CHARGE", ""))
    return m.group(1) if m else "1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None,
                    help="base name used in TITLE (default = input file stem)")
    a = ap.parse_args()

    name = a.name or Path(a.src).stem
    # MSFragger splits TITLE on '.', so the name must not contain dots.
    name = re.sub(r"[^A-Za-z0-9_-]", "_", name)

    n = 0
    block: list[str] = []
    with open(a.src, "r", encoding="utf-8", errors="ignore") as fi, \
            open(a.out, "w", encoding="utf-8", newline="\n") as fo:
        for line in fi:
            s = line.strip()
            if s == "BEGIN IONS":
                block = [line]
                continue
            if s == "END IONS":
                meta: dict[str, str] = {}
                for bl in block:
                    st = bl.strip()
                    if "=" in st and not st.upper().startswith("BEGIN IONS"):
                        k, v = st.split("=", 1)
                        meta[k.upper()] = v.strip()
                charge = extract_charge(meta)
                n += 1
                fo.write("BEGIN IONS\n")
                fo.write(f"TITLE={name}.{n}.{n}.{charge}\n")
                for bl in block:
                    st = bl.strip()
                    if st.startswith("BEGIN IONS") or st.startswith("TITLE="):
                        continue
                    fo.write(bl)
                fo.write("END IONS\n")
                block = []
            elif block:
                block.append(line)
    print(f"done: {n} spectra -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
