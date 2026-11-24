#!/usr/bin/env bash
# MS-GF+ 1% FDR for a custom MGF + target/decoy peptide FASTA.
#
# The supplied FASTA is already target-decoy (decoys are `>DECOY_...`) and is
# peptide-level, so the search is run with no cleavage (-e 9): MS-GF+ treats
# every entry as one candidate peptide.  Decoys are counted by the `DECOY_`
# prefix in the Protein column.
#
# Usage:
#   bash run_msgfplus_custom.sh <name> <mgf> <td_fasta> <out_dir>
#
# env:
#   MSGFPLUS_JAR  default E:/MSGFPLUS/MSGFPlus_v20240326/MSGFPlus.jar
#   MZID2TSV      default .../MzidToTsvConverter/MzidToTsvConverter.exe
#   THREADS       default 8
#   XMX           default 24G
#   PHOSPHO       1 = add Phospho STY as a variable mod (HeLa)
#   MODS_FILE     override the generated -mod file
set -euo pipefail

MSG="${MSGFPLUS_JAR:-E:/MSGFPLUS/MSGFPlus_v20240326/MSGFPlus.jar}"
MZ2TSV="${MZID2TSV:-E:/MSGFPLUS/MSGFPlus_v20240326/MzidToTsvConverter/MzidToTsvConverter.exe}"
THREADS="${THREADS:-8}"
XMX="${XMX:-24G}"
PHOSPHO="${PHOSPHO:-0}"
FDR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[ $# -ge 4 ] || { echo "usage: $0 <name> <mgf> <td_fasta> <out_dir>"; exit 1; }
NAME="$1"; MGF="$2"; TD_FASTA="$3"
mkdir -p "$4"
OUT="$(cd "$4" && pwd)"

MODS="${MODS_FILE:-$OUT/${NAME}_mods.txt}"
if [ ! -f "$MODS" ]; then
  {
    echo "NumMods=3"
    echo "57.021,C,fix,any,Carbamidomethyl"
    echo "15.9949,M,opt,any,Oxidation"
    if [ "$PHOSPHO" = "1" ]; then
      echo "79.96633,STY,opt,any,Phospho"
    fi
  } > "$MODS"
fi

LOG="$OUT/${NAME}_msgfplus.log"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== MS-GF+ 1% FDR: $NAME ==="
log "[1/3] search (-e 9 no-cleavage, -m 3 HCD, decoys in '$TD_FASTA')"
java -Xmx"$XMX" -jar "$MSG" \
  -s "$MGF" -d "$TD_FASTA" -o "$OUT/${NAME}.mzid" \
  -t 10ppm -ti 0,0 -e 9 -m 3 -inst 1 \
  -minCharge 2 -maxCharge 4 \
  -mod "$MODS" -thread "$THREADS" -allowDenseCentroidedPeaks 1 >>"$LOG" 2>&1

log "[2/3] mzid -> tsv (-showDecoy)"
# Windows: run the .exe directly.  Linux: use mono if the exe is present.
if [[ "$MZ2TSV" == *.exe ]] && command -v mono >/dev/null 2>&1; then
  MZ2TSV_CMD=(mono "$MZ2TSV")
else
  MZ2TSV_CMD=("$MZ2TSV")
fi
"${MZ2TSV_CMD[@]}" -mzid:"$OUT/${NAME}.mzid" -tsv:"$OUT/${NAME}.tsv" \
  -unroll -showDecoy >>"$LOG" 2>&1

log "[3/3] 1% FDR count (DECOY_ prefix, <=30aa)"
# Multi-RAW MGFs reuse ScanNum across files; SpecId (index=N) is the unique
# per-spectrum key and must be used for the best-per-spectrum dedup.
python "$FDR_DIR/../fdr_full/engine_fdr.py" --engine msgfplus \
  --tsv "$OUT/${NAME}.tsv" --decoy-token DECOY_ --fdr 0.01 --max-len 30 \
  --scan SpecId \
  | tee -a "$LOG"

log "done -> $OUT/${NAME}.tsv ; log: $LOG"
