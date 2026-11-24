# DeepPS:  Generative mass spectrometry simulation enables

# highly accurate de novo peptide sequencing and peptide

# database search



The official code repository for DeepPS. This repository provides the complete implementation for training and inference of DeepPS. DeepPS is a Transformer-based neural network that predicts theoretical MS/MS spectra from peptide sequences and precursor charge states. The repository also includes downstream analysis pipelines for de novo peptide rescoring, database search, 1\\% FDR evaluation, case studies, and cosine-similarity analysis on the MassIVE-KB dataset.

## 1\. Recommended directory layout

```
DeepPS/                    # this repository; run commands here
runs/v200918\\\_massivekb46/
  best\\\_model.pt                      # V20 checkpoint
piprime\\\_embedding/piprime\\\_embedding/
  model.ckpt                         # PiPrime checkpoint
  mgf\\\_to\\\_embedding\\\_batch.py          # MGF -> NPZ embedding entry
  encoder\\\_only.py                    # encoder implementation used for embeddings
  piprime\\\_utils.py                   # peak processing and configuration
  run\\\_iterative\\\_correction.py        # called in-process by rescoring
  PrimeNovo/                         # PiPrime dependency package
  depthcharge/                       # PrimeNovo dependency
MSGFPlus\\\_v20240326/
  MSGFPlus.jar                       # MS-GF+
  MzidToTsvConverter/
    MzidToTsvConverter.exe           # mzid -> TSV
v20\\\_data/                            # dataset files
  \\\*/\\\*.mgf
  \\\*/\\\*.fasta
v20\\\_output/                          # all results (created by commands)
```

\---

## 2\. Environment setup

DeepPS was developed and tested under Linux with Python 3.10 and NVIDIA CUDA-enabled GPUs.
We recommend using Conda or Miniconda to create an isolated environment.

### 2.1 Create the Conda environment

```bash
# Create a dedicated environment with the Python version used for DeepPS.
conda create -n deepps python=3.10.20 -y
conda activate deepps

# Upgrade pip before installing Python packages.
python -m pip install --upgrade pip
```

### 2.2 Install PyTorch with CUDA support

The released DeepPS experiments were run with PyTorch 2.8.0 and CUDA 12.8.
For an NVIDIA GPU with a compatible driver, install the CUDA 12.8 build of PyTorch:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

If a different CUDA version is required on your system, install the corresponding
PyTorch build from the official PyTorch installation instructions before installing
the remaining DeepPS dependencies.

### 2.3 Install the remaining dependencies

```bash
pip install -r requirements.txt
```

DeepPS uses common scientific-computing and proteomics packages including NumPy,
SciPy, pandas, PyArrow, Pyteomics, spectrum-utils, scikit-learn, PyTorch Lightning,
Matplotlib, OpenPyXL, KoinaPy and CuPy.

For CUDA 12.x systems, CuPy can also be installed explicitly from its precompiled
wheel if needed:

```bash
pip install cupy-cuda12x==14.1.1
```

### 2.4 Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

A CUDA-enabled installation should report `CUDA available: True`.

You can additionally verify the core DeepPS dependencies with:

```bash
python -c "import numpy, pandas, scipy, pyarrow, pyteomics, cupy; print('DeepPS dependencies imported successfully.')"
```

### 2.5 Tested environment

The released DeepPS model and the experiments reported in the paper were tested
with the following software environment:

| Component | Tested version |
|---|---|
| Operating system | Ubuntu Linux 6.8 (x86_64) |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime used by PyTorch | 12.8 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pandas | 2.3.3 |
| PyArrow | 24.0.0 |
| Pyteomics | 5.0 |
| PyTorch Lightning | 2.6.5 |
| spectrum-utils | 0.5.0 |
| scikit-learn | 1.7.2 |
| CuPy | 14.1.1 |
| KoinaPy | 0.0.11 |
| OpenPyXL | 3.1.5 |

The reference experiments were executed on NVIDIA GPUs. The recorded development
environment contained two NVIDIA GeForce RTX 5090 GPUs.

### 2.6 Optional external software

Some downstream evaluation pipelines require additional third-party software that
is not installed through `requirements.txt`:

- **MS-GF+** for conventional database-search comparisons.
- **MzidToTsvConverter** for converting MS-GF+ output to TSV.
- **MSFragger / FragPipe** for the MSFragger database-search baseline.
- **PiPrime / pi-PrimeNovo encoder checkpoint** for latent-spectrum embedding and
  de novo rescoring experiments.

These tools are only required for the corresponding downstream analyses; they are
not required for basic DeepPS model training or spectrum prediction.

---

## 3\. Data preprocessing: MassiveKB → average spectra → 39×46×6 NPZ

Filters peaks, averages spectra per peptide, and converts them into the grid used
by V20. The released checkpoint uses 0.05 Da filtering/averaging, 256
average-spectrum buckets, 8 workers, 5000-spectrum chunks, intensity clip 50000,
and the Q/K sequence constraint (Q and K are treated as distinct masses).

**Bash one-shot**

```bash
# Full MassiveKB pipeline: filter -> average -> shard NPZ -> merge, with the released settings.
DATA\\\_DIR=../v20\\\_data/massivekb PREFIX=massivekb\\\_82c0124b \\\\
  NUM\\\_WORKERS=8 CHUNK\\\_SIZE=5000 NUM\\\_BUCKETS=256 SEQ\\\_CONSTRAINED=qk \\\\
  bash dataset/scripts/run\\\_massivekb\\\_pipeline.sh
```

**Python step-by-step**

```bash
# 1) Filter peaks/tokens (0.05 Da tolerance, 8 workers).
python dataset/TokenFilter\\\_parallel.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train.mgf \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_filtered.mgf \\\\
  --tol 0.05 --workers 8

# 2) Average spectra per peptide on a 256-bucket grid (0.05 Da tolerance).
python dataset/AverageSpectrum\\\_diskbased.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_filtered.mgf \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_averaged.mgf \\\\
  --tol 0.05 --buckets 256

# 3) Build 39x46x6 NPZ shards with the Q/K sequence constraint.
python dataset/NPZBuilder\\\_chunked.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_averaged.mgf \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_npz \\\\
  --tol 0.05 --chunk-size 5000 --clip-intensity 50000 --workers 8 \\\\
  --sequence-constrained qk

# 4) Merge train shards into one memmap corpus.
python dataset/merge\\\_shards\\\_to\\\_memmap.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_npz \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_merged

# 5) Merge val shards into a single NPZ.
python dataset/shards\\\_to\\\_npz.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_val\\\_npz \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_val.npz

# 6) Merge test shards into a single NPZ.
python dataset/shards\\\_to\\\_npz.py \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test\\\_npz \\\\
  ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz
```

\---

## 4\. Training

The checkpoint `v200918\\\_massivekb46/best\\\_model.pt` uses 10 epochs and 8 workers;
all other hyperparameters are defaults.

**Bash one-shot**

```bash
# Train the released V20 checkpoint: 10 epochs, 8 workers.
V20\\\_OUTPUT\\\_DIR=../runs/v20\\\_train \\\\
V20\\\_TRAIN\\\_NPZ=../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_merged \\\\
V20\\\_VAL\\\_NPZ=../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_val.npz \\\\
V20\\\_TEST\\\_NPZ=../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  bash transfomer/run\\\_v20\\\_clean\\\_train.sh
```

**Python step-by-step**

```bash
# Only epochs (default 20) and workers (default 4) differ from the defaults.
python transfomer/v20\\\_clean\\\_train.py \\\\
  --v20-output-dir ../runs/v20\\\_train \\\\
  --v20-train-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_merged \\\\
  --v20-val-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_val.npz \\\\
  --v20-test-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  --v20-epochs 10 --v20-num-workers 8
```

```bash
# Generic trainer with the same 10-epoch / 8-worker recipe.
python transfomer/train.py \\\\
  --train-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train\\\_merged \\\\
  --val-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_val.npz \\\\
  --test-npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  --output-dir ../runs/v20\\\_train --epochs 10 --num-workers 8
```

\---

## 5\. Evaluation and inference

```bash
# Grid-level metrics on a test NPZ.
python transfomer/eval\\\_test\\\_npz.py \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  --out-json ../v20\\\_output/test\\\_metrics.json

# NPZ -> MGF (--postprocess\\\_mode none exports the raw grid).
python transfomer/predict\\\_npz\\\_to\\\_mgf.py \\\\
  --npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --output\\\_mgf ../v20\\\_output/predicted.mgf \\\\
  --postprocess\\\_mode none --device cuda:0

# Predict one peptide: MGF, peak list, full grid and provenance (--min-intensity filters peaks).
python transfomer/casestudy/case\\\_study.py predict \\\\
  --peptide QLEEAEEEAQR --charge 2 \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --out-dir ../v20\\\_output/casestudy/QLEEAEEEAQR\\\_z2 \\\\
  --min-intensity 0.20
```

`--npz` must be an NPZ produced by dataset construction or `make\\\_peptide\\\_npz.py`;
`--output\\\_mgf` is the output file. Single-peptide prediction is done with
`case\\\_study.py predict`: `--peptide` is the modified sequence (`C+57.021`,
`M+15.995` allowed) and `--charge` is the precursor charge; it writes
`prediction.mgf`, `peaks.tsv`, `grid.npz`, `grid.csv` and `metadata.json`, and
supports intensity filtering with `--min-intensity`.

\---

## 6\. Predicted vs real spectrum cosine

For each unique (peptide, charge), predicts one spectrum and compares it with all
real spectra of that peptide in the PiPrime embedding space.

**Bash one-shot**

```bash
# V20: predict one spectrum per unique peptide and collect all real spectra of that peptide.
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt GPU=0 \\\\
SRC=../v20\\\_data/v1-search \\\\
PIPELINE\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt \\\\
NPZ\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt/npz \\\\
  bash transfomer/eval\\\_10k\\\_gt/run\\\_10k\\\_gt\\\_per\\\_species.sh

# Prosit: same peptide set, with Prosit/Koina predictions instead of V20.
PREDICTOR=prosit SRC=../v20\\\_data/v1-search \\\\
PIPELINE\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt \\\\
NPZ\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt/npz \\\\
  bash transfomer/eval\\\_10k\\\_gt/run\\\_10k\\\_gt\\\_per\\\_species.sh
```

**Python step-by-step (mouse)**

```bash
# 0) Build the mouse peptide table -> input NPZ.
python transfomer/eval\\\_10k\\\_gt/make\\\_peptide\\\_npz.py \\\\
  --root ../v20\\\_data/v1-search --species mouse \\\\
  --out ../v20\\\_output/eval\\\_10k\\\_gt/npz\\\_mouse.npz

# 1) Predict one V20 spectrum per peptide (raw grid export).
python transfomer/predict\\\_npz\\\_to\\\_mgf.py \\\\
  --npz ../v20\\\_output/eval\\\_10k\\\_gt/npz\\\_mouse.npz \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --output\\\_mgf ../v20\\\_output/eval\\\_10k\\\_gt/mouse/predicted.mgf \\\\
  --postprocess\\\_mode none --device cuda:0

# 1') Or predict with Prosit/Koina (batch size 1500).
python transfomer/eval\\\_10k\\\_gt/prosit\\\_predict\\\_mgf.py \\\\
  --npz ../v20\\\_output/eval\\\_10k\\\_gt/npz\\\_mouse.npz \\\\
  --out ../v20\\\_output/eval\\\_10k\\\_gt/mouse/predicted.mgf --batch-size 1500

# 2) Collect every real spectrum of each peptide together with its prediction.
python transfomer/collect\\\_original\\\_with\\\_prediction.py \\\\
  --predicted-mgf ../v20\\\_output/eval\\\_10k\\\_gt/mouse/predicted.mgf \\\\
  --root-dir ../v20\\\_data/v1-search/mouse \\\\
  --output-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Grouped\\\_MGFs

# 3) Encode all grouped MGFs with the PiPrime encoder.
python ../piprime\\\_embedding/piprime\\\_embedding/mgf\\\_to\\\_embedding\\\_batch.py \\\\
  --mgf\\\_dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Grouped\\\_MGFs \\\\
  --output\\\_dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Embeddings \\\\
  --model ../piprime\\\_embedding/piprime\\\_embedding/model.ckpt

# 4) Per-peptide cosine summary.
python transfomer/batch\\\_cosine\\\_summary.py \\\\
  --mgf-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Grouped\\\_MGFs \\\\
  --npz-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Embeddings \\\\
  --output-csv ../v20\\\_output/eval\\\_10k\\\_gt/mouse/cosine\\\_summary.csv

# 5) Per-spectrum cosine details.
python transfomer/eval\\\_10k\\\_gt/cosine\\\_detail\\\_csv.py \\\\
  --mgf-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Grouped\\\_MGFs \\\\
  --npz-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse/Embeddings \\\\
  --out-dir ../v20\\\_output/eval\\\_10k\\\_gt/mouse
```

**Result matrices and average spectra**

```bash
# V20 matrix: raw+grid projections, rel 0.05/0.10/0.20, sqrt\\\_clip, no pred transform.
RUN\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt/mouse \\\\
PROJ=raw,grid RELS=0.05,0.10,0.20 EXP=sqrt\\\_clip PRED=none \\\\
ADD\\\_EXPNONE=1 GRID\\\_PROJ\\\_PRED=1 \\\\
  bash transfomer/eval\\\_10k\\\_gt/run\\\_matrix.sh

# Prosit matrix: same levels, square-root prediction transform.
RUN\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt/mouse \\\\
PROJ=raw,grid RELS=0.05,0.10,0.20 EXP=sqrt\\\_clip PRED=sqrt \\\\
ADD\\\_EXPNONE=1 GRID\\\_PROJ\\\_PRED=1 \\\\
  bash transfomer/eval\\\_10k\\\_gt/run\\\_matrix.sh

# Collect all matrix variants into one CSV.
python transfomer/eval\\\_10k\\\_gt/collect\\\_results.py \\\\
  --root ../v20\\\_output/eval\\\_10k\\\_gt/mouse \\\\
  --out ../v20\\\_output/eval\\\_10k\\\_gt/mouse/matrix\\\_summary.csv

# Build average real spectra for the same peptide groups.
RUN\\\_DIR=../v20\\\_output/eval\\\_10k\\\_gt/mouse \\\\
  bash transfomer/eval\\\_10k\\\_gt/run\\\_avg\\\_cosine.sh

# Collect the average-spectrum cosine results into one CSV.
python transfomer/eval\\\_10k\\\_gt/collect\\\_avg\\\_results.py \\\\
  --root ../v20\\\_output/eval\\\_10k\\\_gt/mouse \\\\
  --out ../v20\\\_output/eval\\\_10k\\\_gt/mouse/avg\\\_summary.csv

# Direct grid cosine on an NPZ.
python transfomer/eval\\\_10k\\\_gt/grid\\\_cosine\\\_npz.py \\\\
  --npz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_test.npz \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --out-dir ../v20\\\_output/grid\\\_cosine --gpu 0
```

\---

## 7\. De novo rescoring

Rescores PrimeNovo candidate pools with V20 and evaluates the top-1 by mass
matching. The released results use hard 5 ppm, simplified fusion α = 0.5 / grid
weight 0.5. HS-DF does not use the mby branch; it uses rel 0.10, PN weight 0.3
and `sqrtclip` experimental intensities, with two normalization variants: `raw`
and `minmax`. The bracketed save options were enabled for the released runs, so
the per-candidate `full\\\_results.csv` needed by case export is retained; remove
the brackets (`--save-full-results` / `SAVE\\\_FULL=1`) to reproduce that.

```bash
# Simplified two-term score: 0.50 postproc + 0.50 grid, alpha 0.5, hard 5 ppm.
python transfomer/scripts/rescoring/rescore\\\_simplified.py \\\\
  --parquet ../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2/v2\\\_mouse.parquet \\\\
  --mgf-dir ../v20\\\_data/90982/Mus\\\_mouse \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --piprime-model ../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
  --fusion-alpha 0.5 --grid-weight 0.5 --row-order \\\\
  --precursor-hard-threshold 5 --gpu 0 \\\\
  \\\[--save-full-results] \\\\
  --output-dir ../v20\\\_output/rescoring/mouse\\\_simplified

# HS-DF raw fusion, no mby: d/f equal-weight, rel 0.10, PN 0.3, sqrtclip, hard 5 ppm.
python transfomer/scripts/rescoring/rescore\\\_hsdf.py \\\\
  --parquet ../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2/v2\\\_mouse.parquet \\\\
  --mgf ../v20\\\_data/90982/Mus\\\_mouse/peaks.db.mgf \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --rel-spec 0.10 --rel-frag 0.10 --fusion-norm raw --tol 0.05 \\\\
  --exp-int-transform sqrtclip --pn-weight 0.3 \\\\
  --precursor-hard-threshold 5 --gpu 0 \\\\
  \\\[--save-full-results] \\\\
  --output-dir ../v20\\\_output/rescoring/mouse\\\_hsdf\\\_raw

# HS-DF min-max fusion, no mby: same settings, with within-spectrum min-max normalization of d/f.
python transfomer/scripts/rescoring/rescore\\\_hsdf.py \\\\
  --parquet ../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2/v2\\\_mouse.parquet \\\\
  --mgf ../v20\\\_data/90982/Mus\\\_mouse/peaks.db.mgf \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --rel-spec 0.10 --rel-frag 0.10 --fusion-norm minmax --tol 0.05 \\\\
  --exp-int-transform sqrtclip --pn-weight 0.3 \\\\
  --precursor-hard-threshold 5 --gpu 0 \\\\
  \\\[--save-full-results] \\\\
  --output-dir ../v20\\\_output/rescoring/mouse\\\_hsdf\\\_minmax
```

**All-species Bash suites**

```bash
# Simplified suite across all species (alpha 0.5, hard 5 ppm).
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
PIPRIME=../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
BASE=../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2 \\\\
MGF\\\_ROOT=../v20\\\_data/90982 \\\\
ALPHA=0.5 HARD\\\_THRESHOLD=5 \\\\
\\\[SAVE\\\_FULL=1] \\\\
OUTPUT\\\_BASE=../v20\\\_output/rescoring/simplified \\\\
  bash transfomer/scripts/suites/run\\\_simplified\\\_suite.sh

# HS-DF raw-fusion suite, no mby (rel 0.10, sqrtclip, PN 0.3, hard 5 ppm).
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
BASE\\\_V2=../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2 \\\\
MGF\\\_ROOT=../v20\\\_data/90982 \\\\
FUSION=raw USE\\\_MBY=0 \\\\
\\\[SAVE\\\_FULL=1] \\\\
OUTPUT\\\_BASE=../v20\\\_output/rescoring/hsdf\\\_raw \\\\
  bash transfomer/scripts/suites/run\\\_hsdf\\\_suite.sh

# HS-DF min-max-fusion suite, no mby (same remaining settings as the raw run).
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
BASE\\\_V2=../v20\\\_data/PI\\\_CasanovoV2\\\_candidates/v2 \\\\
MGF\\\_ROOT=../v20\\\_data/90982 \\\\
FUSION=minmax USE\\\_MBY=0 \\\\
\\\[SAVE\\\_FULL=1] \\\\
OUTPUT\\\_BASE=../v20\\\_output/rescoring/hsdf\\\_minmax \\\\
  bash transfomer/scripts/suites/run\\\_hsdf\\\_suite.sh
```

Both suites handle ricebean and mmazei automatically: ricebean uses a 20 ppm
precursor hard threshold plus
`--precursor-mismatch-mode fix --precursor-mismatch-ppm 20` because of the 57 Da
misalignment; mmazei uses `--row-alignment-mode occurrence\\\_aware --mgf-dir` in
the HS-DF suite (repeated scan numbers) and `--drop-seq-with-o` in the simplified
suite. Add the corresponding flags when running one species manually.

**Case export and summary**

```bash
# Export per-spectrum cases (5 ppm mass tolerance) from the min-max HS-DF full results.
python transfomer/scripts/rescoring/save\\\_predictions\\\_cases.py \\\\
  --results ../v20\\\_output/rescoring/mouse\\\_hsdf\\\_minmax/full\\\_results.csv \\\\
  --out-dir ../v20\\\_output/rescoring/cases \\\\
  --mgf-dir ../v20\\\_data/90982/Mus\\\_mouse --ppm 5

# Summarize the mainline rescoring metrics.
python transfomer/scripts/summary/summarize\\\_v2\\\_mainline.py \\\\
  --output-base ../v20\\\_output/rescoring/hsdf\\\_minmax \\\\
  --out-prefix ../v20\\\_output/rescoring/hsdf\\\_minmax\\\_summary
```

`--parquet` is the candidate parquet; `--mgf-dir`/`--mgf` are the experimental
spectra; `--piprime-model` is the PiPrime encoder; `--output-dir` is the output
directory.

\---

## 8\. No-FDR database search

Ranks database candidates for each spectrum and reports top-1/3/10 accuracy. All
branches share the same V20 prediction pass.

Branches and normalization: `--minimal` produces, at each rel level,
`d\\\_spec\\\_rel\\\*`, `f\\\_frag\\\_rel\\\*`, `m\\\_by\\\_prec\\\_rel\\\*`, `hs\\\_d\\\*\\\_f\\\*` (within-spectrum
min-max normalized average of d+f), `hs\\\_d\\\*\\\_f\\\*\\\_mby` (min-max average of
d+f+mby), `dfexp\\\_rel\\\*` (raw equal-weight d+f average, no normalization) and
`dfexp\\\_mby\\\_rel\\\*` (raw equal-weight d+f+mby average). Without `--minimal`, the
pipeline also produces `a\\\_grid\\\_nofilt`, `b\\\_grid\\\_rel\\\*`, `c\\\_spec\\\_nofilt`,
`e\\\_frag\\\_nofilt`, `m\\\_by\\\_frag`, `m\\\_by\\\_cos`, `frag\\\_prec`, `g\\\_by`, `hs\\\_by`, `g\\\_dfp`,
`hs\\\_dfp`, and the 10 `FUSION\\\_DEFS` (`g\\\_fuse`, `g\\\_fuse5`, `h\\\_score`, `h\\\_score5`,
`g\\\_bc`, `g\\\_bf`, `g\\\_cf`, `hs\\\_bc`, `hs\\\_bf`, `hs\\\_cf`); `--cross-fuse` adds every
`hs\\\_d<l1>\\\_f<l2>` cross-level combination. `--fusion-norm minmax|zscore` controls
the normalization of score-average fusions (rank fusions are unaffected), and
`--exp-transform none|sqrt\\\_clip|sqrt` controls experimental intensity
transformation. `--save-branches` saves predicted grids only for the listed
branches. The nine-species results are these 21 branches
(7 classes × 3 rel levels); the summary tables further compare the `df`, `hsdf`,
`dfmby` and `hsdfmby` families (raw vs min-max normalization × with/without
mby), while the per-spectrum CSV keeps all seven classes.

**Bash one-shot**

```bash
# Full DS pipeline for mouse: scan mods -> variant FASTA -> mass match -> score.
V1\\\_SEARCH\\\_ROOT=../v20\\\_data/v1-search \\\\
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
SPECIES=mouse OUTPUT\\\_DIR=../v20\\\_output/ds \\\\
PPM=10 BATCH\\\_SIZE=128 REL\\\_INTENSITIES=0.05,0.10,0.20 \\\\
FUSION\\\_NORM=minmax EXP\\\_TRANSFORM=sqrt\\\_clip MINIMAL=1 \\\\
  bash transfomer/ds/run\\\_pipeline.sh
```

**Python step-by-step**

```bash
# 1) Scan the MGFs for observed modification deltas.
python transfomer/ds/step1\\\_scan\\\_mgf\\\_mods.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --out-dir ../v20\\\_output/ds --config-dir ../v20\\\_output/ds/configs

# 2) Build the modification-variant FASTA from the scan configuration.
python transfomer/ds/step2\\\_build\\\_candidate\\\_fasta.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --mods-config ../v20\\\_output/ds/configs/mods\\\_config.json \\\\
  --out-fasta ../v20\\\_output/ds/data/mouse/variants\\\_mouse.fasta \\\\
  --out-dir ../v20\\\_output/ds

# 3) Match spectra to candidates within 10 ppm.
python transfomer/ds/step3\\\_match\\\_by\\\_mass.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --fasta ../v20\\\_output/ds/data/mouse/variants\\\_mouse.fasta \\\\
  --out-candidates ../v20\\\_output/ds/data/mouse/candidates\\\_mouse\\\_scans.tsv.gz \\\\
  --out-dir ../v20\\\_output/ds --ppm 10

# 4) Predict and score with V20 on the minimal branch set at rel 0.05/0.10/0.20.
python transfomer/ds/step4\\\_predict\\\_score.py \\\\
  --species mouse \\\\
  --candidates ../v20\\\_output/ds/data/mouse/candidates\\\_mouse\\\_scans.tsv.gz \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --out-dir ../v20\\\_output/ds --gpu 0 --batch-size 128 \\\\
  --minimal --rel-intensities 0.05,0.10,0.20 \\\\
  --fusion-norm minmax --exp-transform sqrt\\\_clip

# 5) Score the same candidates with Prosit/Koina.
python transfomer/ds/step4\\\_prosit\\\_score.py \\\\
  --species mouse \\\\
  --candidates ../v20\\\_output/ds/data/mouse/candidates\\\_mouse\\\_scans.tsv.gz \\\\
  --out-dir ../v20\\\_output/ds --rel-intensities 0.05,0.10,0.20 \\\\
  --batch-size 2000 --exp-transform sqrt\\\_clip

# 6) Report accuracy restricted to backbone length <= 30.
python transfomer/ds/results\\\_len30\\\_accuracy.py \\\\
  --csv ../v20\\\_output/ds/output/mouse/results\\\_step4\\\_mouse.csv \\\\
  --out ../v20\\\_output/ds/len30\\\_accuracy\\\_mouse.json
```

`--root`/`V1\\\_SEARCH\\\_ROOT` is the corpus root; `--out-dir`/`OUTPUT\\\_DIR` receives
the config, candidate FASTA, candidates TSV, scores and summaries.

\---

## 9\. 1% FDR: V20 and Prosit

Builds a target + reversed-decoy library and reports the largest prefix satisfying
`(decoy+1)/target ≤ 1%` per branch.

Branches and score combinations: the V20 five components `gc`, `d`, `f\\\_bidir`,
`f\\\_exp`, `mby` are computed in one GPU prediction pass; each branch is any
comma-separated subset of these components, and branches are separated by `;`.
The default branch set is
`d;gc;f\\\_bidir;gc,f\\\_exp;d,gc,f\\\_exp;d,f\\\_exp;d,f\\\_bidir;d,gc,f\\\_bidir;mby;d,f\\\_exp,mby`;
the released result CSVs actually contain all 10 branches × rel 0.05/0.10/0.20
(30 branch keys), and the four-method intersection uses
`d\\\_f\\\_exp\\\_mby\\\_rel020` (the `d\\\_f\\\_exp\\\_\\\*` series is also retained). Each branch is
combined with a rel level into `<branch>\\\_rel<lvl>` (e.g.
`d\\\_f\\\_exp\\\_mby\\\_rel020`); omitting `BRANCHES` (or `--branches` in Python) generates
all 10 default branches. The FDR fused score is the raw equal-weight mean of its
components, without within-spectrum normalization, so it is comparable across
spectra; `--save-full` writes
`full\\\_<branch>\\\_rel<lvl>\\\_<species>.csv.gz` for every branch, and step4 can then
recompute any new component combination with `--full` plus `--branches` without
re-running prediction. `--exp-transform none|sqrt\\\_clip|sqrt` controls experimental
intensity transformation.

Prosit scoring branches are (`d\\\_spec`, `f\\\_frag`, `m\\\_by\\\_prec`, `hs\\\_d\\\_f`,
`hs\\\_d\\\_f\\\_mby`, `dfexp`, `dfexp\\\_mby`) × rel 0/0.05/0.10/0.20, i.e. 28 branch keys.
The four-method intersection and unique-peptide comparison use `f\\\_frag\\\_rel000`.

**Bash one-shot**

```bash
# V20 1% FDR: TD library -> mass match -> predict/score -> FDR curve.
V1\\\_SEARCH\\\_ROOT=../v20\\\_data/v1-search \\\\
CKPT=../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
SPECIES=mouse OUTPUT\\\_DIR=../v20\\\_output/fdr \\\\
REL=0.05,0.10,0.20 MAX\\\_VAR\\\_MODS=3 \\\\
EXP\\\_TRANSFORM=sqrt\\\_clip BATCH\\\_SIZE=128 \\\[SAVE\\\_FULL=1] \\\\
  bash transfomer/fdr/run\\\_fdr.sh

# Recompute the <=30 aa FDR curves from the saved full tables.
OUTPUT\\\_DIR=../v20\\\_output/fdr SPECIES=mouse \\\\
  bash transfomer/fdr/run\\\_fdr\\\_len30.sh

# Prosit 1% FDR: reuse TD candidates, predict with Koina, then count.
ROOT=../v20\\\_data/v1-search SPECIES=mouse \\\\
RELS=0,0.05,0.10,0.20 \\\\
OUTPUT\\\_DIR=../v20\\\_output/fdr NOFDR\\\_CACHE=0 \\\\
  bash transfomer/fdr/run\\\_prosit\\\_fdr\\\_all.sh
```

**Python step-by-step**

```bash
# 1) Build the target+decoy FASTA (enumerating all variable-mod variants).
python transfomer/fdr/step1\\\_build\\\_td\\\_fasta.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --out-fasta ../v20\\\_output/fdr/data/mouse/mouse\\\_td.fasta \\\\
  --out-dir ../v20\\\_output/fdr/output/mouse

# 2) Match spectra to TD candidates within 10 ppm.
python transfomer/fdr/step2\\\_match\\\_td.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --fasta ../v20\\\_output/fdr/data/mouse/mouse\\\_td.fasta \\\\
  --out-candidates ../v20\\\_output/fdr/data/mouse/candidates\\\_mouse\\\_td.tsv.gz \\\\
  --out-dir ../v20\\\_output/fdr/output/mouse --ppm 10

# 3) Predict once and score all branches at rel 0.05/0.10/0.20.
python transfomer/fdr/step3\\\_score\\\_fdr.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --candidates ../v20\\\_output/fdr/data/mouse/candidates\\\_mouse\\\_td.tsv.gz \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --out-dir ../v20\\\_output/fdr/output/mouse \\\\
  --rel 0.05,0.10,0.20 \\\\
  --batch-size 128 --tol 0.05 --exp-transform sqrt\\\_clip --gpu 0 \\\\
  \\\[--save-full]

# 4) Compute the 1% FDR prefix on the full-length table.
python transfomer/fdr/step4\\\_fdr\\\_curve.py \\\\
  --species mouse --out-dir ../v20\\\_output/fdr/output/mouse --fdr-targets 0.01

# 5) Recompute the 1% FDR prefix restricted to <=30 aa from the saved full table.
python transfomer/fdr/step4\\\_fdr\\\_curve.py \\\\
  --species mouse --out-dir ../v20\\\_output/fdr/output/mouse/le30 \\\\
  --full ../v20\\\_output/fdr/output/mouse/full\\\_d\\\_f\\\_exp\\\_rel020\\\_mouse.csv.gz \\\\
  --max-len 30 --fdr-targets 0.01

# 6) Score the same TD candidates with Prosit at rel 0/0.05/0.10/0.20.
python transfomer/fdr/prosit\\\_fdr\\\_score.py \\\\
  --species mouse --root ../v20\\\_data/v1-search \\\\
  --candidates ../v20\\\_output/fdr/data/mouse/candidates\\\_mouse\\\_td.tsv.gz \\\\
  --out-dir ../v20\\\_output/fdr/output/mouse \\\\
  --rel-intensities 0,0.05,0.10,0.20 \\\\
  --batch-size 1500 --exp-transform sqrt\\\_clip
```

**Four-method peptide intersection**

```bash
# Intersect <=30 aa 1% FDR peptides from MS-GF+, MSFragger, V20 and Prosit.
python transfomer/fdr/postprocess\\\_peptide\\\_intersection.py \\\\
  --base ../v20\\\_output/fdr \\\\
  --msgf-dir ../v20\\\_output/engines/msgfplus \\\\
  --fragger-dir ../v20\\\_output/engines/msfragger \\\\
  --out-dir ../v20\\\_output/intersection --max-len 30 --fdr 0.01 \\\\
  --v20-branch d\\\_f\\\_exp\\\_mby\\\_rel020 --prosit-branch f\\\_frag\\\_rel000
```

\---

## 10\. Engine 1% FDR

```bash
# Create the engine output directory.
mkdir -p ../v20\\\_output/engines/mouse

# Build the engine target-decoy FASTA and modification list.
python transfomer/fdr\\\_full/build\\\_td\\\_fasta.py \\\\
  --fasta ../v20\\\_data/v1-search/mouse/for\\\_mouse\\\_test\\\_inscope\\\_peptide.fasta \\\\
  --mods-config transfomer/fdr\\\_full/mods\\\_config.example.json \\\\
  --out-fasta ../v20\\\_output/engines/mouse/mouse\\\_td.fasta \\\\
  --mods-out ../v20\\\_output/engines/mouse/mouse\\\_mods.txt

# MS-GF+ chain: search, convert mzid to TSV, count 1% FDR.
bash transfomer/fdr\\\_full/run\\\_msgfplus.sh mouse \\\\
  ../v20\\\_data/v1-search/mouse/for\\\_mouse\\\_test\\\_inscope\\\_peptide.fasta \\\\
  ../v20\\\_data/v1-search/mouse/high.mouse.mzML \\\\
  ../v20\\\_output/engines/mouse
```

Manual MS-GF+ equivalent (Windows / no Bash):

```bash
# Run the MS-GF+ search: 10 ppm precursor, 0 Da fragment tolerance.
java -Xmx12G -jar ../MSGFPlus\\\_v20240326/MSGFPlus.jar \\\\
  -s ../v20\\\_data/v1-search/mouse/high.mouse.mzML \\\\
  -d ../v20\\\_output/engines/mouse/mouse\\\_td.fasta \\\\
  -o ../v20\\\_output/engines/mouse/mouse.mzid \\\\
  -t 10ppm -ti 0,0 -m 3 -inst 1 -e 9 \\\\
  -minCharge 2 -maxCharge 4 \\\\
  -mod ../v20\\\_output/engines/mouse/mouse\\\_mods.txt \\\\
  -thread 8 -allowDenseCentroidedPeaks 1

# Convert the mzid result to a per-PSM TSV.
../MSGFPlus\\\_v20240326/MzidToTsvConverter/MzidToTsvConverter.exe \\\\
  -mzid:../v20\\\_output/engines/mouse/mouse.mzid \\\\
  -tsv:../v20\\\_output/engines/mouse/mouse.tsv -unroll -showDecoy

# Count unique <=30 aa targets at 1% FDR (use --scan SpecId for multi-RAW input).
python transfomer/fdr\\\_full/engine\\\_fdr.py --engine msgfplus \\\\
  --tsv ../v20\\\_output/engines/mouse/mouse.tsv \\\\
  --decoy-token rev\\\_ --fdr 0.01 --max-len 30
```

MSFragger: search in FragPipe with database
`../v20\\\_output/engines/mouse/mouse\\\_td.fasta`, decoy generation `None` and prefix
`rev\\\_`, nocleavage, isotope error 0, ±10 ppm, calibration off, then:

```bash
# Count unique <=30 aa MSFragger targets at 1% FDR.
python transfomer/fdr\\\_full/engine\\\_fdr.py --engine msfragger \\\\
  --tsv ../v20\\\_output/engines/msfragger/mouse/psm.tsv \\\\
  --decoy-token rev\\\_ --fdr 0.01 --max-len 30
```

For repeated scan numbers (multi-RAW MGF), add `--scan SpecId` for MS-GF+, or
renumber the MGF before MSFragger (section 11).

```bash
# V20 leg of the engine comparison at 1% FDR.
bash transfomer/fdr\\\_full/run\\\_v20\\\_fdr.sh
```

\---

## 11\. DeepSearch dataset 1% FDR analysis

**V20**

```bash
# DeepSearch TD dataset, V20 leg: convert -> match -> predict/score -> FDR -> unique peptides.
python transfomer/fdr/custom\\\_td\\\_1pct/run\\\_custom\\\_fdr.py \\\\
  --name AT \\\\
  --mgf ../v20\\\_data/custom/AT/AT.mgf \\\\
  --fasta ../v20\\\_data/custom/AT/Ath\\\_target\\\_decoy\\\_peptide.fasta \\\\
  --work ../v20\\\_output/custom --ppm 10 --batch-size 128 \\\\
  --rel 0.05,0.10,0.20 --fdr-targets 0.01 \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt --gpu 0
```

Default modifications are fixed C+57.021 + variable M+15.995 (the script
default). `--rel 0.05,0.10,0.20` sets the rel levels (the default). `--work` is
the output directory; `--mgf` and `--fasta` are the experimental spectra and the
target-decoy peptide FASTA.

**Prosit**

```bash
# Prefetch all candidate keys into the Prosit caches (4 shards per label).
python transfomer/fdr/custom\\\_td\\\_1pct/run\\\_custom\\\_prosit.py \\\\
  --name AT \\\\
  --mgf ../v20\\\_data/custom/AT/AT.mgf \\\\
  --candidates ../v20\\\_output/custom/AT/data/candidates\\\_AT\\\_td.tsv.gz \\\\
  --work ../v20\\\_output/custom --jobs 4 --prefetch-only

# Score the custom TD candidates with the prefetched Prosit caches.
python transfomer/fdr/custom\\\_td\\\_1pct/run\\\_custom\\\_prosit.py \\\\
  --name AT \\\\
  --mgf ../v20\\\_data/custom/AT/AT.mgf \\\\
  --candidates ../v20\\\_output/custom/AT/data/candidates\\\_AT\\\_td.tsv.gz \\\\
  --work ../v20\\\_output/custom --skip-prefetch
```

Large caches: build a SQLite index once and score with `--disk-cache`
(`--resume` after interruption):

```bash
# Convert the gzipped Prosit target cache into a disk-backed SQLite index.
python transfomer/fdr/custom\\\_td\\\_1pct/disk\\\_prosit\\\_cache.py \\\\
  ../v20\\\_output/custom/AT/data/prosit\\\_target\\\_cache.tsv.gz

# Score with the disk cache at rel 0/0.05/0.10/0.20; resume appends missing rows.
python transfomer/fdr/custom\\\_td\\\_1pct/prosit\\\_fdr\\\_score\\\_custom.py \\\\
  --species AT --mgf ../v20\\\_data/custom/AT/AT.mgf \\\\
  --candidates ../v20\\\_output/custom/AT/data/candidates\\\_AT\\\_td.tsv.gz \\\\
  --out-dir ../v20\\\_output/custom/AT/output \\\\
  --cache ../v20\\\_output/custom/AT/data/prosit\\\_target\\\_cache.tsv.gz \\\\
  --decoy-cache ../v20\\\_output/custom/AT/data/prosit\\\_decoy\\\_cache.tsv.gz \\\\
  --rel-intensities 0,0.05,0.10,0.20 \\\\
  --batch-size 1500 --exp-transform sqrt\\\_clip \\\\
  --disk-cache --resume
```

**Engines**

```bash
# Convert the TD FASTA to the plain engine FASTA (target + decoy, DECOY\\\_ prefix).
python transfomer/fdr/custom\\\_td\\\_1pct/convert\\\_td\\\_fasta.py \\\\
  --fasta ../v20\\\_data/custom/AT/Ath\\\_target\\\_decoy\\\_peptide.fasta \\\\
  --engine-only --engine-out ../v20\\\_output/custom/AT/data/AT\\\_engine.fasta

# Run the AT MS-GF+ search (default = C57 + M).
bash transfomer/fdr/custom\\\_td\\\_1pct/run\\\_msgfplus\\\_custom.sh AT \\\\
  ../v20\\\_data/custom/AT/AT.mgf \\\\
  ../v20\\\_output/custom/AT/data/AT\\\_engine.fasta \\\\
  ../v20\\\_output/custom/AT/msgfplus

# Rewrite TITLEs to unique species.scan.charge keys for MSFragger.
python transfomer/fdr/custom\\\_td\\\_1pct/make\\\_unique\\\_scans.py \\\\
  --in ../v20\\\_data/custom/AT/AT.mgf \\\\
  --out ../v20\\\_output/custom/AT/data/AT\\\_msfragger.mgf

# Count unique <=30 aa MSFragger targets at 1% FDR.
python transfomer/fdr\\\_full/engine\\\_fdr.py --engine msfragger \\\\
  --tsv ../v20\\\_output/custom/AT/msfragger/psm.tsv \\\\
  --decoy-token DECOY\\\_ --scan scannum --fdr 0.01 --max-len 30
```

MS-GF+ default modifications are C57 + M (the script default).

**Unique peptides**

```bash
# Unique <=30 aa MS-GF+ peptides at 1% FDR.
python transfomer/fdr/custom\\\_td\\\_1pct/unique\\\_peptides.py \\\\
  --engine msgfplus \\\\
  --tsv ../v20\\\_output/custom/AT/msgfplus/AT.tsv \\\\
  --decoy-token DECOY\\\_ --scan SpecId \\\\
  --out-dir ../v20\\\_output/custom/AT/unique --name msgfplus

# Unique <=30 aa V20 peptides at 1% FDR from the saved scoring table.
python transfomer/fdr/custom\\\_td\\\_1pct/unique\\\_peptides\\\_fdr\\\_csv.py \\\\
  --results ../v20\\\_output/custom/AT/output/results\\\_step3\\\_AT.csv \\\\
  --branch d\\\_f\\\_exp\\\_mby\\\_rel020 \\\\
  --name v20 --row-max-len 30 \\\\
  --out-dir ../v20\\\_output/custom/AT/unique
```

The DeepSearch V20 results also contain all 10 default branches × 3 rel levels;
the exported unique-peptide sets are `d\\\_f\\\_exp\\\_\\\*` and `d\\\_f\\\_exp\\\_mby\\\_\\\*`, and the
four-method intersection uses `d\\\_f\\\_exp\\\_mby\\\_rel020`. Prosit unique-peptide sets
are exported for `f\\\_frag\\\_rel000/005/010/020`; branches with no spectra passing
1% FDR are written as header-only files.

\---

## 12\. Case study

Predicts a single peptide and exports its grid as an MGF. The intensity filter is
applied in model output space.

```bash
# Predict one peptide and save the full 39x46x6 grid with provenance.
python transfomer/casestudy/case\\\_study.py predict \\\\
  --peptide QLEEAEEEAQR --charge 2 \\\\
  --checkpoint ../runs/v200918\\\_massivekb46/best\\\_model.pt \\\\
  --out-dir ../v20\\\_output/casestudy/QLEEAEEEAQR\\\_z2 \\\\
  --min-intensity 0.20

# Re-export an existing grid: relative-100 intensities, 0.20 filter, 0.05 Da merge.
python transfomer/casestudy/case\\\_study.py export \\\\
  --grid ../v20\\\_output/casestudy/QLEEAEEEAQR\\\_z2/grid.npz \\\\
  --out-dir ../v20\\\_output/casestudy/QLEEAEEEAQR\\\_z2\\\_export \\\\
  --min-intensity 0.20 --mz-merge-tol 0.05 --intensity-space relative100
```

Outputs: `grid.npz`, `prediction\\\_grid.npy`, `grid.csv`, `prediction.mgf`,
`peaks.tsv`, `metadata.json`, all in the given output directory.

\---

## 13\. MassiveKB-30000 embedding cosine results

**Bash one-shot**

```bash
# Full MassiveKB PiPrime cosine pipeline: sample -> prepare -> embeddings/cosine.
OUT\\\_DIR=../v20\\\_output/massivekb \\\\
TRAIN\\\_MGF\\\_XZ=../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train.mgf.xz \\\\
PIPRIME\\\_MODEL=../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
N\\\_GROUPS=30000 MIN\\\_SPECTRA=2 PREFER\\\_SPECTRA=3 WORKERS=8 \\\\
TOL\\\_FILTER=0.05 TOL\\\_AVERAGE=0.05 INTENSITY\\\_TRANSFORM=none,sqrt\\\_clip GPU=0 \\\\
  bash transfomer/massivekb\\\_pi\\\_cosine/run\\\_massivekb\\\_pi\\\_cosine.sh
```

**Python staged run**

```bash
# Stage 1: sample 30000 peptide groups (at least 2 spectra each).
python transfomer/massivekb\\\_pi\\\_cosine/run\\\_pipeline.py \\\\
  --out-dir ../v20\\\_output/massivekb \\\\
  --train-mgf-xz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train.mgf.xz \\\\
  --model ../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \\\\
  --domains none,sqrt\\\_clip --tol-filter 0.05 --tol-average 0.05 \\\\
  --workers 8 --gpu 0 --stop-after sample

# Stage 2: filter, predict, group and average spectra for the sampled groups.
python transfomer/massivekb\\\_pi\\\_cosine/run\\\_pipeline.py \\\\
  --out-dir ../v20\\\_output/massivekb \\\\
  --train-mgf-xz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train.mgf.xz \\\\
  --model ../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \\\\
  --domains none,sqrt\\\_clip --tol-filter 0.05 --tol-average 0.05 \\\\
  --workers 8 --gpu 0 --stop-after prepare

# Stage 3: embeddings, pairwise cosine and summaries for the full pipeline.
python transfomer/massivekb\\\_pi\\\_cosine/run\\\_pipeline.py \\\\
  --out-dir ../v20\\\_output/massivekb \\\\
  --train-mgf-xz ../v20\\\_data/massivekb/massivekb\\\_82c0124b\\\_train.mgf.xz \\\\
  --model ../piprime\\\_embedding/piprime\\\_embedding/model.ckpt \\\\
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \\\\
  --domains none,sqrt\\\_clip --tol-filter 0.05 --tol-average 0.05 \\\\
  --workers 8 --gpu 0 --stop-after full
```

The `prepare` stage performs filtering, Prosit prediction, grouping and average
spectra; the `full` stage performs embeddings, pairwise cosine and summaries.

\---

## 14\. Dataset sources

1. **Datasets for model training, testing and evaluation**: MassIVE-KB dataset, revised nine-species benchmark dataset (MSV000090982)
2. **Datasets for predicted-vs-real spectrum cosine similarity analysis**: revised nine-species benchmark dataset (MSV000090982)
3. **Datasets for de novo peptide sequencing**:

   revised nine-species benchmark dataset (MSV000090982)

   Snake Venoms (PXD044934)

   Candidatus "Scalindua brodae" (PXD044934)

   HeLa degradome (PXD044934)

4. **No-FDR search accuracy**:

   nine-species benchmark dataset (MSV000081382 \& **?**)

5. **Datasets and FASTA for 1% FDR result analysis**:

   nine-species benchmark dataset (MSV000081382 \& **?**)

   Arabidopsis thaliana dataset (PXD012708 \& UP000006548)

   Escherichia coli dataset (PXD000498 \& UP000000625)

   Caenorhabditis elegans dataset (PXD001364 \& UP000001940)



References:

Wang, M. et al. Assembling the community-scale discoverable human proteome. Cell Syst. 7, 412-421.e5 (2018). (**MassIVE-KB**)

Tran, N. H., Zhang, X., Xin, L., Shan, B. \& Li, M. De novo peptide sequencing by deep learning. Proc. Natl Acad. Sci. USA 114, 8247–8252 (2017). (**nine-species benchmark dataset**)

Yilmaz, M. et al. Sequence-to-sequence translation from mass spectra to peptides with a transformer model. Nat. Commun. 15, 6427 (2024). (**revised nine-species benchmark dataset**)

Zhang, H. et al. Arabidopsis proteome and the mass spectral assay library. Sci. Data 6, 278 (2019). (**Arabidopsis thaliana dataset**)

Walther, D. M. et al. Widespread proteome remodeling and aggregation in aging C. elegans. Cell 161, 919–932 (2015). (**Caenorhabditis elegans dataset**)

Schmidt, A. et al. The quantitative and condition-dependent Escherichia coli proteome. Nat. Biotechnol. 34, 104–110 (2016). (**Escherichia coli dataset**)

Eloff, K., Kalogeropoulos, K., Mabona, A. *et al.* InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments. *Nat Mach Intell* **7**, 565–579 (2025). (**Snake Venoms, Candidatus "Scalindua brodae", HeLa degradome**)

## 15\. Notes

* All commands are run under `v20\\\_0918\\\_release/`; relative paths resolve from there;

