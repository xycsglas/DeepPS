# DeepPS:  Generative mass spectrometry simulation enables highly accurate de novo peptide sequencing and peptide database search
The official code repository for DeepPS. This repository provides the complete implementation for training and inference of DeepPS. DeepPS is a Transformer-based neural network that predicts theoretical MS/MS spectra from peptide sequences and precursor charge states. The repository also includes downstream analysis pipelines for de novo peptide rescoring, database search, 1% FDR evaluation, case studies, and cosine-similarity analysis on the MassIVE-KB dataset.

<img width="2369" height="1983" alt="Github" src="https://github.com/user-attachments/assets/2004fc6c-def3-4129-8c70-1eb6520f6dd2" />



## 1. Recommended directory layout

```text
DeepPS/                              # this repository; run commands here
runs/DeepPS_massivekb46/
  best_model.pt                      # DeepPS checkpoint

piprime_embedding/piprime_embedding/
  model.ckpt                         # PiPrime checkpoint
  mgf_to_embedding_batch.py          # MGF -> NPZ embedding entry
  encoder_only.py                    # encoder implementation used for embeddings
  piprime_utils.py                   # peak processing and configuration
  run_iterative_correction.py        # called in-process by rescoring
  PrimeNovo/                         # PiPrime dependency package
  depthcharge/                       # PrimeNovo dependency

MSGFPlus_deepps240326/
  MSGFPlus.jar                       # MS-GF+
  MzidToTsvConverter/
    MzidToTsvConverter.exe           # mzid -> TSV

DeepPS_data/                         # dataset files
  */*.mgf
  */*.fasta

DeepPS_output/                       # all results created by DeepPS pipelines
```

---

## 2. Environment setup

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
### 2.7 Download model checkpoint of DeepPS
https://drive.google.com/file/d/1_TLpY7NFlfVuJdubj8fXIpZV-iB9yX77/view?usp=sharing

## 3. Data preprocessing: MassiveKB → average spectra → 39×46×6 NPZ

Filters peaks, averages spectra per peptide, and converts them into the grid used
by DeepPS. The released checkpoint uses 0.05 Da filtering/averaging, 256
average-spectrum buckets, 8 workers, 5000-spectrum chunks, intensity clip 50000,
and the Q/K sequence constraint (Q and K are treated as distinct masses).

**Bash one-shot**

```bash
# Full MassiveKB pipeline: filter -> average -> shard NPZ -> merge, with the released settings.
DATA_DIR=../DeepPS_data/massivekb PREFIX=massivekb_82c0124b \
  NUM_WORKERS=8 CHUNK_SIZE=5000 NUM_BUCKETS=256 SEQ_CONSTRAINED=qk \
  bash dataset/scripts/run_massivekb_pipeline.sh
```

**Python step-by-step**

```bash
# 1) Filter peaks/tokens (0.05 Da tolerance, 8 workers).
python dataset/TokenFilter_parallel.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train.mgf \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_filtered.mgf \
  --tol 0.05 --workers 8

# 2) Average spectra per peptide on a 256-bucket grid (0.05 Da tolerance).
python dataset/AverageSpectrum_diskbased.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_filtered.mgf \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_averaged.mgf \
  --tol 0.05 --buckets 256

# 3) Build 39x46x6 NPZ shards with the Q/K sequence constraint.
python dataset/NPZBuilder_chunked.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_averaged.mgf \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_npz \
  --tol 0.05 --chunk-size 5000 --clip-intensity 50000 --workers 8 \
  --sequence-constrained qk

# 4) Merge train shards into one memmap corpus.
python dataset/merge_shards_to_memmap.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_npz \
  ../DeepPS_data/massivekb/massivekb_82c0124b_train_merged

# 5) Merge val shards into a single NPZ.
python dataset/shards_to_npz.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_val_npz \
  ../DeepPS_data/massivekb/massivekb_82c0124b_val.npz

# 6) Merge test shards into a single NPZ.
python dataset/shards_to_npz.py \
  ../DeepPS_data/massivekb/massivekb_82c0124b_test_npz \
  ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz
```

---
## 4. Training

The checkpoint `DeepPS_massivekb46/best_model.pt` uses 10 epochs and 8 workers;
all other hyperparameters are defaults.

**Bash one-shot**

```bash
# Train the released DeepPS checkpoint: 10 epochs, 8 workers.
DEEPPS_OUTPUT_DIR=../runs/DeepPS_train \
DEEPPS_TRAIN_NPZ=../DeepPS_data/massivekb/massivekb_82c0124b_train_merged \
DEEPPS_VAL_NPZ=../DeepPS_data/massivekb/massivekb_82c0124b_val.npz \
DEEPPS_TEST_NPZ=../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  bash transfomer/run_deepps_clean_train.sh
```

**Python step-by-step**

```bash
# Only epochs (default 20) and workers (default 4) differ from the defaults.
python transfomer/deepps_clean_train.py \
  --deepps-output-dir ../runs/DeepPS_train \
  --deepps-train-npz ../DeepPS_data/massivekb/massivekb_82c0124b_train_merged \
  --deepps-val-npz ../DeepPS_data/massivekb/massivekb_82c0124b_val.npz \
  --deepps-test-npz ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  --deepps-epochs 10 --deepps-num-workers 8
```

```bash
# Generic trainer with the same 10-epoch / 8-worker recipe.
python transfomer/train.py \
  --train-npz ../DeepPS_data/massivekb/massivekb_82c0124b_train_merged \
  --val-npz ../DeepPS_data/massivekb/massivekb_82c0124b_val.npz \
  --test-npz ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  --output-dir ../runs/DeepPS_train --epochs 10 --num-workers 8
```

---
## 5. Evaluation and inference

```bash
# Grid-level metrics on a test NPZ.
python transfomer/eval_test_npz.py \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --npz ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  --out-json ../DeepPS_output/test_metrics.json

# NPZ -> MGF (--postprocess_mode none exports the raw grid).
python transfomer/predict_npz_to_mgf.py \
  --npz ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --output_mgf ../DeepPS_output/predicted.mgf \
  --postprocess_mode none --device cuda:0

# Predict one peptide: MGF, peak list, full grid and provenance (--min-intensity filters peaks).
python transfomer/casestudy/case_study.py predict \
  --peptide QLEEAEEEAQR --charge 2 \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --out-dir ../DeepPS_output/casestudy/QLEEAEEEAQR_z2 \
  --min-intensity 0.20
```

`--npz` must be an NPZ produced by dataset construction or `make_peptide_npz.py`;
`--output_mgf` is the output file. Single-peptide prediction is done with
`case_study.py predict`: `--peptide` is the modified sequence (`C+57.021`,
`M+15.995` allowed) and `--charge` is the precursor charge; it writes
`prediction.mgf`, `peaks.tsv`, `grid.npz`, `grid.csv` and `metadata.json`, and
supports intensity filtering with `--min-intensity`.

---
## 6. Predicted vs real spectrum cosine

For each unique (peptide, charge), predicts one spectrum and compares it with all
real spectra of that peptide in the PiPrime embedding space.

**Bash one-shot**

```bash
# DeepPS: predict one spectrum per unique peptide and collect all real spectra of that peptide.
CKPT=../runs/DeepPS_massivekb46/best_model.pt GPU=0 \
SRC=../DeepPS_data/v1-search \
PIPELINE_DIR=../DeepPS_output/eval_10k_gt \
NPZ_DIR=../DeepPS_output/eval_10k_gt/npz \
  bash transfomer/eval_10k_gt/run_10k_gt_per_species.sh

# Prosit: same peptide set, with Prosit/Koina predictions instead of DeepPS.
PREDICTOR=prosit SRC=../DeepPS_data/v1-search \
PIPELINE_DIR=../DeepPS_output/eval_10k_gt \
NPZ_DIR=../DeepPS_output/eval_10k_gt/npz \
  bash transfomer/eval_10k_gt/run_10k_gt_per_species.sh
```

**Python step-by-step (mouse)**

```bash
# 0) Build the mouse peptide table -> input NPZ.
python transfomer/eval_10k_gt/make_peptide_npz.py \
  --root ../DeepPS_data/v1-search --species mouse \
  --out ../DeepPS_output/eval_10k_gt/npz_mouse.npz

# 1) Predict one DeepPS spectrum per peptide (raw grid export).
python transfomer/predict_npz_to_mgf.py \
  --npz ../DeepPS_output/eval_10k_gt/npz_mouse.npz \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --output_mgf ../DeepPS_output/eval_10k_gt/mouse/predicted.mgf \
  --postprocess_mode none --device cuda:0

# 1') Or predict with Prosit/Koina (batch size 1500).
python transfomer/eval_10k_gt/prosit_predict_mgf.py \
  --npz ../DeepPS_output/eval_10k_gt/npz_mouse.npz \
  --out ../DeepPS_output/eval_10k_gt/mouse/predicted.mgf --batch-size 1500

# 2) Collect every real spectrum of each peptide together with its prediction.
python transfomer/collect_original_with_prediction.py \
  --predicted-mgf ../DeepPS_output/eval_10k_gt/mouse/predicted.mgf \
  --root-dir ../DeepPS_data/v1-search/mouse \
  --output-dir ../DeepPS_output/eval_10k_gt/mouse/Grouped_MGFs

# 3) Encode all grouped MGFs with the PiPrime encoder.
python ../piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py \
  --mgf_dir ../DeepPS_output/eval_10k_gt/mouse/Grouped_MGFs \
  --output_dir ../DeepPS_output/eval_10k_gt/mouse/Embeddings \
  --model ../piprime_embedding/piprime_embedding/model.ckpt

# 4) Per-peptide cosine summary.
python transfomer/batch_cosine_summary.py \
  --mgf-dir ../DeepPS_output/eval_10k_gt/mouse/Grouped_MGFs \
  --npz-dir ../DeepPS_output/eval_10k_gt/mouse/Embeddings \
  --output-csv ../DeepPS_output/eval_10k_gt/mouse/cosine_summary.csv

# 5) Per-spectrum cosine details.
python transfomer/eval_10k_gt/cosine_detail_csv.py \
  --mgf-dir ../DeepPS_output/eval_10k_gt/mouse/Grouped_MGFs \
  --npz-dir ../DeepPS_output/eval_10k_gt/mouse/Embeddings \
  --out-dir ../DeepPS_output/eval_10k_gt/mouse
```

**Result matrices and average spectra**

```bash
# DeepPS matrix: raw+grid projections, rel 0.05/0.10/0.20, sqrt_clip, no pred transform.
RUN_DIR=../DeepPS_output/eval_10k_gt/mouse \
PROJ=raw,grid RELS=0.05,0.10,0.20 EXP=sqrt_clip PRED=none \
ADD_EXPNONE=1 GRID_PROJ_PRED=1 \
  bash transfomer/eval_10k_gt/run_matrix.sh

# Prosit matrix: same levels, square-root prediction transform.
RUN_DIR=../DeepPS_output/eval_10k_gt/mouse \
PROJ=raw,grid RELS=0.05,0.10,0.20 EXP=sqrt_clip PRED=sqrt \
ADD_EXPNONE=1 GRID_PROJ_PRED=1 \
  bash transfomer/eval_10k_gt/run_matrix.sh

# Collect all matrix variants into one CSV.
python transfomer/eval_10k_gt/collect_results.py \
  --root ../DeepPS_output/eval_10k_gt/mouse \
  --out ../DeepPS_output/eval_10k_gt/mouse/matrix_summary.csv

# Build average real spectra for the same peptide groups.
RUN_DIR=../DeepPS_output/eval_10k_gt/mouse \
  bash transfomer/eval_10k_gt/run_avg_cosine.sh

# Collect the average-spectrum cosine results into one CSV.
python transfomer/eval_10k_gt/collect_avg_results.py \
  --root ../DeepPS_output/eval_10k_gt/mouse \
  --out ../DeepPS_output/eval_10k_gt/mouse/avg_summary.csv

# Direct grid cosine on an NPZ.
python transfomer/eval_10k_gt/grid_cosine_npz.py \
  --npz ../DeepPS_data/massivekb/massivekb_82c0124b_test.npz \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --out-dir ../DeepPS_output/grid_cosine --gpu 0
```

---
## 7. De novo rescoring

Rescores PrimeNovo candidate pools with DeepPS and evaluates the top-1 by mass
matching. The released results use hard 5 ppm, simplified fusion α = 0.5 / grid
weight 0.5. HS-DF does not use the mby branch; it uses rel 0.10, PN weight 0.3
and `sqrtclip` experimental intensities, with two normalization variants: `raw`
and `minmax`. The bracketed save options were enabled for the released runs, so
the per-candidate `full_results.csv` needed by case export is retained; remove
the brackets (`--save-full-results` / `SAVE_FULL=1`) to reproduce that.

```bash
# Simplified two-term score: 0.50 postproc + 0.50 grid, alpha 0.5, hard 5 ppm.
python transfomer/scripts/rescoring/rescore_simplified.py \
  --parquet ../DeepPS_data/PI_CasanovoV2_candidates/v2/v2_mouse.parquet \
  --mgf-dir ../DeepPS_data/90982/Mus_mouse \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --piprime-model ../piprime_embedding/piprime_embedding/model.ckpt \
  --fusion-alpha 0.5 --grid-weight 0.5 --row-order \
  --precursor-hard-threshold 5 --gpu 0 \
  [--save-full-results] \
  --output-dir ../DeepPS_output/rescoring/mouse_simplified

# HS-DF raw fusion, no mby: d/f equal-weight, rel 0.10, PN 0.3, sqrtclip, hard 5 ppm.
python transfomer/scripts/rescoring/rescore_hsdf.py \
  --parquet ../DeepPS_data/PI_CasanovoV2_candidates/v2/v2_mouse.parquet \
  --mgf ../DeepPS_data/90982/Mus_mouse/peaks.db.mgf \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --rel-spec 0.10 --rel-frag 0.10 --fusion-norm raw --tol 0.05 \
  --exp-int-transform sqrtclip --pn-weight 0.3 \
  --precursor-hard-threshold 5 --gpu 0 \
  [--save-full-results] \
  --output-dir ../DeepPS_output/rescoring/mouse_hsdf_raw

# HS-DF min-max fusion, no mby: same settings, with within-spectrum min-max normalization of d/f.
python transfomer/scripts/rescoring/rescore_hsdf.py \
  --parquet ../DeepPS_data/PI_CasanovoV2_candidates/v2/v2_mouse.parquet \
  --mgf ../DeepPS_data/90982/Mus_mouse/peaks.db.mgf \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --rel-spec 0.10 --rel-frag 0.10 --fusion-norm minmax --tol 0.05 \
  --exp-int-transform sqrtclip --pn-weight 0.3 \
  --precursor-hard-threshold 5 --gpu 0 \
  [--save-full-results] \
  --output-dir ../DeepPS_output/rescoring/mouse_hsdf_minmax
```

**All-species Bash suites**

```bash
# Simplified suite across all species (alpha 0.5, hard 5 ppm).
CKPT=../runs/DeepPS_massivekb46/best_model.pt \
PIPRIME=../piprime_embedding/piprime_embedding/model.ckpt \
BASE=../DeepPS_data/PI_CasanovoV2_candidates/v2 \
MGF_ROOT=../DeepPS_data/90982 \
ALPHA=0.5 HARD_THRESHOLD=5 \
[SAVE_FULL=1] \
OUTPUT_BASE=../DeepPS_output/rescoring/simplified \
  bash transfomer/scripts/suites/run_simplified_suite.sh

# HS-DF raw-fusion suite, no mby (rel 0.10, sqrtclip, PN 0.3, hard 5 ppm).
CKPT=../runs/DeepPS_massivekb46/best_model.pt \
BASE_V2=../DeepPS_data/PI_CasanovoV2_candidates/v2 \
MGF_ROOT=../DeepPS_data/90982 \
FUSION=raw USE_MBY=0 \
[SAVE_FULL=1] \
OUTPUT_BASE=../DeepPS_output/rescoring/hsdf_raw \
  bash transfomer/scripts/suites/run_hsdf_suite.sh

# HS-DF min-max-fusion suite, no mby (same remaining settings as the raw run).
CKPT=../runs/DeepPS_massivekb46/best_model.pt \
BASE_V2=../DeepPS_data/PI_CasanovoV2_candidates/v2 \
MGF_ROOT=../DeepPS_data/90982 \
FUSION=minmax USE_MBY=0 \
[SAVE_FULL=1] \
OUTPUT_BASE=../DeepPS_output/rescoring/hsdf_minmax \
  bash transfomer/scripts/suites/run_hsdf_suite.sh
```

Both suites handle ricebean and mmazei automatically: ricebean uses a 20 ppm
precursor hard threshold plus
`--precursor-mismatch-mode fix --precursor-mismatch-ppm 20` because of the 57 Da
misalignment; mmazei uses `--row-alignment-mode occurrence_aware --mgf-dir` in
the HS-DF suite (repeated scan numbers) and `--drop-seq-with-o` in the simplified
suite. Add the corresponding flags when running one species manually.

**Case export and summary**

```bash
# Export per-spectrum cases (5 ppm mass tolerance) from the min-max HS-DF full results.
python transfomer/scripts/rescoring/save_predictions_cases.py \
  --results ../DeepPS_output/rescoring/mouse_hsdf_minmax/full_results.csv \
  --out-dir ../DeepPS_output/rescoring/cases \
  --mgf-dir ../DeepPS_data/90982/Mus_mouse --ppm 5

# Summarize the mainline rescoring metrics.
python transfomer/scripts/summary/summarize_v2_mainline.py \
  --output-base ../DeepPS_output/rescoring/hsdf_minmax \
  --out-prefix ../DeepPS_output/rescoring/hsdf_minmax_summary
```

`--parquet` is the candidate parquet; `--mgf-dir`/`--mgf` are the experimental
spectra; `--piprime-model` is the PiPrime encoder; `--output-dir` is the output
directory.

---
## 8. No-FDR database search

Ranks database candidates for each spectrum and reports top-1/3/10 accuracy. All
branches share the same DeepPS prediction pass.

Branches and normalization: `--minimal` produces, at each rel level,
`d_spec_rel*`, `f_frag_rel*`, `m_by_prec_rel*`, `hs_d*_f*` (within-spectrum
min-max normalized average of d+f), `hs_d*_f*_mby` (min-max average of
d+f+mby), `dfexp_rel*` (raw equal-weight d+f average, no normalization) and
`dfexp_mby_rel*` (raw equal-weight d+f+mby average). Without `--minimal`, the
pipeline also produces `a_grid_nofilt`, `b_grid_rel*`, `c_spec_nofilt`,
`e_frag_nofilt`, `m_by_frag`, `m_by_cos`, `frag_prec`, `g_by`, `hs_by`, `g_dfp`,
`hs_dfp`, and the 10 `FUSION_DEFS` (`g_fuse`, `g_fuse5`, `h_score`, `h_score5`,
`g_bc`, `g_bf`, `g_cf`, `hs_bc`, `hs_bf`, `hs_cf`); `--cross-fuse` adds every
`hs_d<l1>_f<l2>` cross-level combination. `--fusion-norm minmax|zscore` controls
the normalization of score-average fusions (rank fusions are unaffected), and
`--exp-transform none|sqrt_clip|sqrt` controls experimental intensity
transformation. `--save-branches` saves predicted grids only for the listed
branches. The nine-species results are these 21 branches
(7 classes × 3 rel levels); the summary tables further compare the `df`, `hsdf`,
`dfmby` and `hsdfmby` families (raw vs min-max normalization × with/without
mby), while the per-spectrum CSV keeps all seven classes.

**Bash one-shot**

```bash
# Full DS pipeline for mouse: scan mods -> variant FASTA -> mass match -> score.
V1_SEARCH_ROOT=../DeepPS_data/v1-search \
CKPT=../runs/DeepPS_massivekb46/best_model.pt \
SPECIES=mouse OUTPUT_DIR=../DeepPS_output/ds \
PPM=10 BATCH_SIZE=128 REL_INTENSITIES=0.05,0.10,0.20 \
FUSION_NORM=minmax EXP_TRANSFORM=sqrt_clip MINIMAL=1 \
  bash transfomer/ds/run_pipeline.sh
```

**Python step-by-step**

```bash
# 1) Scan the MGFs for observed modification deltas.
python transfomer/ds/step1_scan_mgf_mods.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --out-dir ../DeepPS_output/ds --config-dir ../DeepPS_output/ds/configs

# 2) Build the modification-variant FASTA from the scan configuration.
python transfomer/ds/step2_build_candidate_fasta.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --mods-config ../DeepPS_output/ds/configs/mods_config.json \
  --out-fasta ../DeepPS_output/ds/data/mouse/variants_mouse.fasta \
  --out-dir ../DeepPS_output/ds

# 3) Match spectra to candidates within 10 ppm.
python transfomer/ds/step3_match_by_mass.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --fasta ../DeepPS_output/ds/data/mouse/variants_mouse.fasta \
  --out-candidates ../DeepPS_output/ds/data/mouse/candidates_mouse_scans.tsv.gz \
  --out-dir ../DeepPS_output/ds --ppm 10

# 4) Predict and score with DeepPS on the minimal branch set at rel 0.05/0.10/0.20.
python transfomer/ds/step4_predict_score.py \
  --species mouse \
  --candidates ../DeepPS_output/ds/data/mouse/candidates_mouse_scans.tsv.gz \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --out-dir ../DeepPS_output/ds --gpu 0 --batch-size 128 \
  --minimal --rel-intensities 0.05,0.10,0.20 \
  --fusion-norm minmax --exp-transform sqrt_clip

# 5) Score the same candidates with Prosit/Koina.
python transfomer/ds/step4_prosit_score.py \
  --species mouse \
  --candidates ../DeepPS_output/ds/data/mouse/candidates_mouse_scans.tsv.gz \
  --out-dir ../DeepPS_output/ds --rel-intensities 0.05,0.10,0.20 \
  --batch-size 2000 --exp-transform sqrt_clip

# 6) Report accuracy restricted to backbone length <= 30.
python transfomer/ds/results_len30_accuracy.py \
  --csv ../DeepPS_output/ds/output/mouse/results_step4_mouse.csv \
  --out ../DeepPS_output/ds/len30_accuracy_mouse.json
```

`--root`/`V1_SEARCH_ROOT` is the corpus root; `--out-dir`/`OUTPUT_DIR` receives
the config, candidate FASTA, candidates TSV, scores and summaries.

---
## 9. 1% FDR: DeepPS and Prosit

Builds a target + reversed-decoy library and reports the largest prefix satisfying
`(decoy+1)/target ≤ 1%` per branch.

Branches and score combinations: the DeepPS five components `gc`, `d`, `f_bidir`,
`f_exp`, `mby` are computed in one GPU prediction pass; each branch is any
comma-separated subset of these components, and branches are separated by `;`.
The default branch set is
`d;gc;f_bidir;gc,f_exp;d,gc,f_exp;d,f_exp;d,f_bidir;d,gc,f_bidir;mby;d,f_exp,mby`;
the released result CSVs actually contain all 10 branches × rel 0.05/0.10/0.20
(30 branch keys), and the four-method intersection uses
`d_f_exp_mby_rel020` (the `d_f_exp_*` series is also retained). Each branch is
combined with a rel level into `<branch>_rel<lvl>` (e.g.
`d_f_exp_mby_rel020`); omitting `BRANCHES` (or `--branches` in Python) generates
all 10 default branches. The FDR fused score is the raw equal-weight mean of its
components, without within-spectrum normalization, so it is comparable across
spectra; `--save-full` writes
`full_<branch>_rel<lvl>_<species>.csv.gz` for every branch, and step4 can then
recompute any new component combination with `--full` plus `--branches` without
re-running prediction. `--exp-transform none|sqrt_clip|sqrt` controls experimental
intensity transformation.

Prosit scoring branches are (`d_spec`, `f_frag`, `m_by_prec`, `hs_d_f`,
`hs_d_f_mby`, `dfexp`, `dfexp_mby`) × rel 0/0.05/0.10/0.20, i.e. 28 branch keys.
The four-method intersection and unique-peptide comparison use `f_frag_rel000`.

**Bash one-shot**

```bash
# DeepPS 1% FDR: TD library -> mass match -> predict/score -> FDR curve.
V1_SEARCH_ROOT=../DeepPS_data/v1-search \
CKPT=../runs/DeepPS_massivekb46/best_model.pt \
SPECIES=mouse OUTPUT_DIR=../DeepPS_output/fdr \
REL=0.05,0.10,0.20 MAX_VAR_MODS=3 \
EXP_TRANSFORM=sqrt_clip BATCH_SIZE=128 [SAVE_FULL=1] \
  bash transfomer/fdr/run_fdr.sh

# Recompute the <=30 aa FDR curves from the saved full tables.
OUTPUT_DIR=../DeepPS_output/fdr SPECIES=mouse \
  bash transfomer/fdr/run_fdr_len30.sh

# Prosit 1% FDR: reuse TD candidates, predict with Koina, then count.
ROOT=../DeepPS_data/v1-search SPECIES=mouse \
RELS=0,0.05,0.10,0.20 \
OUTPUT_DIR=../DeepPS_output/fdr NOFDR_CACHE=0 \
  bash transfomer/fdr/run_prosit_fdr_all.sh
```

**Python step-by-step**

```bash
# 1) Build the target+decoy FASTA (enumerating all variable-mod variants).
python transfomer/fdr/step1_build_td_fasta.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --out-fasta ../DeepPS_output/fdr/data/mouse/mouse_td.fasta \
  --out-dir ../DeepPS_output/fdr/output/mouse

# 2) Match spectra to TD candidates within 10 ppm.
python transfomer/fdr/step2_match_td.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --fasta ../DeepPS_output/fdr/data/mouse/mouse_td.fasta \
  --out-candidates ../DeepPS_output/fdr/data/mouse/candidates_mouse_td.tsv.gz \
  --out-dir ../DeepPS_output/fdr/output/mouse --ppm 10

# 3) Predict once and score all branches at rel 0.05/0.10/0.20.
python transfomer/fdr/step3_score_fdr.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --candidates ../DeepPS_output/fdr/data/mouse/candidates_mouse_td.tsv.gz \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --out-dir ../DeepPS_output/fdr/output/mouse \
  --rel 0.05,0.10,0.20 \
  --batch-size 128 --tol 0.05 --exp-transform sqrt_clip --gpu 0 \
  [--save-full]

# 4) Compute the 1% FDR prefix on the full-length table.
python transfomer/fdr/step4_fdr_curve.py \
  --species mouse --out-dir ../DeepPS_output/fdr/output/mouse --fdr-targets 0.01

# 5) Recompute the 1% FDR prefix restricted to <=30 aa from the saved full table.
python transfomer/fdr/step4_fdr_curve.py \
  --species mouse --out-dir ../DeepPS_output/fdr/output/mouse/le30 \
  --full ../DeepPS_output/fdr/output/mouse/full_d_f_exp_rel020_mouse.csv.gz \
  --max-len 30 --fdr-targets 0.01

# 6) Score the same TD candidates with Prosit at rel 0/0.05/0.10/0.20.
python transfomer/fdr/prosit_fdr_score.py \
  --species mouse --root ../DeepPS_data/v1-search \
  --candidates ../DeepPS_output/fdr/data/mouse/candidates_mouse_td.tsv.gz \
  --out-dir ../DeepPS_output/fdr/output/mouse \
  --rel-intensities 0,0.05,0.10,0.20 \
  --batch-size 1500 --exp-transform sqrt_clip
```

**Four-method peptide intersection**

```bash
# Intersect <=30 aa 1% FDR peptides from MS-GF+, MSFragger, DeepPS and Prosit.
python transfomer/fdr/postprocess_peptide_intersection.py \
  --base ../DeepPS_output/fdr \
  --msgf-dir ../DeepPS_output/engines/msgfplus \
  --fragger-dir ../DeepPS_output/engines/msfragger \
  --out-dir ../DeepPS_output/intersection --max-len 30 --fdr 0.01 \
  --deepps-branch d_f_exp_mby_rel020 --prosit-branch f_frag_rel000
```

---
## 10. Engine 1% FDR

```bash
# Create the engine output directory.
mkdir -p ../DeepPS_output/engines/mouse

# Build the engine target-decoy FASTA and modification list.
python transfomer/fdr_full/build_td_fasta.py \
  --fasta ../DeepPS_data/v1-search/mouse/for_mouse_test_inscope_peptide.fasta \
  --mods-config transfomer/fdr_full/mods_config.example.json \
  --out-fasta ../DeepPS_output/engines/mouse/mouse_td.fasta \
  --mods-out ../DeepPS_output/engines/mouse/mouse_mods.txt

# MS-GF+ chain: search, convert mzid to TSV, count 1% FDR.
bash transfomer/fdr_full/run_msgfplus.sh mouse \
  ../DeepPS_data/v1-search/mouse/for_mouse_test_inscope_peptide.fasta \
  ../DeepPS_data/v1-search/mouse/high.mouse.mzML \
  ../DeepPS_output/engines/mouse
```

Manual MS-GF+ equivalent (Windows / no Bash):

```bash
# Run the MS-GF+ search: 10 ppm precursor, 0 Da fragment tolerance.
java -Xmx12G -jar ../MSGFPlus_deepps240326/MSGFPlus.jar \
  -s ../DeepPS_data/v1-search/mouse/high.mouse.mzML \
  -d ../DeepPS_output/engines/mouse/mouse_td.fasta \
  -o ../DeepPS_output/engines/mouse/mouse.mzid \
  -t 10ppm -ti 0,0 -m 3 -inst 1 -e 9 \
  -minCharge 2 -maxCharge 4 \
  -mod ../DeepPS_output/engines/mouse/mouse_mods.txt \
  -thread 8 -allowDenseCentroidedPeaks 1

# Convert the mzid result to a per-PSM TSV.
../MSGFPlus_deepps240326/MzidToTsvConverter/MzidToTsvConverter.exe \
  -mzid:../DeepPS_output/engines/mouse/mouse.mzid \
  -tsv:../DeepPS_output/engines/mouse/mouse.tsv -unroll -showDecoy

# Count unique <=30 aa targets at 1% FDR (use --scan SpecId for multi-RAW input).
python transfomer/fdr_full/engine_fdr.py --engine msgfplus \
  --tsv ../DeepPS_output/engines/mouse/mouse.tsv \
  --decoy-token rev_ --fdr 0.01 --max-len 30
```

MSFragger: search in FragPipe with database
`../DeepPS_output/engines/mouse/mouse_td.fasta`, decoy generation `None` and prefix
`rev_`, nocleavage, isotope error 0, ±10 ppm, calibration off, then:

```bash
# Count unique <=30 aa MSFragger targets at 1% FDR.
python transfomer/fdr_full/engine_fdr.py --engine msfragger \
  --tsv ../DeepPS_output/engines/msfragger/mouse/psm.tsv \
  --decoy-token rev_ --fdr 0.01 --max-len 30
```

For repeated scan numbers (multi-RAW MGF), add `--scan SpecId` for MS-GF+, or
renumber the MGF before MSFragger (section 11).

```bash
# DeepPS leg of the engine comparison at 1% FDR.
bash transfomer/fdr_full/run_deepps_fdr.sh
```

---
## 11. DeepSearch dataset 1% FDR analysis

**DeepPS**

```bash
# DeepSearch TD dataset, DeepPS leg: convert -> match -> predict/score -> FDR -> unique peptides.
python transfomer/fdr/custom_td_1pct/run_custom_fdr.py \
  --name AT \
  --mgf ../DeepPS_data/custom/AT/AT.mgf \
  --fasta ../DeepPS_data/custom/AT/Ath_target_decoy_peptide.fasta \
  --work ../DeepPS_output/custom --ppm 10 --batch-size 128 \
  --rel 0.05,0.10,0.20 --fdr-targets 0.01 \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt --gpu 0
```

Default modifications are fixed C+57.021 + variable M+15.995 (the script
default). `--rel 0.05,0.10,0.20` sets the rel levels (the default). `--work` is
the output directory; `--mgf` and `--fasta` are the experimental spectra and the
target-decoy peptide FASTA.

**Prosit**

```bash
# Prefetch all candidate keys into the Prosit caches (4 shards per label).
python transfomer/fdr/custom_td_1pct/run_custom_prosit.py \
  --name AT \
  --mgf ../DeepPS_data/custom/AT/AT.mgf \
  --candidates ../DeepPS_output/custom/AT/data/candidates_AT_td.tsv.gz \
  --work ../DeepPS_output/custom --jobs 4 --prefetch-only

# Score the custom TD candidates with the prefetched Prosit caches.
python transfomer/fdr/custom_td_1pct/run_custom_prosit.py \
  --name AT \
  --mgf ../DeepPS_data/custom/AT/AT.mgf \
  --candidates ../DeepPS_output/custom/AT/data/candidates_AT_td.tsv.gz \
  --work ../DeepPS_output/custom --skip-prefetch
```

Large caches: build a SQLite index once and score with `--disk-cache`
(`--resume` after interruption):

```bash
# Convert the gzipped Prosit target cache into a disk-backed SQLite index.
python transfomer/fdr/custom_td_1pct/disk_prosit_cache.py \
  ../DeepPS_output/custom/AT/data/prosit_target_cache.tsv.gz

# Score with the disk cache at rel 0/0.05/0.10/0.20; resume appends missing rows.
python transfomer/fdr/custom_td_1pct/prosit_fdr_score_custom.py \
  --species AT --mgf ../DeepPS_data/custom/AT/AT.mgf \
  --candidates ../DeepPS_output/custom/AT/data/candidates_AT_td.tsv.gz \
  --out-dir ../DeepPS_output/custom/AT/output \
  --cache ../DeepPS_output/custom/AT/data/prosit_target_cache.tsv.gz \
  --decoy-cache ../DeepPS_output/custom/AT/data/prosit_decoy_cache.tsv.gz \
  --rel-intensities 0,0.05,0.10,0.20 \
  --batch-size 1500 --exp-transform sqrt_clip \
  --disk-cache --resume
```

**Engines**

```bash
# Convert the TD FASTA to the plain engine FASTA (target + decoy, DECOY_ prefix).
python transfomer/fdr/custom_td_1pct/convert_td_fasta.py \
  --fasta ../DeepPS_data/custom/AT/Ath_target_decoy_peptide.fasta \
  --engine-only --engine-out ../DeepPS_output/custom/AT/data/AT_engine.fasta

# Run the AT MS-GF+ search (default = C57 + M).
bash transfomer/fdr/custom_td_1pct/run_msgfplus_custom.sh AT \
  ../DeepPS_data/custom/AT/AT.mgf \
  ../DeepPS_output/custom/AT/data/AT_engine.fasta \
  ../DeepPS_output/custom/AT/msgfplus

# Rewrite TITLEs to unique species.scan.charge keys for MSFragger.
python transfomer/fdr/custom_td_1pct/make_unique_scans.py \
  --in ../DeepPS_data/custom/AT/AT.mgf \
  --out ../DeepPS_output/custom/AT/data/AT_msfragger.mgf

# Count unique <=30 aa MSFragger targets at 1% FDR.
python transfomer/fdr_full/engine_fdr.py --engine msfragger \
  --tsv ../DeepPS_output/custom/AT/msfragger/psm.tsv \
  --decoy-token DECOY_ --scan scannum --fdr 0.01 --max-len 30
```

MS-GF+ default modifications are C57 + M (the script default).

**Unique peptides**

```bash
# Unique <=30 aa MS-GF+ peptides at 1% FDR.
python transfomer/fdr/custom_td_1pct/unique_peptides.py \
  --engine msgfplus \
  --tsv ../DeepPS_output/custom/AT/msgfplus/AT.tsv \
  --decoy-token DECOY_ --scan SpecId \
  --out-dir ../DeepPS_output/custom/AT/unique --name msgfplus

# Unique <=30 aa DeepPS peptides at 1% FDR from the saved scoring table.
python transfomer/fdr/custom_td_1pct/unique_peptides_fdr_csv.py \
  --results ../DeepPS_output/custom/AT/output/results_step3_AT.csv \
  --branch d_f_exp_mby_rel020 \
  --name deepps --row-max-len 30 \
  --out-dir ../DeepPS_output/custom/AT/unique
```

The DeepSearch DeepPS results also contain all 10 default branches × 3 rel levels;
the exported unique-peptide sets are `d_f_exp_*` and `d_f_exp_mby_*`, and the
four-method intersection uses `d_f_exp_mby_rel020`. Prosit unique-peptide sets
are exported for `f_frag_rel000/005/010/020`; branches with no spectra passing
1% FDR are written as header-only files.

---
## 12. Case study

Predicts a single peptide and exports its grid as an MGF. The intensity filter is
applied in model output space.

```bash
# Predict one peptide and save the full 39x46x6 grid with provenance.
python transfomer/casestudy/case_study.py predict \
  --peptide QLEEAEEEAQR --charge 2 \
  --checkpoint ../runs/DeepPS_massivekb46/best_model.pt \
  --out-dir ../DeepPS_output/casestudy/QLEEAEEEAQR_z2 \
  --min-intensity 0.20

# Re-export an existing grid: relative-100 intensities, 0.20 filter, 0.05 Da merge.
python transfomer/casestudy/case_study.py export \
  --grid ../DeepPS_output/casestudy/QLEEAEEEAQR_z2/grid.npz \
  --out-dir ../DeepPS_output/casestudy/QLEEAEEEAQR_z2_export \
  --min-intensity 0.20 --mz-merge-tol 0.05 --intensity-space relative100
```

Outputs: `grid.npz`, `prediction_grid.npy`, `grid.csv`, `prediction.mgf`,
`peaks.tsv`, `metadata.json`, all in the given output directory.

---
## 13. MassiveKB-30000 embedding cosine results

**Bash one-shot**

```bash
# Full MassiveKB PiPrime cosine pipeline: sample -> prepare -> embeddings/cosine.
OUT_DIR=../DeepPS_output/massivekb \
TRAIN_MGF_XZ=../DeepPS_data/massivekb/massivekb_82c0124b_train.mgf.xz \
PIPRIME_MODEL=../piprime_embedding/piprime_embedding/model.ckpt \
N_GROUPS=30000 MIN_SPECTRA=2 PREFER_SPECTRA=3 WORKERS=8 \
TOL_FILTER=0.05 TOL_AVERAGE=0.05 INTENSITY_TRANSFORM=none,sqrt_clip GPU=0 \
  bash transfomer/massivekb_pi_cosine/run_massivekb_pi_cosine.sh
```

**Python staged run**

```bash
# Stage 1: sample 30000 peptide groups (at least 2 spectra each).
python transfomer/massivekb_pi_cosine/run_pipeline.py \
  --out-dir ../DeepPS_output/massivekb \
  --train-mgf-xz ../DeepPS_data/massivekb/massivekb_82c0124b_train.mgf.xz \
  --model ../piprime_embedding/piprime_embedding/model.ckpt \
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \
  --domains none,sqrt_clip --tol-filter 0.05 --tol-average 0.05 \
  --workers 8 --gpu 0 --stop-after sample

# Stage 2: filter, predict, group and average spectra for the sampled groups.
python transfomer/massivekb_pi_cosine/run_pipeline.py \
  --out-dir ../DeepPS_output/massivekb \
  --train-mgf-xz ../DeepPS_data/massivekb/massivekb_82c0124b_train.mgf.xz \
  --model ../piprime_embedding/piprime_embedding/model.ckpt \
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \
  --domains none,sqrt_clip --tol-filter 0.05 --tol-average 0.05 \
  --workers 8 --gpu 0 --stop-after prepare

# Stage 3: embeddings, pairwise cosine and summaries for the full pipeline.
python transfomer/massivekb_pi_cosine/run_pipeline.py \
  --out-dir ../DeepPS_output/massivekb \
  --train-mgf-xz ../DeepPS_data/massivekb/massivekb_82c0124b_train.mgf.xz \
  --model ../piprime_embedding/piprime_embedding/model.ckpt \
  --n-groups 30000 --min-spectra 2 --prefer-spectra 3 \
  --domains none,sqrt_clip --tol-filter 0.05 --tol-average 0.05 \
  --workers 8 --gpu 0 --stop-after full
```

The `prepare` stage performs filtering, Prosit prediction, grouping and average
spectra; the `full` stage performs embeddings, pairwise cosine and summaries.

---
## 14. Dataset sources

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


