# MRI reconstruction baselines: BOBs T1/T2 and IXI T1

A small local-first starting point for the MRI project: a two-channel 2D
convolutional autoencoder with one or two channels, fixed subject splits, Hydra configs, deterministic
training, a lockfile, local run records and fail-closed regression checks.
Run all commands from this directory. It can become its own Git repository;
no repository has been initialized automatically.

## Scope and interpretation

- **Separate experiments:** BOBs T1w + T2w (paired on the same physical grid),
  or the local prepared IXI T1-only dataset. IXI is now the default.
  Keep dHCP and BONBID out of these experiments.
- BOBs is an infant segmentation repository, **not a verified healthy cohort**.
  The manifest and run records say `health_unverified`. Review participant
  eligibility before calling this healthy-only anomaly training. If exclusions
  are needed, curate the cohort and freeze a new manifest before a formal run.
- This is a reconstruction baseline, **not a validated anomaly detector**.
  Reconstruction error is not a lesion probability. There are no lesion labels
  here, so no AUROC/AUPRC, small-lesion AUROC or ECE is fabricated.
- The local dataset description declares **CC-BY-4.0**. Verify the source terms
  before publishing derivatives, and cite
  [Feczko et al., Baby Open Brains](https://doi.org/10.1101/2024.10.02.616147)
  and the [dataset DOI](https://doi.org/10.17605/OSF.IO/WDR78).
  Do not publish raw scans, sidecars or clinical metadata in Git.
- The IXI adapter targets the local derivative layout
  `sub-*/ses-*/anat/msub-*_ses-*_T1w_brain_affine_mni.nii.gz`, **not raw IXI
  archives**. Review the derivative's provenance, processing, source subject
  mapping, license and eligibility before interpreting or publishing results.
  It also remains `health_unverified`; no eligibility is inferred from filenames.

## Switching datasets correctly

Changing `data_root` only relocates files referenced by the current manifest.
It does not discover a new dataset. Dataset, modalities, frozen splits and
checkpoint must agree. The profiles select the root and manifest together:
`dataset=ixi` selects one T1 channel and `eval/ixi_manifest.json`;
`dataset=bobs` restores two T1/T2 channels and `eval/frozen_manifest.json`.
`MRI_DATA_ROOT`, if set, overrides either profile's root; unset it when switching
datasets unless it deliberately points to the selected dataset.

The first IXI freeze contains **409 subjects / 409 T1 volumes**:
327 train, 41 validation and 41 test subjects, using split seed 42.

For the new IXI baseline:

```bash
# Keep the installed CUDA environment; no dependency changes are required.
uv run --no-sync python -m cvproj.data.prepare --dataset IXI \
  --data-root /data/hf_brain_struc/IXI --output eval/ixi_manifest.json

# First smoke-test actual IXI scans (not a baseline):
uv run --no-sync python -m cvproj.train dataset=ixi device=cuda \
  max_epochs=1 subset_subjects=2 image_size=64 slice_step=16

# Then start a NEW full-data training run:
uv run --no-sync python -m cvproj.train dataset=ixi device=cuda seed=0

# Evaluate its checkpoint; use the smoke preprocessing overrides if evaluating a smoke checkpoint.
uv run --no-sync python -m cvproj.evaluate dataset=ixi device=cuda \
  checkpoint=outputs/<date>/<ixi-run>/best.pt
```

Preparation refuses to overwrite frozen splits; use `--check` for an existing
manifest. BOBs files, its original manifest and checkpoints are left unchanged.
An old BOBs checkpoint remains evaluable with `dataset=bobs` and an explicit
checkpoint path. Evaluation no longer defaults to a hard-coded BOBs checkpoint.
`cvproj.evaluate` does **not train** a new model, and changing `seed` there
does not retrain the checkpoint.

Keep reviewed IXI baselines separately in `eval/ixi/`:
copy the IXI run's `metrics.json` and `protocol.json` to
`eval/ixi/baseline_metrics.json` and `eval/ixi/baseline_protocol.json`, respectively.
Use those paths in the gate command; do not overwrite a BOBs baseline or compare
their MSE as equivalent performance. The protocol gate rejects different data
or modalities. For IXI reproduction:

```bash
MRI_BACKEND=cuda MRI_BASELINE_DIR=eval/ixi bash scripts/reproduce.sh dataset=ixi seed=0
```

The original BOBs instructions follow; all model invocations explicitly select
`dataset=bobs` so they do not silently switch to the new default.

## 1. Install and freeze the data once

```bash
uv sync --frozen --extra cpu
uv run --frozen --extra cpu python -m cvproj.data.prepare \
  --data-root /data/bobs --output eval/frozen_manifest.json
```

If the manifest already exists, use `--check` instead. Preparation checks
pairing, 3D shape, T1/T2 physical-grid agreement, finite intensities and positive
foreground, then records SHA256 fingerprints and canonical shapes.
All visits from a subject share one deterministic 80/10/10 train/val/test split
(split seed 42, independent of training seed). It refuses to overwrite an
existing manifest. Commit the manifest and lockfile once this is a Git repo.

The first local freeze contains **51 subjects / 71 paired visits**:
41 subjects / 58 visits in train, 5 / 5 in validation, and 5 / 8 in test.

The external `/data` symlink stays external: no image copying, no source-data
mutation, and no DVC upload. Relocation is supported via
`export MRI_DATA_ROOT=/new/location/bobs`.
Every real-data training/evaluation invocation verifies all manifest file hashes.
Review montages, foreground coverage, age distribution and cohort eligibility
before full training; basic numeric QC cannot replace visual QC.

## 2. Validate the wiring before training

```bash
uv run --frozen --extra cpu ruff check .
uv run --frozen --extra cpu ruff format --check .
uv run --frozen --extra cpu mypy src
uv run --frozen --extra cpu pytest

# Synthetic smoke: no dataset access, no claims about model performance.
uv run --frozen --extra cpu python -m cvproj.train \
  dataset=bobs synthetic=true max_epochs=1 image_size=32 width=4 threads=1

# Real paired-data smoke; this is NOT a baseline.
uv run --frozen --extra cpu python -m cvproj.train \
  dataset=bobs max_epochs=1 subset_subjects=2 image_size=64 slice_step=16
```

The included GitHub workflow is for this directory as the future repository
root. It uses only synthetic data in CI; a hosted runner never needs the
external dataset. C++/Docker/GPU-eval jobs are deliberately absent until those
components exist, rather than shipping failing template jobs.

## 3. Train the first full-data baseline

```bash
# CPU, if desired:
uv run --frozen --extra cpu python -m cvproj.train dataset=bobs seed=0

# CUDA 12.8 wheel supports the local RTX 5080 (Blackwell).
# Switch extras rather than installing a second environment.
uv sync --frozen --extra cuda
uv run --frozen --extra cuda python -m cvproj.train dataset=bobs device=cuda seed=0
```

The default is 10 epochs, 128x128 axial slices, batch size 16, Adam at 1e-3.
The two-channel input is T1 then T2. Each volume is reoriented to canonical RAS;
positive voxels are independently clipped/scaled at the 1st/99th percentiles
per modality. The central 60% of axial slices is sampled every four slices.
Both channels get the same bilinear resize. No atlas registration, isotropic
resampling, skull stripping or age correction is claimed. Resizing can distort
physical geometry; this is a quick 2D baseline, not the final preprocessing
protocol. Zero padding is excluded from the loss, but positive foreground is
**not** an anatomical brain mask.

Volume order is seeded and shuffled per epoch; slices remain contiguous to
avoid repeatedly decompressing NIfTI volumes. A two-volume cache and zero data
workers bound memory. Training uses slice-level masked MSE; evaluation averages
within each subject, then across subjects, so subjects with more visits or
slices do not dominate model selection. Validation chooses `best.pt`.
The test split is never accessed by training. Five subjects per held-out split
is a small pilot, not enough for precise subgroup or clinical-performance claims.

Each unique run directory contains:

- `.hydra/config.yaml`: resolved settings are also stored in the checkpoint.
- `run.json`: seed, device, PyTorch/CUDA versions, manifest/source/lock hashes,
  Git commit and dirty state (explicitly null before Git is initialized).
- `history.jsonl`, `metrics.json`, `subject_scores.json`: local metrics only.
- `protocol.json`: split, frozen-data hash and preprocessing/aggregation contract.
- `best.pt`: best validation checkpoint.

Do not reuse a run directory or use a mutable `outputs/latest` alias for gates.
There is no automatic external tracking, registration or baseline promotion.

## 4. Review and freeze a baseline, then gate candidates

Do not promote a synthetic or subset smoke run. After reviewing a **full-data**
validation run, explicitly freeze its result:

```bash
BASELINE=outputs/<date>/<reviewed-run>
cp "$BASELINE/metrics.json" eval/baseline_metrics.json
cp "$BASELINE/protocol.json" eval/baseline_protocol.json
```

Record the checkpoint/run ID, interpretation and limitations alongside the
baseline in your experiment report. No baseline scores are prefilled.
The current validation gate is an absolute reconstruction-MSE regression
tolerance of **0.001**, a provisional engineering guardrail, not a clinical
threshold. Adjust it from observed seed variance and record that decision.

```bash
CANDIDATE=outputs/<date>/<candidate-run>
uv run --frozen --no-sync python scripts/eval_gate.py \
  --candidate "$CANDIDATE/metrics.json" --baseline eval/baseline_metrics.json \
  --rules eval/reconstruction_rules.json \
  --candidate-protocol "$CANDIDATE/protocol.json" \
  --baseline-protocol eval/baseline_protocol.json
```

Metrics are gated individually. Missing metrics fail; invalid directions,
negative tolerances, booleans, nonnumeric/NaN/Inf values and incompatible
protocols are errors. Exit codes: 0 pass, 1 regression/missing metric, 2 invalid
input. The original lesion/calibration/latency rules are preserved in
`eval/gate_rules.json` for later; applying them now **must fail** because those
metrics cannot yet be measured. Later add a compatible lesion-labeled cohort,
explicit lesion-size/scanner definitions, calibration on validation data,
and a same-device synchronized benchmark before enabling that gate.
The gate uses only the standard library; `--no-sync` avoids unnecessarily
switching a CUDA training environment back to CPU wheels.

## 5. Reproduce and measure seed variance

```bash
# Uses local frozen data, not a nonexistent DVC remote.
bash scripts/reproduce.sh dataset=bobs
MRI_BACKEND=cuda bash scripts/reproduce.sh dataset=bobs seed=0

# Hydra multirun (sequential, no orchestrator):
uv run --frozen --extra cuda python -m cvproj.train -m dataset=bobs device=cuda seed=0,1,2
uv run --frozen --extra cuda python scripts/summarize_seeds.py \
  multirun/<date>/<sweep>/0 multirun/<date>/<sweep>/1 multirun/<date>/<sweep>/2
```

Reproduction verifies fingerprints, trains, then gates against a reviewed
baseline. Before baseline approval it exits **3** after training with an
explicit message, never reporting a gate pass. The summary uses sample standard
deviation and rejects duplicate seeds, different protocols, different training
configs beyond seed, or different code/runtime provenance.

Once the method is fixed, explicitly evaluate the held-out test split with the
same preprocessing settings as training (do not use it repeatedly for tuning):

```bash
uv run --frozen --extra cuda python -m cvproj.evaluate device=cuda \
  dataset=bobs checkpoint=outputs/<date>/<run>/best.pt split=test
```

Deterministic flags provide repeatability on the same software/hardware, not
bitwise equality across devices. Unsupported deterministic operations fail
explicitly. There is no checkpoint-resume support yet.

## Add only when needed

1. Confirm eligibility/age strata and visualize preprocessing before claiming
   healthy-only results. Add dHCP as a separately versioned experiment after its
   download and modality/cohort review finish.
2. Use the baseline to establish runtime/memory/seed variance, then add a
   stronger reconstruction or feature model behind the same eval protocol.
3. Add DVC when distributing data or managing expensive cached preprocessing;
   configure a real remote before adding `dvc pull/repro`.
4. Add opt-in MLflow/W&B when local run directories become difficult to manage;
   keep raw medical images and metadata out of external tracking by default.
5. Add ONNX/C++ and a real Python-vs-C++ parity fixture when there is a C++
   implementation. Do not add a permanently skipped parity test now.
6. Add a pinned container when moving training off this machine. Defer model
   registry aliases, GPU CI and orchestration until there is a release/deployment.
