# BOBs paired T1/T2 reconstruction baseline

A small local-first starting point for the MRI project: a two-channel 2D
convolutional autoencoder, fixed subject splits, Hydra configs, deterministic
training, a lockfile, local run records and fail-closed regression checks.
Run all commands from this directory. It can become its own Git repository;
no repository has been initialized automatically.

## Scope and interpretation

- **Only BOBs T1w + T2w**, paired within each visit on the same physical grid.
  Keep dHCP (still downloading) and BONBID out of this experiment.
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
  synthetic=true max_epochs=1 image_size=32 width=4 threads=1

# Real paired-data smoke; this is NOT a baseline.
uv run --frozen --extra cpu python -m cvproj.train \
  max_epochs=1 subset_subjects=2 image_size=64 slice_step=16
```

The included GitHub workflow is for this directory as the future repository
root. It uses only synthetic data in CI; a hosted runner never needs the
external dataset. C++/Docker/GPU-eval jobs are deliberately absent until those
components exist, rather than shipping failing template jobs.

## 3. Train the first full-data baseline

```bash
# CPU, if desired:
uv run --frozen --extra cpu python -m cvproj.train seed=0

# CUDA 12.8 wheel supports the local RTX 5080 (Blackwell).
# Switch extras rather than installing a second environment.
uv sync --frozen --extra cuda
uv run --frozen --extra cuda python -m cvproj.train device=cuda seed=0
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
bash scripts/reproduce.sh
MRI_BACKEND=cuda bash scripts/reproduce.sh seed=0

# Hydra multirun (sequential, no orchestrator):
uv run --frozen --extra cuda python -m cvproj.train -m device=cuda seed=0,1,2
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
  checkpoint=outputs/<date>/<run>/best.pt split=test
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
