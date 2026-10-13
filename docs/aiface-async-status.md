# AI-Face Async Task Status

Last verified: 2026-10-13 (Asia/Ho_Chi_Minh)

Attached AI-Face dataset: `nguyentrann0703/aiface-pilot-32k-single-bin`, current
Kaggle dataset version `1`.

This ledger tracks Kaggle runs and artifact downloads that are in progress. A task
is not marked complete until its `run_metadata.json` and expected output directory
have been checked. The fixed generator split remains the seed-42 manifest for all
training seeds.

## Active tasks

| Task | Owner | Status | Source / expected artifact | Next action | Completion check |
|---|---|---|---|---|---|
| Notebook 10, training seed 42 | Owner | **Verified full run** | Kaggle Global/Local baseline; seed-42 output downloaded from shared notebook | Use as seed-42 baseline input for Notebook 12 | `seed=42`, `smoke=false`, 8,000 validation/test predictions and checkpoints |
| Notebook 11, training seed 42 | Teammate / shared | **Verified full run** | Kaggle Spectral Forensic; `downloads/aiface-seed42-fusion/` input evidence | Use as seed-42 forensic input for Notebook 12 | `seed=42`, `smoke=false`, 8,000 validation/test predictions and checkpoint |
| Notebook 12, fusion seed 42 | Owner / shared | **Verified complete** | Kaggle `ai-face-phase-e-fusion-seed-42`; `downloads/aiface-seed42-fusion/` | Keep fusion ZIP and report metrics | 4/4 input tests pass; both fusion outputs and rescue/harm complete |
| Notebook 10, training seed 43 | Owner | **Verified full run** | Kaggle Global/Local baseline; `downloads/aiface-seed43-baseline/aiface_phase_e_seed43_baseline/` | Use as seed-43 baseline input for Notebook 12 | `seed=43`, `smoke=false`, `test_rows=8000`, manifest aligned with spectral |
| Notebook 10, training seed 44 | Owner | **Verified full run** | Kaggle version label `6`; `downloads/aiface-seed44-baseline/aiface_phase_e_seed44_baseline/` | Use as seed-44 baseline input for Notebook 12 | `seed=44`, `smoke=false`, test rows `8000`, manifest aligned with spectral |
| Notebook 11, training seed 43 | Teammate / shared | **Verified full run** | Kaggle notebook `ai-face-phase-e-spectral-forensic`; `downloads/aiface-seed43-spectral-full/` | Use as seed-43 forensic input for Notebook 12 | `smoke=false`, `seed=43`, 20 epochs, 8,000 validation/test predictions |
| Notebook 11, training seed 44 | Teammate / shared | **Verified full run** | `downloads/aiface-seed44-spectral-full/outputs/aiface_phase_e_seed44_spectral/forensic_only/` | Pair only with the seed-44 baseline for Notebook 12 | `seed=44`, `smoke=false`, 20 epochs, 8,000 validation/test predictions |
| Notebook 12, fusion seed 43 | Owner / shared | **Verified complete** | Kaggle `ai-face-phase-e-fusion-seed-43`; `downloads/aiface-seed43-fusion/` | Keep fusion ZIP and report metrics | Input tests pass; both fusion outputs complete with seed-43 alignment |
| Notebook 12, fusion seed 44 | Owner / shared | **Verified complete** | Kaggle `ai-face-phase-e-fusion-seed-44`; `downloads/aiface-seed44-fusion/` | Keep fusion ZIP and report metrics | GPU-enabled T4 run; input tests pass; both fusion outputs complete with seed-44 alignment |
| Manual seed-43 baseline download | Owner | **Verified** | `downloads/aiface-seed43-baseline/aiface_phase_e_seed43_baseline/` | Keep as local provenance copy | `seed=43`, `smoke=false`, manifest aligned with forensic |

## Phase F status

| Task | Status | Required output |
|---|---|---|
| Robustness inference | **Verified complete** | `outputs/aiface_phase_f_seed42_complete/`, `outputs/aiface_phase_f_seed43/`, `outputs/aiface_phase_f_seed44/`; 5 conditions, 30 predictions per seed, per-generator metrics and clean deltas |
| Bootstrap uncertainty | **Verified complete** | `outputs/aiface_phase_f_uncertainty/`; 2,000 replicates, seed 2026, 95% CI and paired differences; 0 NaN values |
| Validation-only model selection | **Verified complete** | `outputs/aiface_phase_f_model_selection_v2/model_selection.json`; selected `global_local_forensic` without test metrics |
| Final report/status update | **Verified complete** | Phase F results and limitations recorded in `docs/aiface-phase-e-report.md` and `docs/aiface-phase-f-plan.md` |

### Phase F verification summary

- All seeds use the same `generator_disjoint_fixed_v1` manifests and 8,000 held-out test IDs.
- Robustness conditions are `clean`, `jpeg_q70`, `resize_112_224`, `blur_r1` and `center_crop_80`.
- Seed 44 required validation-only reconstruction from clean validation images because the compact Kaggle artifact omitted validation CSVs; test labels were not used for threshold selection.
- Bootstrap outputs contain 180 model/condition CI rows and 90 paired-delta rows, all with `replicates=2000`, `bootstrap_seed=2026` and no NaN values.

## Locked protocol

- Train only the three independent branches: Global-only, Local-only and Forensic-only.
- Derive the three remaining configurations by equal late probability fusion.
- Do not combine artifacts from different training seeds.
- Select branch and fusion thresholds on validation only; evaluate test afterward.
- Reject smoke outputs as scientific results.
- Use only `generator_disjoint_fixed_v1`; no random fallback or split regeneration.

## Artifact checklist

Before running Notebook 12 for a seed, confirm both inputs contain:

- `run_metadata.json` with the same `seed`, `split_mode`, `generator_splits` and manifest checksums;
- `smoke=false`;
- validation predictions, or baseline `feature_cache.pt` plus the required classifier checkpoints;
- test predictions with matching `image_id`, `label`, `generator_id` and `split`.

After fusion, record the Kaggle URL/version and keep:

```text
comparison.csv
comparison_by_generator.csv
local_forensic/rescue_harm.csv
global_local_forensic/rescue_harm.csv
facetrace_fusion_artifacts.zip
```

Update this ledger after every download, run, or metadata verification. Do not
replace **Pending**, **Running** or **Waiting** with **Verified** based only on a
Kaggle page title.
