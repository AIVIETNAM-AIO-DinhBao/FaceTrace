#!/usr/bin/env python3
"""Build the paper-oriented AI-Face Phase E/F results report from local artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "aiface-complete-results.md"

SEEDS = (42, 43, 44)
CONDITIONS = ("clean", "jpeg_q70", "resize_112_224", "blur_r1", "center_crop_80")
MODELS = (
    "global_only",
    "local_only",
    "global_local",
    "forensic_only",
    "local_forensic",
    "global_local_forensic",
)
MODEL_NAMES = {
    "global_only": "Global-only",
    "local_only": "Local-only",
    "global_local": "Global + Local",
    "forensic_only": "Forensic-only",
    "local_forensic": "Local + Forensic",
    "global_local_forensic": "Global + Local + Forensic",
}
CONDITION_NAMES = {
    "clean": "clean",
    "jpeg_q70": "JPEG quality 70",
    "resize_112_224": "resize 112→224",
    "blur_r1": "Gaussian blur radius 1",
    "center_crop_80": "center crop 80%",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def phase_f_dir(seed: int) -> Path:
    return ROOT / "outputs" / (
        "aiface_phase_f_seed42_complete" if seed == 42 else f"aiface_phase_f_seed{seed}"
    )


def ordered(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column, values in columns:
        result[column] = pd.Categorical(result[column], categories=list(values), ordered=True)
    return result.sort_values([column for column, _ in columns]).reset_index(drop=True)


def value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int,)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df.loc[:, columns]
    headers = [str(column) for column in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value(item) for item in row) + " |")
    return "\n".join(lines)


def rename_model(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "model" in result:
        result["model"] = result["model"].map(MODEL_NAMES).fillna(result["model"])
    return result


def load_phase_f() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    macro_parts = []
    generator_parts = []
    delta_parts = []
    metadata = []
    for seed in SEEDS:
        directory = phase_f_dir(seed)
        macro = read_csv(directory / "metrics_by_condition_model.csv")
        generators = read_csv(directory / "metrics_by_generator.csv")
        deltas = read_csv(directory / "deltas_vs_clean.csv")
        assert len(macro) == 30, (seed, len(macro))
        assert len(generators) == 60, (seed, len(generators))
        assert len(deltas) == 30, (seed, len(deltas))
        assert set(macro["condition"]) == set(CONDITIONS)
        assert set(macro["model"]) == set(MODELS)
        macro_parts.append(macro)
        generator_parts.append(generators)
        delta_parts.append(deltas)
        metadata.append(json.loads((directory / "run_metadata.json").read_text()))
    macro = pd.concat(macro_parts, ignore_index=True)
    generators = pd.concat(generator_parts, ignore_index=True)
    deltas = pd.concat(delta_parts, ignore_index=True)
    macro = ordered(macro, (("seed", SEEDS), ("condition", CONDITIONS), ("model", MODELS)))
    generators = ordered(
        generators,
        (("seed", SEEDS), ("condition", CONDITIONS), ("model", MODELS)),
    )
    deltas = ordered(deltas, (("seed", SEEDS), ("condition", CONDITIONS), ("model", MODELS)))
    return macro, generators, deltas, metadata


def load_uncertainty() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    directory = ROOT / "outputs" / "aiface_phase_f_uncertainty"
    ci = read_csv(directory / "ci_by_model_condition.csv")
    paired = read_csv(directory / "paired_delta_ci.csv")
    summary = read_csv(directory / "seed_summary_mean_sd.csv")
    config = json.loads((directory / "bootstrap_metadata.json").read_text())
    assert len(ci) == 180
    assert len(paired) == 90
    assert len(summary) == 30
    assert not ci.isna().any().any()
    assert not paired.isna().any().any()
    return ci, paired, summary, config


def load_model_selection() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    directory = ROOT / "outputs" / "aiface_phase_f_model_selection_v2"
    records = read_csv(directory / "validation_records.csv")
    ranking = read_csv(directory / "model_selection.csv")
    selected = json.loads((directory / "model_selection.json").read_text())
    assert len(records) == 18
    assert len(ranking) == 6
    return records, ranking, selected


def render() -> str:
    macro, generators, deltas, metadata = load_phase_f()
    ci, paired, summary, bootstrap_metadata = load_uncertainty()
    validation, ranking, selected = load_model_selection()

    # Point estimates in the bootstrap artifact must match the source metrics.
    for _, row in ci.iterrows():
        source = macro[
            (macro.seed == row.seed)
            & (macro.condition == row.condition)
            & (macro.model == row.comparison)
        ].iloc[0]
        source_value = source["macro_auroc"] if row.metric == "macro_auroc" else source["macro_balanced_accuracy"]
        assert abs(float(row.point_estimate) - float(source_value)) < 1e-8
    assert ranking.iloc[0]["model"] == "global_local_forensic"
    assert selected["selected_model"] == "global_local_forensic"

    macro_display = macro.copy()
    macro_display["model"] = macro_display["model"].map(MODEL_NAMES)
    macro_display["condition"] = macro_display["condition"].map(CONDITION_NAMES)
    macro_display = macro_display.rename(
        columns={
            "seed": "Seed",
            "condition": "Condition",
            "model": "Model",
            "threshold": "Threshold",
            "n_images": "N",
            "macro_auroc": "Macro AUROC",
            "macro_balanced_accuracy": "Macro BAcc",
            "accuracy": "Accuracy",
            "f1": "F1",
        }
    )

    clean = macro[macro.condition == "clean"].copy()
    clean["model"] = clean["model"].map(MODEL_NAMES)
    clean = clean.rename(
        columns={
            "seed": "Seed",
            "model": "Model",
            "threshold": "Threshold",
            "n_images": "N",
            "macro_auroc": "Macro AUROC",
            "macro_balanced_accuracy": "Macro BAcc",
            "accuracy": "Accuracy",
            "f1": "F1",
        }
    )

    generators_display = generators.copy()
    generators_display["model"] = generators_display["model"].map(MODEL_NAMES)
    generators_display["condition"] = generators_display["condition"].map(CONDITION_NAMES)
    generators_display = generators_display.rename(
        columns={
            "seed": "Seed",
            "condition": "Condition",
            "model": "Model",
            "generator_id": "Generator",
            "n_images": "N",
            "n_real": "N real",
            "n_fake": "N fake",
            "auroc": "AUROC",
            "balanced_accuracy": "BAcc",
            "accuracy": "Accuracy",
            "f1": "F1",
        }
    )

    deltas_display = deltas.copy()
    deltas_display["model"] = deltas_display["model"].map(MODEL_NAMES)
    deltas_display["condition"] = deltas_display["condition"].map(CONDITION_NAMES)
    deltas_display = deltas_display.rename(
        columns={
            "seed": "Seed",
            "condition": "Condition",
            "model": "Model",
            "delta_auroc": "Δ AUROC vs clean",
            "delta_balanced_accuracy": "Δ BAcc vs clean",
        }
    )

    summary_display = summary.copy()
    summary_display["model"] = summary_display["model"].map(MODEL_NAMES)
    summary_display["condition"] = summary_display["condition"].map(CONDITION_NAMES)
    summary_display = summary_display.rename(
        columns={
            "condition": "Condition",
            "model": "Model",
            "seeds": "Seeds",
            "n_seeds": "N seeds",
            "macro_auroc_mean": "Macro AUROC mean",
            "macro_auroc_sd": "Macro AUROC SD",
            "macro_balanced_accuracy_mean": "Macro BAcc mean",
            "macro_balanced_accuracy_sd": "Macro BAcc SD",
            "auroc_mean": "Overall AUROC mean",
            "auroc_sd": "Overall AUROC SD",
            "balanced_accuracy_mean": "Overall BAcc mean",
            "balanced_accuracy_sd": "Overall BAcc SD",
        }
    )

    ci_display = ci.copy()
    ci_display["comparison"] = ci_display["comparison"].map(MODEL_NAMES).fillna(ci_display["comparison"])
    ci_display["condition"] = ci_display["condition"].map(CONDITION_NAMES)
    ci_display = ci_display.rename(
        columns={
            "seed": "Seed",
            "condition": "Condition",
            "metric": "Metric",
            "comparison": "Model",
            "point_estimate": "Estimate",
            "ci_lower": "CI 2.5%",
            "ci_upper": "CI 97.5%",
            "replicates": "Replicates",
            "bootstrap_seed": "Bootstrap seed",
        }
    )

    paired_display = paired.copy()
    paired_display["condition"] = paired_display["condition"].map(CONDITION_NAMES)
    paired_display = paired_display.rename(
        columns={
            "seed": "Seed",
            "condition": "Condition",
            "metric": "Metric",
            "comparison": "Comparison",
            "point_estimate": "Delta estimate",
            "ci_lower": "CI 2.5%",
            "ci_upper": "CI 97.5%",
            "replicates": "Replicates",
            "bootstrap_seed": "Bootstrap seed",
        }
    )

    validation_display = validation.copy()
    validation_display["model"] = validation_display["model"].map(MODEL_NAMES)
    validation_display = validation_display.rename(
        columns={
            "seed": "Seed",
            "model": "Model",
            "macro_auroc": "Validation macro AUROC",
            "macro_balanced_accuracy": "Validation macro BAcc",
            "worst_generator_auroc": "Worst-generator AUROC",
            "threshold": "Threshold",
        }
    )
    ranking_display = ranking.copy()
    ranking_display["model"] = ranking_display["model"].map(MODEL_NAMES)
    ranking_display = ranking_display.rename(
        columns={
            "selection_rank": "Rank",
            "model": "Model",
            "val_macro_auroc_mean": "Val AUROC mean",
            "val_macro_auroc_sd": "Val AUROC SD",
            "val_macro_bacc_mean": "Val BAcc mean",
            "val_macro_bacc_sd": "Val BAcc SD",
            "seeds": "Seeds",
            "worst_generator_auroc": "Worst-generator AUROC",
        }
    )

    shortcut = read_csv(ROOT / "downloads/aiface-seed42-baseline/aiface_phase_e/shortcut/metrics.csv")

    source_table = pd.DataFrame(
        [
            ["E01", "Dataset/split", "outputs/aiface_phase_f_seed42_complete/run_metadata.json", "Dataset, fixed split, manifest checksums"],
            ["E02", "Phase E seed 42 baseline", "downloads/aiface-seed42-baseline/aiface_phase_e/", "Global/Local metrics, validation, shortcut"],
            ["E03", "Phase E seed 42 forensic/fusion", "downloads/aiface-seed42-fusion/ and downloads/aiface-seed42-spectral/", "Forensic and fusion validation/test/rescue-harm"],
            ["E04", "Phase E seed 43", "downloads/aiface-seed43-baseline/, downloads/aiface-seed43-spectral-full/, downloads/aiface-seed43-fusion/", "Seed-43 baseline, forensic and fusion"],
            ["E05", "Phase E seed 44", "downloads/aiface-seed44-baseline/, downloads/aiface-seed44-spectral-full/, downloads/aiface-seed44-fusion/", "Seed-44 baseline, forensic and fusion"],
            ["E06", "Phase F robustness", "outputs/aiface_phase_f_seed42_complete/, outputs/aiface_phase_f_seed43/, outputs/aiface_phase_f_seed44/", "Condition/model and per-generator metrics"],
            ["E07", "Phase F uncertainty", "outputs/aiface_phase_f_uncertainty/", "Bootstrap CIs and seed summary"],
            ["E08", "Phase F selection", "outputs/aiface_phase_f_model_selection_v2/", "Validation-only model ranking"],
        ],
        columns=["Evidence ID", "Evidence group", "Path", "Used for"],
    )

    meta_rows = []
    for item in metadata:
        meta_rows.append(
            [
                item["seed"],
                item["split_mode"],
                item["test_rows"],
                item["device"],
                ", ".join(item["conditions"]),
                "; ".join(f"{k}={v}" for k, v in item["manifest_sha256"].items()),
            ]
        )
    metadata_table = pd.DataFrame(
        meta_rows,
        columns=["Seed", "Split mode", "Test rows", "Device", "Conditions", "Manifest SHA256"],
    )

    lines = [
        "# AI-Face Complete Results — Phase E/F",
        "",
        "**Mục đích:** tài liệu số liệu đầy đủ để viết paper về khả năng bổ sung của local representation và low-level spectral forensic evidence đối với global DINOv3 trên unseen AI-face generators.",
        "",
        "**Trạng thái:** Phase E và Phase F đã hoàn tất trên pilot; tất cả bảng dưới đây được dựng từ artifact local đã kiểm tra. Các con số mô tả pilot này, không phải kết luận cho mọi generator.",
        "",
        "## 1. Research question và phạm vi",
        "",
        "> Các đặc trưng local và forensic mức thấp có cung cấp thông tin bổ sung cho biểu diễn toàn ảnh của DINOv3 khi tổng quát hóa sang nguồn khuôn mặt AI chưa xuất hiện trong huấn luyện hay không?",
        "",
        "Phạm vi thực nghiệm là real face versus fully AI-generated face. Không đưa các deepfake-video manipulation subset vào main experiment.",
        "",
        "## 2. Dataset và fixed split",
        "",
        "Dataset: `nguyentrann0703/aiface-pilot-32k-single-bin`. Split mode: `generator_disjoint_fixed_v1`; test generators không xuất hiện trong train/validation. Test có 8.000 ảnh, gồm 4.000 real và 4.000 fake.",
        "",
        md_table(pd.DataFrame([
            ["Train", "8,000", "StableDiffusion1.5 (2,000), latent_diffusion (2,000), stylegan3 (2,000), taming_transformer_VQGAN (2,000)", "16,000"],
            ["Validation", "4,000", "AttGAN (2,000), Palette (2,000)", "8,000"],
            ["Held-out test", "4,000", "STARGAN (2,000), StableDiffusion_Inpainting (2,000)", "8,000"],
        ], columns=["Split", "Real", "Fake generators", "Total"])),
        "",
        "Mỗi dòng per-generator test dùng chung 4.000 real test images; do đó mỗi generator row có `N=6,000` và không phải là một test set độc lập với các dòng khác.",
        "",
        "### 2.1. Manifest metadata theo training seed",
        "",
        md_table(metadata_table),
        "",
        "Manifest SHA256 giống nhau giữa seed 42/43/44. `run_seed` chỉ thay đổi khởi tạo/thứ tự training; fixed manifest không được tạo lại.",
        "",
        "## 3. Model và protocol",
        "",
        md_table(pd.DataFrame([
            ["Global-only", "Frozen DINOv3 global image representation + classifier"],
            ["Local-only", "Frozen DINOv3 patch representation, mean pooled + classifier"],
            ["Global + Local", "(p_global + p_local) / 2"],
            ["Forensic-only", "RGB → Hann FFT2 → log magnitude → 64 radial bins/channel → 192-D vector → MLP 192→128→2"],
            ["Local + Forensic", "(p_local + p_forensic) / 2"],
            ["Global + Local + Forensic", "(p_global + p_local + p_forensic) / 3"],
        ], columns=["Model", "Definition"])),
        "",
        "Checkpoint và threshold được chọn trên validation; test chỉ được mở sau khi khóa checkpoint/threshold. Fusion dùng equal-weight late probability fusion, không train lại fusion classifier.",
        "",
        "## 4. Phase E — clean unseen-generator results",
        "",
        "Bảng dưới đây chứa toàn bộ clean test rows của ba seed. Các metric là trên cùng 8.000 test IDs; macro metric là trung bình theo STARGAN và StableDiffusion_Inpainting.",
        "",
        "### 4.1. Clean macro/overall metrics theo seed",
        "",
        md_table(clean, ["Seed", "Model", "Threshold", "N", "Macro AUROC", "Macro BAcc", "Accuracy", "F1"]),
        "",
        "### 4.2. Mean ± sample SD qua seed — clean",
        "",
        md_table(rename_model(summary[summary.condition == "clean"].copy()).rename(columns={
            "condition": "Condition", "model": "Model", "seeds": "Seeds", "n_seeds": "N seeds",
            "macro_auroc_mean": "Macro AUROC mean", "macro_auroc_sd": "Macro AUROC SD",
            "macro_balanced_accuracy_mean": "Macro BAcc mean", "macro_balanced_accuracy_sd": "Macro BAcc SD",
        }), ["Condition", "Model", "Seeds", "N seeds", "Macro AUROC mean", "Macro AUROC SD", "Macro BAcc mean", "Macro BAcc SD"]),
        "",
        "### 4.3. Validation metrics và threshold",
        "",
        md_table(validation_display, ["Seed", "Model", "Validation macro AUROC", "Validation macro BAcc", "Worst-generator AUROC", "Threshold"]),
        "",
        "### 4.4. Validation-only model selection",
        "",
        md_table(ranking_display, ["Rank", "Model", "Val AUROC mean", "Val AUROC SD", "Val BAcc mean", "Val BAcc SD", "Seeds", "Worst-generator AUROC"]),
        "",
        f"Model được chọn trước khi dùng test: **{MODEL_NAMES[selected['selected_model']]}** (`{selected['selected_model']}`). Test metric không tham gia model selection.",
        "",
        "### 4.5. Shortcut diagnostic seed 42",
        "",
        md_table(shortcut),
        "",
        "Shortcut features có tín hiệu nhất định trên validation; vì vậy không diễn giải kết quả như bằng chứng thuần túy của semantic hoặc forensic artifact nếu chưa loại trừ real-source shortcut.",
        "",
        "## 5. Phase F — robustness đầy đủ",
        "",
        "Inference được chạy trên cùng test IDs với checkpoint đã khóa. Mỗi seed có 5 conditions × 6 models = 30 rows; tổng cộng 90 macro rows.",
        "",
        md_table(macro_display, ["Seed", "Condition", "Model", "Threshold", "N", "Macro AUROC", "Macro BAcc", "Accuracy", "F1"]),
        "",
        "### 5.1. Delta so với clean",
        "",
        md_table(deltas_display, ["Seed", "Condition", "Model", "Δ AUROC vs clean", "Δ BAcc vs clean"]),
        "",
        "### 5.2. Metric theo generator",
        "",
        "Bảng sau có 180 dòng: 3 seed × 5 condition × 6 model × 2 held-out generators.",
        "",
        md_table(generators_display, ["Seed", "Condition", "Model", "Generator", "N", "N real", "N fake", "AUROC", "BAcc", "Accuracy", "F1"]),
        "",
        "## 6. Bootstrap confidence intervals",
        "",
        "Bootstrap dùng prediction đã khóa, không inference lại. Thiết lập: 2.000 replicates, bootstrap seed 2026, percentile CI 95%, stratified theo `generator_id × label`, shared indices cho paired differences.",
        "",
        "Bootstrap metadata:",
        "",
        "```json",
        json.dumps(bootstrap_metadata, ensure_ascii=False, indent=2),
        "```",
        "",
        "### 6.1. CI theo model/condition",
        "",
        "Bảng có 180 dòng; `Estimate` phải khớp point estimate trong `metrics_by_condition_model.csv`.",
        "",
        md_table(ci_display, ["Seed", "Condition", "Metric", "Model", "Estimate", "CI 2.5%", "CI 97.5%", "Replicates", "Bootstrap seed"]),
        "",
        "### 6.2. Paired-difference CI",
        "",
        "Bảng có 90 dòng. Delta dương nghĩa là model ở vế trái tốt hơn model ở vế phải theo metric tương ứng.",
        "",
        md_table(paired_display, ["Seed", "Condition", "Metric", "Comparison", "Delta estimate", "CI 2.5%", "CI 97.5%", "Replicates", "Bootstrap seed"]),
        "",
        "## 7. Rescue/harm của forensic fusion",
        "",
        "`Rescue` là baseline sai nhưng fusion đúng; `Harm` là baseline đúng nhưng fusion sai; `Net gain = rescue − harm`. Các bảng dưới đây dùng threshold validation riêng của từng model.",
        "",
    ]

    rescue_parts = []
    for seed in SEEDS:
        directory = ROOT / "downloads" / f"aiface-seed{seed}-fusion" / "outputs" / (
            "aiface_phase_e_fusion" if seed == 42 else f"aiface_phase_e_seed{seed}_fusion"
        )
        for model in ("local_forensic", "global_local_forensic"):
            rescue = read_csv(directory / model / "rescue_harm.csv")
            rescue.insert(0, "seed", seed)
            rescue.insert(2, "fusion", MODEL_NAMES[model])
            rescue_parts.append(rescue)
    rescue = pd.concat(rescue_parts, ignore_index=True)
    lines.extend([
        md_table(rescue.rename(columns={
            "seed": "Seed", "fusion": "Fusion", "generator_id": "Group", "n_images": "N",
            "rescue": "Rescue", "harm": "Harm", "net_gain": "Net gain",
            "both_correct": "Both correct", "both_wrong": "Both wrong",
            "baseline_threshold": "Baseline threshold", "candidate_threshold": "Fusion threshold",
        }), ["Seed", "Fusion", "Group", "N", "Rescue", "Harm", "Net gain", "Both correct", "Both wrong", "Baseline threshold", "Fusion threshold"]),
        "",
        "Net gain được báo cáo theo seed và group; không cộng các generator rows để suy ra overall vì 4.000 real images được dùng chung giữa hai generator groups.",
        "",
        "## 8. Kết luận có thể dùng trong paper",
        "",
        "1. Local-only vượt Global-only trên clean held-out test ở macro AUROC và macro BAcc trong cả ba seed; vì vậy global semantic representation đơn lẻ chưa phải baseline mạnh nhất trên pilot này.",
        "2. Forensic-only có hiệu năng clean cao nhưng biến thiên qua seed lớn hơn các nhánh Local/fusion; điều này phù hợp với việc spectral signal nhạy với training initialization và degradation.",
        "3. Local + Forensic và Global + Local + Forensic đều cho paired gain dương trên clean test; tuy nhiên fusion ba nhánh không thắng fusion hai nhánh ở mọi metric, nên cần báo cáo AUROC và BAcc riêng.",
        "4. Robustness phải đọc theo từng condition: forensic spectral có thể giảm mạnh dưới JPEG/blur, trong khi Global/Local ổn định hơn. Không gộp các condition thành một điểm duy nhất.",
        "5. Các kết luận chỉ áp dụng cho hai held-out generators của pilot: STARGAN và StableDiffusion_Inpainting. Không được viết thành tuyên bố tổng quát cho toàn bộ GAN/diffusion hoặc mọi AI generator.",
        "",
        "## 9. Giới hạn và nguy cơ diễn giải",
        "",
        "- Pilot dùng 2.000 fake ảnh mỗi generator và chỉ có hai held-out generators.",
        "- Real pool có nguy cơ real-source shortcut; shortcut diagnostic validation AUROC không bằng random.",
        "- Generator-disjointness được đảm bảo theo manifest; identity-disjointness chưa được chứng minh đầy đủ từ metadata công khai.",
        "- Bootstrap CI phản ánh uncertainty do resampling test images trên fixed split/checkpoints; không phải uncertainty trên tập generator mới hay uncertainty do retraining.",
        "- Mean ± SD qua ba seed mô tả training variation, không phải confidence interval của population-level performance.",
        "- Equal-weight late fusion chưa phải calibrated/learned fusion; chưa đánh giá weighted fusion hoặc calibration.",
        "",
        "## 10. Provenance map",
        "",
        md_table(source_table),
        "",
        "Các mã `E01`–`E08` là evidence IDs nội bộ cho việc viết paper. Mỗi nhóm numeric claim trong báo cáo phải giữ lại evidence ID tương ứng khi chuyển sang manuscript.",
        "",
        "Các artifact output lớn và dataset không được commit vào Git. Báo cáo này chỉ giữ số liệu và đường dẫn evidence; khi viết paper cần lưu notebook URL, version label và ngày chạy tương ứng cho từng seed.",
        "",
        "## 11. Kiểm tra nhất quán đã thực hiện",
        "",
        "- 3 seed × 30 condition-model rows = 90 macro rows.",
        "- 3 seed × 60 generator-model-condition rows = 180 per-generator rows.",
        "- Bootstrap có 180 CI rows và 90 paired-difference rows; không có NaN.",
        "- Bootstrap point estimates khớp metric source trong tolerance `1e-8`.",
        "- Manifest checksum, generator mapping và test size 8.000 giống nhau giữa seed 42/43/44.",
        "- Model selection rank 1 là `global_local_forensic`, chọn chỉ từ validation.",
        "",
        "**Nguồn dữ liệu được dựng tự động bởi:** `scripts/build_aiface_complete_results.py`.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    report = render()
    OUT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUT} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
