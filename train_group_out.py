# -*- coding: utf-8 -*-
"""Attention 模型按采集日期进行 leave-one-date-out 训练与预测。"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import train_attention as model


SCRIPT_DIR = Path(__file__).resolve().parent
FUSION_DIR = SCRIPT_DIR.parent / "OCTA_Fusion"
if str(FUSION_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_DIR))

from group_out_common import save_group_predictions, validate_group_table


DATA_PATH = SCRIPT_DIR / "feature_out" / "features_summary_renamed.csv"
LEAVE_DATE_MANIFEST_PATH = (
    "/home/wxy/Classification/syy/syy/strock_test/OCTA_pytorch/"
    "OCTA_Attention/feature_out/leave_date.csv"
)
OUTPUT_ROOT = SCRIPT_DIR / "output" / "attention_group_out_results"
PREDICTION_PATH = OUTPUT_ROOT / "leave_date_out_predictions.csv"
EXPECTED_DATES = {"2025-10-28", "2025-11-01", "2025-11-03"}
GROUP_TYPE = "date"
GROUP_COLUMN = "date"
ABLATION_MODE = "full"


def load_leave_date_manifest(manifest_path):
    manifest = pd.read_csv(
        manifest_path,
        dtype={"animal_id": str, "video_key": str, "date": str},
    )
    manifest.columns = manifest.columns.astype(str).str.strip()
    required = {"animal_id", "video_key", "label", "group", "date"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"leave_date.csv 缺少字段：{sorted(missing)}")

    for column in ("animal_id", "video_key", "date"):
        manifest[column] = manifest[column].astype(str).str.strip()
    manifest["label"] = pd.to_numeric(
        manifest["label"], errors="raise"
    ).astype(int)
    manifest["group"] = pd.to_numeric(
        manifest["group"], errors="raise"
    ).astype(int)

    if manifest["video_key"].duplicated().any():
        duplicated = manifest.loc[
            manifest["video_key"].duplicated(keep=False), "video_key"
        ].tolist()
        raise ValueError(f"leave_date.csv 存在重复视频：{duplicated[:10]}")
    if set(manifest["group"]) != {1, 2, 3}:
        raise ValueError("leave_date.csv 的 group 必须恰好为 1、2、3")
    if set(manifest["date"]) != EXPECTED_DATES:
        raise ValueError(
            "leave_date.csv 的日期必须恰好为："
            f"{sorted(EXPECTED_DATES)}"
        )
    if not manifest.groupby("group")["date"].nunique().eq(1).all():
        raise ValueError("leave_date.csv 中同一 group 对应了多个日期")
    if not manifest.groupby("date")["group"].nunique().eq(1).all():
        raise ValueError("leave_date.csv 中同一日期对应了多个 group")
    return manifest


def load_data():
    model.DATA_PATH = str(DATA_PATH)
    dataframe = model.load_input_data()
    manifest = load_leave_date_manifest(LEAVE_DATE_MANIFEST_PATH)
    video_meta = model.build_video_table(dataframe)

    data_keys = set(video_meta["video_key"].astype(str))
    manifest_keys = set(manifest["video_key"].astype(str))
    if data_keys != manifest_keys:
        missing = sorted(data_keys - manifest_keys)
        extra = sorted(manifest_keys - data_keys)
        raise ValueError(
            "Attention 数据与 leave_date.csv 视频集合不一致："
            f"清单缺少={missing[:10]}，清单多出={extra[:10]}"
        )

    data_labels = dict(
        zip(video_meta["video_key"], video_meta["label"].astype(int))
    )
    expected_labels = manifest["video_key"].map(data_labels)
    if not manifest["label"].eq(expected_labels).all():
        bad = manifest.loc[
            ~manifest["label"].eq(expected_labels), "video_key"
        ].tolist()
        raise ValueError(
            f"Attention 数据与 leave_date.csv 标签不一致：{bad[:10]}"
        )

    metadata = manifest[["video_key", "group", "date"]]
    dataframe = dataframe.merge(
        metadata,
        on="video_key",
        how="left",
        validate="many_to_one",
    )
    return dataframe, manifest


def build_group_video_meta(dataframe):
    metadata = dataframe[
        ["video_key", "label", "group", "date"]
    ].drop_duplicates()
    if metadata["video_key"].duplicated().any():
        bad = metadata.loc[
            metadata["video_key"].duplicated(keep=False), "video_key"
        ].tolist()
        raise ValueError(f"同一视频对应多个标签或日期：{bad[:10]}")
    return metadata.sort_values("video_key").reset_index(drop=True)


def run_inner_oof(
    dataframe,
    train_video_meta,
    group_index,
    class_weight,
    candidate_index,
    output_dir,
):
    class_counts = train_video_meta["label"].value_counts()
    n_splits = min(model.INNER_FOLD, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Group-out 训练集内部至少需要每类 2 个视频")

    inner_cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=model.SEED + group_index * 100,
    )
    keys = train_video_meta["video_key"].astype(str).to_numpy()
    labels = train_video_meta["label"].to_numpy(np.int64)
    key_to_index = {key: index for index, key in enumerate(keys)}
    probabilities = np.full(len(keys), np.nan, dtype=np.float64)
    selected_epochs = []

    split_iterator = inner_cv.split(
        train_video_meta,
        y=train_video_meta["label"],
        groups=train_video_meta["video_key"],
    )
    for inner_index, (fit_index, val_index) in enumerate(
        split_iterator,
        start=1,
    ):
        fit_keys = set(
            train_video_meta.iloc[fit_index]["video_key"].astype(str)
        )
        val_keys = set(
            train_video_meta.iloc[val_index]["video_key"].astype(str)
        )
        if fit_keys & val_keys:
            raise AssertionError("Group-out inner train/validation 视频重叠")

        data = model.prepare_sequences(
            dataframe[dataframe["video_key"].isin(fit_keys)].copy(),
            dataframe[dataframe["video_key"].isin(val_keys)].copy(),
            ABLATION_MODE,
        )
        log_path = (
            Path(output_dir)
            / "inner"
            / f"candidate_{candidate_index + 1}"
            / f"inner_fold_{inner_index}"
            / "epoch_metrics.csv"
        )
        _, best_epoch, val_probability = model.train_with_val(
            data,
            class_weight,
            model.SEED + group_index * 1000 + inner_index,
            log_path=str(log_path),
            log_metadata={
                "ablation": ABLATION_MODE,
                "outer_fold": group_index,
                "inner_fold": inner_index,
            },
        )
        selected_epochs.append(best_epoch)
        for key, probability in zip(data["keys_eval"], val_probability):
            probabilities[key_to_index[str(key)]] = float(probability)

    if np.any(~np.isfinite(probabilities)):
        raise RuntimeError("Group-out inner OOF 未完整覆盖训练视频")
    selected_epoch = max(1, int(np.median(selected_epochs)))
    return labels, probabilities, selected_epoch


def select_training_configuration(
    dataframe,
    train_video_meta,
    group_index,
    output_dir,
):
    candidates = []
    for candidate_index, class_weight in enumerate(
        model.CLASS_WEIGHT_CANDIDATES
    ):
        labels, probabilities, selected_epoch = run_inner_oof(
            dataframe,
            train_video_meta,
            group_index,
            class_weight,
            candidate_index,
            output_dir,
        )
        threshold = model.search_best_threshold(labels, probabilities)
        metrics = model.calculate_metrics(
            labels,
            (probabilities >= threshold).astype(np.int64),
            probabilities,
        )
        candidates.append(
            {
                "class_weight": class_weight,
                "threshold": threshold,
                "selected_epoch": selected_epoch,
                "labels": labels,
                "probabilities": probabilities,
                "metrics": metrics,
            }
        )

    best = max(
        candidates,
        key=lambda item: (
            item["metrics"]["Balanced_ACC"],
            item["metrics"]["F1"],
            item["metrics"]["ACC"],
            item["metrics"]["AUC"]
            if np.isfinite(item["metrics"]["AUC"])
            else -1.0,
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "class_weight": str(item["class_weight"]),
                "threshold": item["threshold"],
                "oof_balanced_accuracy": item["metrics"]["Balanced_ACC"],
                "oof_f1": item["metrics"]["F1"],
                "oof_accuracy": item["metrics"]["ACC"],
                "oof_auc": item["metrics"]["AUC"],
                "epochs": item["selected_epoch"],
                "threshold_objective": model.THRESHOLD_OBJECTIVE,
            }
            for item in candidates
        ]
    ).to_csv(
        output_dir / "threshold_selection_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction = (
        best["probabilities"] >= best["threshold"]
    ).astype(np.int64)
    pd.DataFrame(
        {
            "model": "attention",
            "ablation": ABLATION_MODE,
            "animal_id": train_video_meta["video_key"].astype(str),
            "video_key": train_video_meta["video_key"].astype(str),
            "label": best["labels"],
            "probability": best["probabilities"],
            "prediction": prediction,
            "outer_fold": group_index,
            "fold_threshold": best["threshold"],
            "threshold_objective": model.THRESHOLD_OBJECTIVE,
        }
    ).to_csv(
        output_dir / "inner_oof_video_pred.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return best


def build_config_snapshot():
    return {
        "source_file": str(Path(__file__).resolve()),
        "model_source_file": str(Path(model.__file__).resolve()),
        "data_path": str(DATA_PATH.resolve()),
        "leave_date_manifest": LEAVE_DATE_MANIFEST_PATH,
        "output_root": str(OUTPUT_ROOT.resolve()),
        "expected_dates": sorted(EXPECTED_DATES),
        "group_type": GROUP_TYPE,
        "ablation_mode": ABLATION_MODE,
        "inner_folds": model.INNER_FOLD,
        "final_seeds": model.FINAL_SEEDS,
        "threshold_objective": model.THRESHOLD_OBJECTIVE,
        "threshold_grid": model.THRESHOLD_GRID.tolist(),
        "class_weight_candidates": model.CLASS_WEIGHT_CANDIDATES,
        "max_segments": model.MAX_SEGMENTS,
        "epochs": model.EPOCHS,
        "batch_size": model.BATCH_SIZE,
        "learning_rate": model.LR,
        "weight_decay": model.WEIGHT_DECAY,
        "patience": model.PATIENCE,
        "feature_cols": model.FEATURE_COLS,
        "use_missing_indicators": model.USE_MISSING_INDICATORS,
    }


def save_run_manifest(summary, config_path, start_time):
    config_sha256 = model.sha256_file(config_path)
    training_manifest_sha256 = model.sha256_file(
        LEAVE_DATE_MANIFEST_PATH
    )
    code_commit = model.get_git_commit()
    end_time = datetime.now().isoformat(timespec="seconds")
    rows = []
    for row in summary.itertuples(index=False):
        bundle_path = OUTPUT_ROOT / row.model_bundle
        rows.append(
            {
                "run_id": f"attention_full_leave_date_out_{row.outer_fold}",
                "model": "attention",
                "ablation": ABLATION_MODE,
                "outer_fold": int(row.outer_fold),
                "held_out_group": str(row.held_out_group),
                "seed": model.SEED,
                "config_file": str(config_path.resolve()),
                "config_sha256": config_sha256,
                "training_manifest_sha256": training_manifest_sha256,
                "code_commit": code_commit,
                "start_time": start_time,
                "end_time": end_time,
                "selected_epoch": int(row.selected_epoch),
                "selected_threshold": float(row.threshold),
                "threshold_objective": model.THRESHOLD_OBJECTIVE,
                "checkpoint_path": str(bundle_path.resolve()),
                "relative_path": bundle_path.relative_to(
                    OUTPUT_ROOT
                ).as_posix(),
                "bytes": bundle_path.stat().st_size,
                "sha256": model.sha256_file(bundle_path),
            }
        )
    logs_dir = OUTPUT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        logs_dir / "run_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    start_time = datetime.now().isoformat(timespec="seconds")
    model.seed_everything(model.SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataframe, manifest = load_data()
    video_meta = build_group_video_meta(dataframe)
    held_out_dates = validate_group_table(dataframe, GROUP_COLUMN)

    config_path = OUTPUT_ROOT / "config_snapshot.json"
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(
            build_config_snapshot(),
            file_obj,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file_obj.write("\n")
    shutil.copyfile(
        LEAVE_DATE_MANIFEST_PATH,
        OUTPUT_ROOT / "leave_date.csv",
    )

    prediction_rows = []
    summary_rows = []
    for group_index, held_out_date in enumerate(held_out_dates, start=1):
        print(
            f"\n========== Attention leave-date-out: {held_out_date} "
            f"({group_index}/{len(held_out_dates)}) =========="
        )
        train_keys = set(
            video_meta.loc[
                video_meta[GROUP_COLUMN].astype(str) != str(held_out_date),
                "video_key",
            ].astype(str)
        )
        test_keys = set(
            video_meta.loc[
                video_meta[GROUP_COLUMN].astype(str) == str(held_out_date),
                "video_key",
            ].astype(str)
        )
        train_video_meta = video_meta[
            video_meta["video_key"].isin(train_keys)
        ].copy().reset_index(drop=True)
        if train_video_meta["label"].nunique() < 2:
            raise ValueError(
                f"留出 {held_out_date} 后训练集只有一个类别，无法训练"
            )
        dataframe_train = dataframe[
            dataframe["video_key"].isin(train_keys)
        ].copy()
        dataframe_test = dataframe[
            dataframe["video_key"].isin(test_keys)
        ].copy()
        held_out_dir = (
            OUTPUT_ROOT
            / "checkpoints"
            / f"held_out_{group_index:03d}"
        )
        selected = select_training_configuration(
            dataframe_train,
            train_video_meta,
            group_index,
            held_out_dir,
        )

        data = model.prepare_sequences(
            dataframe_train,
            dataframe_test,
            ABLATION_MODE,
        )
        final_probabilities = []
        state_dicts = []
        for seed_index in range(model.FINAL_SEEDS):
            attention_model = model.train_fixed(
                data,
                selected["class_weight"],
                model.SEED + group_index * 10000 + seed_index,
                selected["selected_epoch"],
            )
            final_probabilities.append(
                model.predict(
                    attention_model,
                    data["X_eval"],
                    data["M_eval"],
                )
            )
            state_dicts.append(
                {
                    key: value.detach().cpu().clone()
                    for key, value in attention_model.state_dict().items()
                }
            )

        probability = np.mean(np.stack(final_probabilities), axis=0)
        prediction = (
            probability >= selected["threshold"]
        ).astype(np.int64)
        bundle_path = held_out_dir / "attention_model_bundle.pkl"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_attention_bundle(
            str(bundle_path),
            state_dicts,
            data,
            group_index,
            selected["threshold"],
            selected["selected_epoch"],
            selected["class_weight"],
        )

        for key, label, prob, pred in zip(
            data["keys_eval"],
            data["y_eval"],
            probability,
            prediction,
        ):
            prediction_rows.append(
                {
                    "group_type": GROUP_TYPE,
                    "held_out_group": str(held_out_date),
                    "animal_id": str(key),
                    "label": int(label),
                    "probability": float(prob),
                    "threshold": float(selected["threshold"]),
                    "prediction": int(pred),
                }
            )

        metrics = model.calculate_metrics(
            data["y_eval"],
            prediction,
            probability,
        )
        summary_rows.append(
            {
                "model": "attention",
                "ablation": ABLATION_MODE,
                "outer_fold": group_index,
                "held_out_group": str(held_out_date),
                "class_weight": str(selected["class_weight"]),
                "selected_epoch": selected["selected_epoch"],
                "threshold": selected["threshold"],
                "test_acc": metrics["ACC"],
                "test_auc": metrics["AUC"],
                "test_prauc": metrics["PR-AUC"],
                "test_f1": metrics["F1"],
                "test_bal_acc": metrics["Balanced_ACC"],
                "test_sensitivity": metrics["Sensitivity"],
                "test_specificity": metrics["Specificity"],
                "n_test_videos": len(data["keys_eval"]),
                "model_bundle": bundle_path.relative_to(
                    OUTPUT_ROOT
                ).as_posix(),
            }
        )
        print(
            f"date={held_out_date} epoch={selected['selected_epoch']} "
            f"threshold={selected['threshold']:.2f} "
            f"ACC={metrics['ACC']:.4f} AUC={metrics['AUC']:.4f} "
            f"PR-AUC={metrics['PR-AUC']:.4f}"
        )

    predictions = save_group_predictions(
        prediction_rows,
        PREDICTION_PATH,
    )
    if predictions["animal_id"].duplicated().any():
        raise AssertionError("leave-date-out 预测包含重复视频")
    if set(predictions["animal_id"].astype(str)) != set(
        manifest["video_key"].astype(str)
    ):
        raise AssertionError("leave-date-out 预测未完整覆盖全部视频")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_ROOT / "leave_date_out_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary, config_path, start_time)
    print(f"Attention leave-date-out 预测已保存：{PREDICTION_PATH}")


if __name__ == "__main__":
    main()
