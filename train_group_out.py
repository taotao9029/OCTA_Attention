# -*- coding: utf-8 -*-
"""Attention-set 模型的独立 Group-out 训练入口。"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_attention as model
from group_out_common import (
    attach_manifest,
    load_group_manifest,
    save_group_predictions,
    validate_group_table,
)


DATA_PATH = model.DATA_PATH
MANIFEST_PATH = "./data/rename_manifest_with_groups.csv"
OUTPUT_ROOT = "./output/attention_group_out_results"
GROUP_TYPES = {
    "session": "session_id",
    "date": "date",
}
ABLATION_MODE = "full"


def load_data():
    dataframe = pd.read_csv(DATA_PATH)
    dataframe.columns = dataframe.columns.astype(str).str.strip()
    required = set(model.FEATURE_COLS + ["filename", "label"])
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"特征文件缺少字段：{sorted(missing)}")
    dataframe["filename"] = dataframe["filename"].astype(str)
    dataframe["video_key"] = dataframe["filename"].apply(
        model.get_original_video_key
    )
    dataframe["label"] = dataframe["label"].astype(int)
    manifest = load_group_manifest(MANIFEST_PATH)
    dataframe = attach_manifest(dataframe, manifest)
    return dataframe


def build_video_meta(dataframe, group_column):
    group_counts = dataframe.groupby("video_key")[group_column].nunique()
    if group_counts.max() > 1:
        bad = group_counts[group_counts > 1].index.tolist()
        raise ValueError(
            f"同一视频对应多个 {group_column}，无法进行 Group-out：{bad[:10]}"
        )
    return (
        dataframe[
            ["video_key", "label", group_column]
        ]
        .drop_duplicates("video_key")
        .sort_values("video_key")
        .reset_index(drop=True)
    )


def select_training_configuration(df_train, train_meta, group_index):
    candidates = []
    for class_weight in model.CLASS_WEIGHT_CANDIDATES:
        labels, probabilities, epochs = model.inner_oof(
            df_train,
            train_meta,
            group_index,
            class_weight,
            ABLATION_MODE,
        )
        threshold = model.search_best_threshold(labels, probabilities)
        metric = model.calculate_metrics(
            labels,
            (probabilities >= threshold).astype(np.int64),
            probabilities,
        )
        candidates.append(
            {
                "class_weight": class_weight,
                "threshold": threshold,
                "epochs": epochs,
                "labels": labels,
                "probabilities": probabilities,
                "metric": metric,
            }
        )
    return max(
        candidates,
        key=lambda item: (
            item["metric"]["Balanced_ACC"],
            item["metric"]["F1"],
            item["metric"]["ACC"],
            item["metric"]["AUC"]
            if np.isfinite(item["metric"]["AUC"])
            else -1.0,
        ),
    )


def run_one_group_type(dataframe, group_type, group_column):
    groups = validate_group_table(dataframe, group_column)
    video_meta = build_video_meta(dataframe, group_column)
    output_dir = Path(OUTPUT_ROOT) / group_type
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for group_index, held_out_group in enumerate(groups, start=1):
        print(
            f"\n========== Attention {group_type}: "
            f"held-out {held_out_group} ({group_index}/{len(groups)}) =========="
        )
        train_keys = set(
            video_meta.loc[
                video_meta[group_column].astype(str) != str(held_out_group),
                "video_key",
            ]
        )
        test_keys = set(
            video_meta.loc[
                video_meta[group_column].astype(str) == str(held_out_group),
                "video_key",
            ]
        )
        df_train = dataframe[dataframe["video_key"].isin(train_keys)].copy()
        df_test = dataframe[dataframe["video_key"].isin(test_keys)].copy()
        train_meta = video_meta[video_meta["video_key"].isin(train_keys)].copy()
        if train_meta["label"].nunique() < 2:
            raise ValueError(
                f"留出 {held_out_group} 后训练集只有一个类别，无法训练"
            )

        best = select_training_configuration(
            df_train,
            train_meta,
            group_index,
        )
        data = model.prepare_sequences(df_train, df_test, ABLATION_MODE)
        probabilities = []
        state_dicts = []
        for seed_index in range(model.FINAL_SEEDS):
            seed = model.SEED + group_index * 10000 + seed_index
            fitted = model.train_fixed(
                data,
                best["class_weight"],
                seed,
                best["epochs"],
            )
            probabilities.append(
                model.predict(fitted, data["X_eval"], data["M_eval"])
            )
            state_dicts.append({
                key: value.detach().cpu().clone()
                for key, value in fitted.state_dict().items()
            })

        probability = np.mean(np.stack(probabilities), axis=0)
        threshold = float(best["threshold"])
        prediction = (probability >= threshold).astype(np.int64)
        checkpoint_path = checkpoint_dir / f"held_out_{group_index:03d}.pkl"
        model.save_attention_bundle(
            checkpoint_path,
            state_dicts,
            data,
            group_index,
            threshold,
            best["epochs"],
            best["class_weight"],
        )

        for key, label, prob, pred in zip(
            np.asarray(data["keys_eval"]).astype(str),
            data["y_eval"].astype(int),
            probability,
            prediction,
        ):
            rows.append(
                {
                    "group_type": group_type,
                    "held_out_group": str(held_out_group),
                    "animal_id": key,
                    "label": int(label),
                    "probability": float(prob),
                    "threshold": threshold,
                    "prediction": int(pred),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_group_predictions(
        rows,
        output_dir / "leave_{}_out_predictions.csv".format(group_type),
    )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # Attention 的 inner OOF 日志和 bundle 全部写入独立 Group-out 目录。
    model.SAVE_ROOT = OUTPUT_ROOT
    dataframe = load_data()
    for group_type, group_column in GROUP_TYPES.items():
        run_one_group_type(dataframe, group_type, group_column)
    print(f"Attention Group-out 结果已保存到：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
