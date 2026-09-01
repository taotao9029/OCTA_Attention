# -*- coding: utf-8 -*-
"""Group-out 实验共用的 manifest 读取、关联和结果保存工具。"""

from pathlib import Path
import re

import pandas as pd


PREDICTION_COLUMNS = [
    "group_type",
    "held_out_group",
    "animal_id",
    "label",
    "probability",
    "threshold",
    "prediction",
]


def _normalise_name(value):
    return Path(str(value)).name.strip()


def _parse_datetime_from_old_name(value):
    match = re.search(
        r"_(\d{4})_(\d{1,2})_(\d{1,2})_"
        r"(\d{1,2})_(\d{1,2})_(\d{1,2})",
        str(value),
    )
    if match is None:
        return pd.NaT
    year, month, day, hour, minute, second = map(int, match.groups())
    return pd.Timestamp(year, month, day, hour, minute, second)


def load_group_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 Group-out manifest：{manifest_path}")

    manifest = pd.read_csv(
        manifest_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
    )
    manifest.columns = [str(column).strip() for column in manifest.columns]
    if "new_name" not in manifest.columns:
        raise ValueError(
            "rename_manifest_with_groups.csv 必须包含 new_name 列，"
            f"当前列为：{list(manifest.columns)}"
        )

    manifest["new_name"] = manifest["new_name"].map(_normalise_name)
    if manifest["new_name"].duplicated().any():
        duplicated = manifest[manifest["new_name"].duplicated(keep=False)]
        raise ValueError(
            "manifest 的 new_name 存在重复，无法唯一关联文件：\n"
            + duplicated.to_string(index=False)
        )

    if "date" not in manifest.columns:
        if "old_name" not in manifest.columns:
            raise ValueError("manifest 缺少 date，且没有 old_name 可用于提取日期")
        parsed = manifest["old_name"].map(_parse_datetime_from_old_name)
        manifest["date"] = parsed.dt.strftime("%Y-%m-%d")
    else:
        manifest["date"] = manifest["date"].astype(str).str.strip()
        manifest.loc[manifest["date"].isin({"", "nan", "NaT"}), "date"] = pd.NA

    if "session_id" not in manifest.columns:
        if "group_key" not in manifest.columns:
            raise ValueError("manifest 缺少 session_id 和 group_key")
        manifest["session_id"] = manifest["group_key"].astype(str).str.strip()
    else:
        manifest["session_id"] = manifest["session_id"].astype(str).str.strip()

    return manifest


def attach_manifest(dataframe, manifest):
    """按 filename=new_name 关联采集日期和 session。"""
    frame = dataframe.copy()
    if "filename" not in frame.columns:
        raise ValueError("模型数据必须包含 filename 列")
    frame["filename"] = frame["filename"].map(_normalise_name)
    columns = ["new_name", "date", "session_id"]
    metadata = manifest[columns].copy()
    frame = frame.merge(
        metadata,
        left_on="filename",
        right_on="new_name",
        how="left",
        validate="many_to_one",
    )
    missing = frame["date"].isna() | frame["session_id"].isna()
    if missing.any():
        examples = frame.loc[missing, "filename"].head(10).tolist()
        raise ValueError(
            f"有 {int(missing.sum())} 个文件没有匹配 manifest，示例：{examples}"
        )
    return frame


def validate_group_table(frame, group_column):
    if group_column not in frame.columns:
        raise ValueError(f"数据缺少 Group-out 字段：{group_column}")
    groups = frame[group_column].dropna().astype(str).str.strip()
    if groups.empty:
        raise ValueError(f"Group-out 字段为空：{group_column}")
    return sorted(groups.unique().tolist())


def save_group_predictions(rows, output_path):
    result = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if result.empty:
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        return result
    result["held_out_group"] = result["held_out_group"].astype(str)
    result = result.sort_values(
        ["held_out_group", "animal_id"]
    ).reset_index(drop=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result
