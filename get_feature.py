# -*- coding: utf-8 -*-
"""从事件流 CSV 提取视频级统计特征。

输入 CSV 需要包含：timestamp(s), x, y, polarity。
输出保留模型使用的 7 个核心字段，并增加极性统计字段：
mean_speed, std_speed, mean_thd, std_thd, pulse_freq, pulse_amp,
event_density, positive_event_ratio, negative_event_ratio,
positive_event_density, negative_event_density。

"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.spatial import KDTree


# ============================================================
# 路径配置
# ============================================================
BASE_ROOT = Path(
    "./data"
)
INPUT_ROOT = BASE_ROOT / "event_streams"
OUTPUT_DIR = BASE_ROOT / "feature_out"
OUTPUT_FILE = OUTPUT_DIR / "features_summary_renamed.csv"
AUDIT_FILE = OUTPUT_DIR / "feature_extraction_manifest.csv"

LABEL_DIRS = {
    0: INPUT_ROOT / "0",
    1: INPUT_ROOT / "1",
}


# ============================================================
# 参数配置
# ============================================================
RANDOM_SEED = 42
TIME_BIN = 0.05
MIN_FREQUENCY = 0.2
MAX_FREQUENCY = 2.0
DEFAULT_WINDOW_SIZE = 1.0
WINDOW_MERGE_STEP = 2
MAX_DISPLACEMENT = 4.0
RANSAC_TRIALS = 50
RANSAC_THRESHOLD = 2.5


CORE_FEATURE_COLUMNS = [
    "mean_speed",
    "std_speed",
    "mean_thd",
    "std_thd",
    "pulse_freq",
    "pulse_amp",
    "event_density",
]

POLARITY_FEATURE_COLUMNS = [
    "positive_event_ratio",
    "negative_event_ratio",
    "positive_event_density",
    "negative_event_density",
]


def _stable_rng(filename):
    digest = hashlib.sha256(str(filename).encode("utf-8")).hexdigest()
    seed = (RANDOM_SEED + int(digest[:8], 16)) % (2**32)
    return np.random.default_rng(seed)


def _safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else np.nan


def _safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values)) if len(values) else np.nan


def _time_step(t_arr):
    unique_t = np.unique(np.asarray(t_arr, dtype=np.float64))
    differences = np.diff(unique_t)
    differences = differences[np.isfinite(differences) & (differences > 1e-9)]
    return float(np.median(differences)) if len(differences) else np.nan


def _frequency_limits(t_arr):
    step = _time_step(t_arr)
    if not np.isfinite(step) or step <= 0:
        return MIN_FREQUENCY, MAX_FREQUENCY
    nyquist = 0.5 / step
    upper = min(MAX_FREQUENCY, 0.9 * nyquist)
    return MIN_FREQUENCY, upper


def _event_counts(t_arr, time_bin):
    t_arr = np.asarray(t_arr, dtype=np.float64)
    t_arr = t_arr[np.isfinite(t_arr)]
    if len(t_arr) < 2:
        return None, None, None

    t_arr = np.sort(t_arr)
    t_min = float(t_arr[0])
    t_max = float(t_arr[-1])
    duration = max(t_max - t_min, 0.0)
    if duration <= 0:
        return None, None, duration

    actual_step = _time_step(t_arr)
    if np.isfinite(actual_step):
        time_bin = max(float(time_bin), actual_step)
    bins = np.arange(t_min, t_max + time_bin + 1e-10, time_bin)
    if len(bins) < 3:
        return None, None, duration
    counts, _ = np.histogram(t_arr, bins=bins)
    return counts.astype(np.float64), bins, duration


def detect_dominant_period(t_arr, time_bin=TIME_BIN):
    """返回主频、主周期、频谱幅度和真实持续时间。

    频率上限根据输入时间戳的实际采样间隔自动限制，避免超过 Nyquist 频率。
    无法可靠估计时返回 NaN，而不是伪造 1 Hz。
    """
    t_arr = np.asarray(t_arr, dtype=np.float64)
    t_arr = t_arr[np.isfinite(t_arr)]
    if len(t_arr) < 8:
        return np.nan, np.nan, np.nan, 0.0

    counts, _, duration = _event_counts(t_arr, time_bin)
    if counts is None or duration < 0.5 or len(counts) < 8:
        return np.nan, np.nan, np.nan, duration
    if np.std(counts) < 1e-8:
        return np.nan, np.nan, np.nan, duration

    actual_step = _time_step(t_arr)
    sampling_frequency = (
        1.0 / actual_step
        if np.isfinite(actual_step) and actual_step > 0
        else 1.0 / time_bin
    )
    freqs, power = signal.periodogram(
        counts,
        fs=sampling_frequency,
        scaling="spectrum",
        detrend="constant",
    )
    freq_min, freq_max = _frequency_limits(t_arr)
    valid = (
        (freqs >= freq_min)
        & (freqs <= freq_max)
        & np.isfinite(power)
    )
    if not np.any(valid):
        return np.nan, np.nan, np.nan, duration

    valid_indices = np.where(valid)[0]
    peak_idx = valid_indices[int(np.argmax(power[valid_indices]))]
    dom_freq = float(freqs[peak_idx])
    if dom_freq <= 0 or not np.isfinite(dom_freq):
        return np.nan, np.nan, np.nan, duration

    pulse_amp = float(np.sqrt(max(power[peak_idx], 0.0)))
    return dom_freq, 1.0 / dom_freq, pulse_amp, duration


def split_nonoverlapping_windows(events, time_col="t", window_size=1.0):
    """将每个事件只分配给一个窗口，覆盖整个时间范围，不重复计数。"""
    if len(events) == 0:
        empty = events.copy()
        empty["window_id"] = np.array([], dtype=np.int64)
        return empty, 0

    result = events.sort_values(time_col).reset_index(drop=True).copy()
    t_arr = result[time_col].to_numpy(dtype=np.float64)
    t_min = float(np.min(t_arr))
    window_size = (
        float(window_size)
        if np.isfinite(window_size) and window_size > 0
        else DEFAULT_WINDOW_SIZE
    )
    window_ids = np.floor((t_arr - t_min) / window_size).astype(np.int64)
    result["window_id"] = window_ids
    return result, int(window_ids.max()) + 1


def merge_adjacent_windows(events, merge_step=WINDOW_MERGE_STEP):
    if len(events) == 0:
        return events.copy(), 0
    if merge_step < 1:
        raise ValueError("merge_step 必须大于等于 1")

    result = events.copy()
    result["window_id"] = (
        result["window_id"].astype(np.int64) // int(merge_step)
    )
    return result, int(result["window_id"].nunique())


def calculate_blood_flow_velocity_ransac(
    window_events,
    time_col="t",
    max_displacement=MAX_DISPLACEMENT,
    ransac_thresh=RANSAC_THRESHOLD,
    trials=RANSAC_TRIALS,
):
    """用时间前后两段事件的唯一最近邻估计粗略运动速度。"""
    if len(window_events) < 15:
        return None, None

    df = window_events.sort_values(time_col).reset_index(drop=True)
    midpoint = len(df) // 2
    previous = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()
    if len(previous) < 4 or len(current) < 4:
        return None, None

    previous_xy = previous[["x", "y"]].to_numpy(dtype=np.float64)
    current_xy = current[["x", "y"]].to_numpy(dtype=np.float64)
    previous_t = previous[time_col].to_numpy(dtype=np.float64)
    current_t = current[time_col].to_numpy(dtype=np.float64)

    # 优先在相同 polarity 内匹配，避免正负事件互相匹配。
    if "polarity" in previous.columns and "polarity" in current.columns:
        polarity_groups = [
            (previous["polarity"].to_numpy(), current["polarity"].to_numpy(), 1),
            (previous["polarity"].to_numpy(), current["polarity"].to_numpy(), -1),
        ]
    else:
        polarity_groups = [(None, None, None)]

    matched_previous = []
    matched_current = []
    matched_dt = []
    for previous_polarity, current_polarity, polarity_value in polarity_groups:
        if polarity_value is None:
            prev_indices = np.arange(len(previous_xy))
            curr_indices = np.arange(len(current_xy))
        else:
            prev_indices = np.where(previous_polarity == polarity_value)[0]
            curr_indices = np.where(current_polarity == polarity_value)[0]
        if len(prev_indices) < 2 or len(curr_indices) < 2:
            continue

        tree = KDTree(current_xy[curr_indices])
        distances, nearest = tree.query(previous_xy[prev_indices], k=1)
        distances = np.asarray(distances, dtype=np.float64)
        nearest = np.asarray(nearest, dtype=np.int64)
        candidate_current = curr_indices[nearest]
        candidate_dt = current_t[candidate_current] - previous_t[prev_indices]
        valid = (
            np.isfinite(distances)
            & np.isfinite(candidate_dt)
            & (distances <= max_displacement)
            & (candidate_dt > 1e-6)
        )
        candidates = [
            (float(distance), int(prev_idx), int(curr_idx), float(dt))
            for distance, prev_idx, curr_idx, dt in zip(
                distances[valid],
                prev_indices[valid],
                candidate_current[valid],
                candidate_dt[valid],
            )
        ]
        # 一个 current 事件只允许被匹配一次。
        used_current = set()
        for _, prev_idx, curr_idx, dt in sorted(candidates):
            if curr_idx in used_current:
                continue
            used_current.add(curr_idx)
            matched_previous.append(prev_idx)
            matched_current.append(curr_idx)
            matched_dt.append(dt)

    if len(matched_previous) < 4:
        return None, None

    prev = previous_xy[np.asarray(matched_previous)]
    curr = current_xy[np.asarray(matched_current)]
    dt = np.asarray(matched_dt, dtype=np.float64)
    displacements = curr - prev
    rng = _stable_rng(str(window_events.index[:3].tolist()))
    best_inlier = np.zeros(len(prev), dtype=bool)

    for _ in range(int(trials)):
        if len(prev) < 4:
            break
        sample_idx = rng.choice(len(prev), size=4, replace=False)
        velocity = np.mean(displacements[sample_idx] / dt[sample_idx, None], axis=0)
        residual = np.linalg.norm(
            displacements - velocity[None, :] * dt[:, None],
            axis=1,
        )
        inlier = residual <= float(ransac_thresh)
        if np.sum(inlier) > np.sum(best_inlier):
            best_inlier = inlier

    if np.sum(best_inlier) < 3:
        return None, None
    velocity = displacements[best_inlier] / dt[best_inlier, None]
    speed = np.linalg.norm(velocity, axis=1)
    speed = speed[np.isfinite(speed)]
    if len(speed) < 3:
        return None, None
    return float(np.mean(speed)), float(np.std(speed))


def calculate_window_thd(window_df, time_col="t", time_bin=TIME_BIN):
    """计算 2～4 次谐波相对于基频的 THD。"""
    t = window_df[time_col].to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 20:
        return np.nan

    counts, _, duration = _event_counts(t, time_bin)
    if counts is None or duration < 0.5 or len(counts) < 8:
        return np.nan
    counts = counts - np.mean(counts)
    if np.std(counts) < 1e-8:
        return np.nan

    actual_step = _time_step(t)
    sampling_frequency = (
        1.0 / actual_step
        if np.isfinite(actual_step) and actual_step > 0
        else 1.0 / time_bin
    )
    spectrum = np.abs(np.fft.rfft(counts))
    freqs = np.fft.rfftfreq(len(counts), d=1.0 / sampling_frequency)
    freq_min, freq_max = _frequency_limits(t)
    fundamental_mask = (
        (freqs >= freq_min)
        & (freqs <= freq_max)
        & (freqs > 0)
    )
    if not np.any(fundamental_mask):
        return np.nan

    fundamental_indices = np.where(fundamental_mask)[0]
    fund_idx = fundamental_indices[int(np.argmax(spectrum[fundamental_indices]))]
    fund_amp = float(spectrum[fund_idx])
    fund_freq = float(freqs[fund_idx])
    if fund_amp < 1e-8 or fund_freq <= 0:
        return np.nan

    harmonic_amplitudes = []
    for harmonic in range(2, 5):
        target = harmonic * fund_freq
        if target >= freqs[-1]:
            continue
        nearest_idx = int(np.argmin(np.abs(freqs - target)))
        tolerance = max(0.15, 1.0 / max(duration, 1e-8))
        if abs(freqs[nearest_idx] - target) <= tolerance:
            harmonic_amplitudes.append(float(spectrum[nearest_idx]))

    if not harmonic_amplitudes:
        return 0.0
    value = np.sqrt(np.sum(np.square(harmonic_amplitudes))) / fund_amp
    return float(value) if np.isfinite(value) else np.nan


def process_single_file(file_path, label):
    file_path = Path(file_path)
    filename = file_path.name
    stats = {
        "filename": filename,
        "label": int(label),
        "status": "failed",
        "error": "",
        "event_count": 0,
        "invalid_polarity_count": 0,
        "duration": np.nan,
    }

    try:
        dataframe = pd.read_csv(file_path)
    except Exception as exc:
        stats["error"] = f"read_error: {exc}"
        return None, stats

    if "timestamp(s)" in dataframe.columns:
        dataframe = dataframe.rename(columns={"timestamp(s)": "t"})
    required = {"t", "x", "y", "polarity"}
    missing = required - set(dataframe.columns)
    if missing:
        stats["error"] = f"missing_columns: {sorted(missing)}"
        return None, stats

    for column in ["t", "x", "y", "polarity"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    dataframe = dataframe.dropna(subset=["t", "x", "y", "polarity"])
    valid_polarity = dataframe["polarity"].isin([-1, 1])
    stats["invalid_polarity_count"] = int((~valid_polarity).sum())
    dataframe = dataframe[valid_polarity].copy()
    dataframe["polarity"] = dataframe["polarity"].astype(np.int8)
    dataframe = dataframe.drop_duplicates(
        subset=["t", "x", "y", "polarity"]
    )
    dataframe = dataframe.sort_values("t").reset_index(drop=True)

    if len(dataframe) < 2:
        stats["error"] = "too_few_events"
        return None, stats

    t_arr = dataframe["t"].to_numpy(dtype=np.float64)
    duration = float(t_arr[-1] - t_arr[0])
    stats["event_count"] = int(len(dataframe))
    stats["duration"] = duration
    if not np.isfinite(duration) or duration <= 0:
        stats["error"] = "non_positive_duration"
        return None, stats

    dom_freq, dom_period, pulse_amp, _ = detect_dominant_period(t_arr)
    window_size = (
        dom_period if np.isfinite(dom_period) else DEFAULT_WINDOW_SIZE
    )
    windowed, _ = split_nonoverlapping_windows(
        dataframe,
        time_col="t",
        window_size=window_size,
    )
    merged, _ = merge_adjacent_windows(windowed)

    speed_means = []
    speed_stds = []
    thd_values = []
    for _, group in merged.groupby("window_id", sort=True):
        speed_mean, speed_std = calculate_blood_flow_velocity_ransac(group)
        if speed_mean is not None and np.isfinite(speed_mean):
            speed_means.append(speed_mean)
        if speed_std is not None and np.isfinite(speed_std):
            speed_stds.append(speed_std)
        thd = calculate_window_thd(group)
        if np.isfinite(thd):
            thd_values.append(thd)

    positive_count = int((dataframe["polarity"] > 0).sum())
    negative_count = int((dataframe["polarity"] < 0).sum())
    total_events = len(dataframe)
    result = {
        "filename": filename,
        "mean_speed": _safe_mean(speed_means),
        "std_speed": _safe_mean(speed_stds),
        "mean_thd": _safe_mean(thd_values),
        "std_thd": _safe_std(thd_values),
        "pulse_freq": float(dom_freq) if np.isfinite(dom_freq) else np.nan,
        "pulse_amp": float(pulse_amp) if np.isfinite(pulse_amp) else np.nan,
        "event_density": float(total_events / duration),
        "positive_event_ratio": float(positive_count / total_events),
        "negative_event_ratio": float(negative_count / total_events),
        "positive_event_density": float(positive_count / duration),
        "negative_event_density": float(negative_count / duration),
        "label": int(label),
    }
    stats["status"] = "ok"
    return result, stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_features = []
    audit_rows = []

    for label, input_dir in LABEL_DIRS.items():
        if not input_dir.is_dir():
            print(f"目录不存在，跳过：{input_dir}")
            continue

        csv_files = sorted(
            path for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".csv"
        )
        print(f"类别 {label}: 找到 {len(csv_files)} 个 CSV")

        for path in csv_files:
            result, stats = process_single_file(path, label)
            audit_rows.append(stats)
            if result is None:
                print(f"跳过 {path.name}: {stats['error']}")
                continue
            all_features.append(result)
            print(f"处理完成: {path.name}")

    pd.DataFrame(audit_rows).to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if not all_features:
        print("没有生成任何有效特征")
        return

    feature_df = pd.DataFrame(all_features).sort_values(
        ["label", "filename"]
    ).reset_index(drop=True)
    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        na_rep="",
    )

    print(f"\n生成完成: {len(feature_df)} 个文件")
    print(f"特征文件: {OUTPUT_FILE}")
    print(f"审计文件: {AUDIT_FILE}")
    print("\n标签统计:")
    print(feature_df["label"].value_counts().sort_index())
    print("\n特征缺失率:")
    print(feature_df.isna().mean().sort_values(ascending=False))


if __name__ == "__main__":
    main()
