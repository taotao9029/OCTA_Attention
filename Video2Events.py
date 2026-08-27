# -*- coding: utf-8 -*-
"""将 AVI/MP4 普通视频转换为可读取的事件流 CSV。

说明：该脚本使用相邻采样视频帧的灰度变化生成事件。
输出字段固定为：timestamp(s), x, y, polarity。
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


# ===================== 路径配置 =====================
VIDEO_ROOT = Path("./data/OCTAvideo")
EXCEL_PATH = VIDEO_ROOT / "视频对应标签(脑卒)_扩增版.xlsx"
OUTPUT_ROOT = Path(
    "./data/event_stream"
)


# ===================== 转换配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_FPS = 2.0
MAX_SAMPLE_FRAMES = 120
THETA = 0.02
USE_LOG_RELATIVE_CHANGE = True
EPS = 1e-6
MAX_EVENTS = 2_000_000
OVERWRITE_EXISTING = False
VIDEO_EXTENSIONS = {".avi", ".mp4"}

EVENT_COLUMNS = ["timestamp(s)", "x", "y", "polarity"]


def _empty_events():
    return pd.DataFrame(
        np.empty((0, 4), dtype=np.float32),
        columns=EVENT_COLUMNS,
    )


def _valid_fps(value, default=25.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or value <= 0.0 or value > 1000.0:
        return default
    return value


def _validate_mask(mask_np, height, width):
    if mask_np is None:
        return None
    mask = np.asarray(mask_np)
    if mask.shape != (height, width):
        raise ValueError(
            f"mask 尺寸 {mask.shape} 与视频帧尺寸 {(height, width)} 不一致"
        )
    return mask.astype(bool, copy=False)


def _append_event_chunk(chunks, timestamps, xs, ys, polarities, limit):
    """将一个正/负事件块加入 CPU 列表，并返回累计事件数。"""
    if xs.size == 0:
        return sum(len(chunk) for chunk in chunks)

    used = sum(len(chunk) for chunk in chunks)
    remaining = max(0, limit - used)
    if remaining == 0:
        return used

    count = min(int(xs.size), remaining)
    chunk = np.column_stack(
        [
            timestamps[:count],
            xs[:count],
            ys[:count],
            polarities[:count],
        ]
    ).astype(np.float32, copy=False)
    chunks.append(chunk)
    return used + count


def video2events_fast(
    vid_path,
    output_path=None,
    mask_np=None,
    device=DEVICE,
    return_stats=False,
):
    """逐帧转换视频，避免一次性把全部帧和事件存入 GPU。"""
    cap = cv2.VideoCapture(str(vid_path))
    stats = {
        "video_path": str(vid_path),
        "fps": np.nan,
        "decoded_frames": 0,
        "sampled_frames": 0,
        "event_count": 0,
        "positive_events": 0,
        "negative_events": 0,
        "sample_limit_reached": 0,
        "event_limit_reached": 0,
        "status": "failed_to_open",
    }

    if not cap.isOpened():
        cap.release()
        result = _empty_events()
        return (result, stats) if return_stats else result

    fps = _valid_fps(cap.get(cv2.CAP_PROP_FPS))
    stats["fps"] = fps
    sampling_interval = 1.0 / float(SAMPLE_FPS)
    next_sample_t = 0.0
    frame_index = 0
    previous_gray = None
    mask = None
    chunks = []
    total_events = 0
    stop_processing = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            stats["decoded_frames"] += 1
            current_t = frame_index / fps
            frame_index += 1

            if current_t + 1e-12 < next_sample_t:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if mask is None:
                mask = _validate_mask(mask_np, gray.shape[0], gray.shape[1])

            current_gray = gray.astype(np.float32) / 255.0
            stats["sampled_frames"] += 1
            next_sample_t += sampling_interval

            if previous_gray is None:
                previous_gray = current_gray
            else:
                prev_tensor = torch.from_numpy(previous_gray).to(
                    device=device,
                    dtype=torch.float32,
                )
                curr_tensor = torch.from_numpy(current_gray).to(
                    device=device,
                    dtype=torch.float32,
                )

                if USE_LOG_RELATIVE_CHANGE:
                    change = torch.log(curr_tensor + EPS) - torch.log(
                        prev_tensor + EPS
                    )
                else:
                    change = curr_tensor - prev_tensor

                positive = change > THETA
                negative = change < -THETA
                if mask is not None:
                    mask_tensor = torch.from_numpy(mask).to(device=device)
                    positive = torch.logical_and(positive, mask_tensor)
                    negative = torch.logical_and(negative, mask_tensor)

                y_pos, x_pos = torch.nonzero(positive, as_tuple=True)
                y_neg, x_neg = torch.nonzero(negative, as_tuple=True)
                pos_count = int(x_pos.numel())
                neg_count = int(x_neg.numel())

                if pos_count:
                    count = min(pos_count, max(0, MAX_EVENTS - total_events))
                    if count:
                        ts = np.full(count, current_t, dtype=np.float32)
                        total_events = _append_event_chunk(
                            chunks,
                            ts,
                            x_pos[:count].detach().cpu().numpy(),
                            y_pos[:count].detach().cpu().numpy(),
                            np.ones(count, dtype=np.float32),
                            MAX_EVENTS,
                        )
                        stats["positive_events"] += count

                if neg_count and total_events < MAX_EVENTS:
                    count = min(neg_count, MAX_EVENTS - total_events)
                    if count:
                        ts = np.full(count, current_t, dtype=np.float32)
                        total_events = _append_event_chunk(
                            chunks,
                            ts,
                            x_neg[:count].detach().cpu().numpy(),
                            y_neg[:count].detach().cpu().numpy(),
                            -np.ones(count, dtype=np.float32),
                            MAX_EVENTS,
                        )
                        stats["negative_events"] += count

                if total_events >= MAX_EVENTS:
                    stats["event_limit_reached"] = 1
                    stop_processing = True

                del prev_tensor, curr_tensor, change, positive, negative
                previous_gray = current_gray

            if stats["sampled_frames"] >= MAX_SAMPLE_FRAMES:
                stats["sample_limit_reached"] = 1
                stop_processing = True

            if stop_processing:
                break
    finally:
        cap.release()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if chunks:
        events_array = np.concatenate(chunks, axis=0)
        result = pd.DataFrame(events_array, columns=EVENT_COLUMNS)
    else:
        result = _empty_events()

    stats["event_count"] = int(len(result))
    stats["status"] = "ok" if len(result) else "empty_events"

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False, encoding="utf-8-sig")

    return (result, stats) if return_stats else result


def load_labels(excel_path):
    labels = pd.read_excel(
        excel_path,
        usecols=["AVI文件名", "是否脑卒"],
        engine="openpyxl",
    )
    labels["AVI文件名"] = labels["AVI文件名"].astype(str).str.strip()
    labels["是否脑卒"] = labels["是否脑卒"].astype(str).str.strip()

    unknown = sorted(
        set(labels["是否脑卒"].dropna()) - {"是", "否"}
    )
    if unknown:
        raise ValueError(f"Excel 中存在未知标签：{unknown}")

    label_dict = {}
    for _, row in labels.iterrows():
        filename = row["AVI文件名"]
        key = filename.lower()
        label = 1 if row["是否脑卒"] == "是" else 0
        if key in label_dict and label_dict[key] != label:
            raise ValueError(f"同一视频对应互相冲突的标签：{filename}")
        label_dict[key] = label
    return label_dict


def iter_videos(video_root):
    for path in sorted(Path(video_root).rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def main():
    if SAMPLE_FPS <= 0:
        raise ValueError("SAMPLE_FPS 必须大于 0")
    if MAX_SAMPLE_FRAMES < 2:
        raise ValueError("MAX_SAMPLE_FRAMES 至少为 2")
    if MAX_EVENTS <= 0:
        raise ValueError("MAX_EVENTS 必须大于 0")

    label_dict = load_labels(EXCEL_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_dirs = {0: OUTPUT_ROOT / "0", 1: OUTPUT_ROOT / "1"}
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    audit_path = OUTPUT_ROOT / "conversion_manifest.csv"
    audit_rows = []
    output_owner = {}

    print(f"device={DEVICE}")
    print(f"video_root={VIDEO_ROOT}")
    print(f"output_root={OUTPUT_ROOT}")

    for video_path in iter_videos(VIDEO_ROOT):
        filename = video_path.name
        label = label_dict.get(filename.lower())
        if label is None:
            print(f"[SKIP] 无标签：{video_path}")
            audit_rows.append({
                "source_path": str(video_path),
                "output_path": "",
                "label": "",
                "status": "missing_label",
            })
            continue

        output_path = output_dirs[label] / f"{video_path.stem}.csv"
        output_key = str(output_path).lower()
        if output_key in output_owner and output_owner[output_key] != str(video_path):
            raise RuntimeError(
                f"发现同名视频会产生输出覆盖：\n"
                f"{output_owner[output_key]}\n{video_path}"
            )
        output_owner[output_key] = str(video_path)

        if output_path.exists() and not OVERWRITE_EXISTING:
            print(f"[EXISTS] 跳过：{output_path}")
            audit_rows.append({
                "source_path": str(video_path),
                "output_path": str(output_path),
                "label": label,
                "status": "exists_skipped",
            })
            continue

        print(f"[PROCESS] {video_path}")
        try:
            _, stats = video2events_fast(
                video_path,
                output_path=output_path,
                device=DEVICE,
                return_stats=True,
            )
            stats.update({
                "source_path": str(video_path),
                "output_path": str(output_path),
                "label": label,
            })
            audit_rows.append(stats)
            print(
                f"[DONE] events={stats['event_count']} "
                f"positive={stats['positive_events']} "
                f"negative={stats['negative_events']}"
            )
        except Exception as exc:
            print(f"[FAILED] {video_path}: {exc}")
            audit_rows.append({
                "source_path": str(video_path),
                "output_path": str(output_path),
                "label": label,
                "status": "failed",
                "error": repr(exc),
            })

    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"转换完成，审计记录：{audit_path}")


if __name__ == "__main__":
    main()
