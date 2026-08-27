"""视频级 Attention 特征消融实验。

保留 Transformer + attention pooling 网络及原有视频级嵌套交叉验证流程，
仅改变输入特征集合。每个消融模式独立训练、选择阈值、保存模型和评估结果。
"""

import os
import re
import random
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


DATA_PATH = "./data/features_summary_renamed.csv"
SAVE_ROOT = "./output"
SEED = 42
OUTER_FOLD = 5
INNER_FOLD = 5

MAX_SEGMENTS = 8
EPOCHS = 180
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 5e-4
PATIENCE = 30
FINAL_SEEDS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BOOTSTRAP_REPS = 2000
CALIBRATION_BINS = 10
THRESHOLD_OBJECTIVE = "balanced_acc"
THRESHOLD_GRID = np.arange(0.0, 1.001, 0.01)
RUN_START_TIME = ""

EPOCH_METRIC_COLUMNS = [
    "model", "ablation", "outer_fold", "inner_fold", "seed", "epoch",
    "train_loss", "val_loss", "val_auc", "val_ap",
    "val_balanced_accuracy", "learning_rate", "checkpoint_saved",
]
RUN_MANIFEST_COLUMNS = [
    "run_id", "model", "ablation", "outer_fold", "seed", "config_file",
    "config_sha256", "code_commit", "start_time", "end_time",
    "selected_epoch", "selected_threshold", "threshold_objective",
    "checkpoint_path",
]
PREDICTION_COLUMNS = [
    "model", "ablation", "animal_id", "label", "outer_fold",
    "probability", "fold_threshold", "prediction", "threshold_objective",
    "checkpoint_id",
]

FEATURE_COLS = [
    "mean_speed",
    "std_speed",
    "mean_thd",
    "std_thd",
    "pulse_freq",
    "pulse_amp",
    "event_density",
]
USE_MISSING_INDICATORS = True
CLASS_WEIGHT_CANDIDATES = [
    None,
    "balanced",
    {0: 1.5, 1: 1.0},
    {0: 2.0, 1: 1.0},
]

ABLATION_MODES = [
    "full",
    "wo_mean_speed",
    "wo_std_speed",
    "wo_mean_thd",
    "wo_std_thd",
    "wo_pulse_freq",
    "wo_pulse_amp",
    "wo_event_density",
    "wo_missing",
    "wo_derived",
    "base_only",
]

DERIVED_FEATURE_DEPENDENCIES = {
    "mean_speed": {
        "d_density_speed",
        "d_speed_density",
        "d_speed_sq",
        "d_stdspeed_speed",
    },
    "std_speed": {
        "d_stdspeed_speed",
        "d_sqrt_freq_speedstd",
    },
    "mean_thd": {
        "d_amp_thd",
        "d_stdthd_thd",
        "d_thd_density",
        "d_thd_sq",
        "d_amp_minus_thd",
    },
    "std_thd": {
        "d_stdthd_thd",
    },
    "pulse_freq": {
        "d_freq_amp",
        "d_freq_density",
        "d_freq_density_ratio",
        "d_freq_sq",
        "d_sqrt_freq_speedstd",
    },
    "pulse_amp": {
        "d_amp_thd",
        "d_freq_amp",
        "d_amp_density",
        "d_amp_sq",
        "d_amp_minus_thd",
    },
    "event_density": {
        "d_density_speed",
        "d_amp_density",
        "d_freq_density",
        "d_thd_density",
        "d_speed_density",
        "d_density_sq",
        "d_freq_density_ratio",
    },
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_original_video_key(filename):
    key = os.path.splitext(os.path.basename(str(filename)))[0]
    return re.sub(
        r"(?:_aug|_augmentation)[-_]?\d+$",
        "",
        key,
        flags=re.IGNORECASE,
    )


def fit_fill_values(df_train):
    values = {}
    for col in FEATURE_COLS:
        value = pd.to_numeric(df_train[col], errors="coerce")
        value = value.replace([np.inf, -np.inf], np.nan)
        median = value.median()
        values[col] = float(median) if np.isfinite(median) else 0.0
    return values


def safe_divide(a, b, eps=1e-3):
    denominator = np.where(
        np.abs(b) < eps,
        np.where(b >= 0, eps, -eps),
        b,
    )
    return np.clip(a / denominator, -1e4, 1e4)


def build_segment_features(df, fill_values):
    base = pd.DataFrame(index=df.index)
    for col in FEATURE_COLS:
        value = pd.to_numeric(df[col], errors="coerce")
        value = value.replace([np.inf, -np.inf], np.nan)
        if USE_MISSING_INDICATORS:
            base[f"{col}__missing"] = value.isna().astype(np.float32)
        base[col] = value.fillna(fill_values[col]).astype(np.float64)

    for col in ["event_density", "pulse_amp"]:
        base[col] = np.log1p(np.clip(base[col].to_numpy(), 0.0, None))

    speed = base["mean_speed"].to_numpy()
    speed_std = base["std_speed"].to_numpy()
    thd = base["mean_thd"].to_numpy()
    thd_std = base["std_thd"].to_numpy()
    freq = base["pulse_freq"].to_numpy()
    amp = base["pulse_amp"].to_numpy()
    density = base["event_density"].to_numpy()

    derived = pd.DataFrame(index=df.index)
    derived["d_amp_thd"] = amp * thd
    derived["d_density_speed"] = safe_divide(density, speed)
    derived["d_stdthd_thd"] = safe_divide(thd_std, thd)
    derived["d_freq_amp"] = freq * amp
    derived["d_amp_density"] = amp * density
    derived["d_freq_density"] = freq * density
    derived["d_thd_density"] = thd * density
    derived["d_speed_density"] = speed * density
    derived["d_amp_sq"] = np.clip(amp ** 2, -1e4, 1e4)
    derived["d_density_sq"] = np.clip(density ** 2, -1e4, 1e4)
    derived["d_speed_sq"] = np.clip(speed ** 2, -1e4, 1e4)
    derived["d_thd_sq"] = np.clip(thd ** 2, -1e4, 1e4)
    derived["d_stdspeed_speed"] = safe_divide(speed_std, speed)
    derived["d_freq_density_ratio"] = safe_divide(freq, density)
    derived["d_freq_sq"] = np.clip(freq ** 2, -1e4, 1e4)
    derived["d_amp_minus_thd"] = amp - thd
    derived["d_sqrt_freq_speedstd"] = np.sqrt(np.abs(freq * speed_std))

    return pd.concat([base, derived], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-1e4, 1e4).astype(np.float32)


def select_feature_columns(all_columns, ablation_mode):
    if ablation_mode not in ABLATION_MODES:
        raise ValueError(f"未知消融模式: {ablation_mode}")
    if ablation_mode == "full":
        return list(all_columns)
    if ablation_mode == "wo_missing":
        return [col for col in all_columns if not col.endswith("__missing")]
    if ablation_mode == "wo_derived":
        return [col for col in all_columns if not col.startswith("d_")]
    if ablation_mode == "base_only":
        return [
            col for col in all_columns
            if col in FEATURE_COLS
        ]
    if ablation_mode.startswith("wo_"):
        base_feature = ablation_mode[3:]
        if base_feature not in FEATURE_COLS:
            raise ValueError(f"未知特征消融模式: {ablation_mode}")
        removed = {
            base_feature,
            f"{base_feature}__missing",
            *DERIVED_FEATURE_DEPENDENCIES.get(base_feature, set()),
        }
        return [col for col in all_columns if col not in removed]
    raise AssertionError(ablation_mode)


def collect_sequence_arrays(frame, feature_cols, max_segments):
    xs, masks, ys, keys = [], [], [], []
    for key, group in frame.groupby("video_key", sort=True):
        values = group[feature_cols].to_numpy(np.float32)
        if len(values) > max_segments:
            values = values[:max_segments]
        mask = np.zeros(max_segments, dtype=bool)
        mask[:len(values)] = True
        padded = np.zeros((max_segments, values.shape[1]), dtype=np.float32)
        padded[:len(values)] = values
        xs.append(padded)
        masks.append(mask)
        ys.append(int(group["label"].iloc[0]))
        keys.append(key)
    return (
        np.stack(xs),
        np.stack(masks),
        np.asarray(ys, dtype=np.int64),
        np.asarray(keys),
    )


def prepare_sequences(df_fit, df_eval, ablation_mode):
    fill_values = fit_fill_values(df_fit)
    fit_row = build_segment_features(df_fit, fill_values).copy()
    eval_row = build_segment_features(df_eval, fill_values).copy()
    fit_row["video_key"] = df_fit["video_key"].to_numpy()
    fit_row["label"] = df_fit["label"].to_numpy()
    eval_row["video_key"] = df_eval["video_key"].to_numpy()
    eval_row["label"] = df_eval["label"].to_numpy()

    all_feature_cols = [
        col for col in fit_row.columns
        if col not in {"video_key", "label"}
    ]
    feature_cols = select_feature_columns(all_feature_cols, ablation_mode)
    max_segments = max(
        MAX_SEGMENTS,
        int(fit_row.groupby("video_key").size().max()),
        int(eval_row.groupby("video_key").size().max()),
    )
    x_fit, m_fit, y_fit, keys_fit = collect_sequence_arrays(
        fit_row, feature_cols, max_segments
    )
    x_eval, m_eval, y_eval, keys_eval = collect_sequence_arrays(
        eval_row, feature_cols, max_segments
    )
    scaler = StandardScaler()
    scaler.fit(x_fit[m_fit])
    x_fit = scaler.transform(x_fit.reshape(-1, x_fit.shape[-1])).reshape(x_fit.shape)
    x_eval = scaler.transform(x_eval.reshape(-1, x_eval.shape[-1])).reshape(x_eval.shape)
    return {
        "X_fit": x_fit.astype(np.float32),
        "M_fit": m_fit,
        "y_fit": y_fit,
        "keys_fit": keys_fit,
        "X_eval": x_eval.astype(np.float32),
        "M_eval": m_eval,
        "y_eval": y_eval,
        "keys_eval": keys_eval,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "fill_values": fill_values,
        "max_segments": max_segments,
        "ablation_mode": ablation_mode,
    }


def prepare_saved_eval_sequences(df_eval, bundle):
    fill_values = {
        str(key): float(value)
        for key, value in bundle["fill_values"].items()
    }
    eval_row = build_segment_features(df_eval, fill_values).copy()
    eval_row["video_key"] = df_eval["video_key"].to_numpy()
    eval_row["label"] = df_eval["label"].to_numpy()
    x, mask, y, keys = collect_sequence_arrays(
        eval_row,
        bundle["feature_cols"],
        int(bundle["max_segments"]),
    )
    shape = x.shape
    x = (
        x.reshape(-1, x.shape[-1])
        - np.asarray(bundle["scaler_mean"], dtype=np.float32)
    ) / np.asarray(bundle["scaler_scale"], dtype=np.float32)
    return x.reshape(shape).astype(np.float32), mask, y, keys


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_ACC": float(balanced_accuracy_score(y_true, y_pred)),
        "Sensitivity": float(tp / max(tp + fn, 1)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def search_best_threshold(y_true, y_prob):
    """
    与 Fusion 模型保持一致：
    使用 inner-OOF 预测，以 Balanced Accuracy 为主要阈值选择目标。
    """
    best_threshold = 0.5
    best_score = None

    for threshold in THRESHOLD_GRID:
        pred = (y_prob >= threshold).astype(np.int64)
        metric = calculate_metrics(y_true, pred, y_prob)

        if THRESHOLD_OBJECTIVE == "balanced_acc":
            objective = metric["Balanced_ACC"]
        elif THRESHOLD_OBJECTIVE == "f1":
            objective = metric["F1"]
        else:
            objective = metric["ACC"]

        score = (
            objective,
            metric["F1"],
            metric["Balanced_ACC"],
            metric["ACC"],
            -abs(float(threshold) - 0.5),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold

def build_calibration_curve(df_video, n_bins=CALIBRATION_BINS):
    y_true = df_video["label"].to_numpy(np.int64)
    y_prob = np.clip(df_video["prob"].to_numpy(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)
    rows = []
    for index in range(n_bins):
        mask = bin_ids == index
        if not np.any(mask):
            rows.append({
                "bin": index,
                "bin_lower": edges[index],
                "bin_upper": edges[index + 1],
                "n_videos": 0,
                "mean_predicted_prob": np.nan,
                "fraction_positive": np.nan,
                "absolute_gap": np.nan,
            })
            continue
        mean_prob = float(y_prob[mask].mean())
        fraction_positive = float(y_true[mask].mean())
        rows.append({
            "bin": index,
            "bin_lower": edges[index],
            "bin_upper": edges[index + 1],
            "n_videos": int(mask.sum()),
            "mean_predicted_prob": mean_prob,
            "fraction_positive": fraction_positive,
            "absolute_gap": abs(mean_prob - fraction_positive),
        })
    return pd.DataFrame(rows)


def calibration_metrics(df_video):
    y_true = df_video["label"].to_numpy(np.int64)
    y_prob = np.clip(df_video["prob"].to_numpy(np.float64), 0.0, 1.0)
    curve = build_calibration_curve(df_video)
    counts = curve["n_videos"].to_numpy(np.float64)
    gaps = curve["absolute_gap"].fillna(0.0).to_numpy(np.float64)
    result = {
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(np.sum(counts * gaps) / max(float(counts.sum()), 1.0)),
        "MCE": float(np.max(gaps)) if len(gaps) else np.nan,
        "Calibration_Intercept": np.nan,
        "Calibration_Slope": np.nan,
        "n_videos": int(len(df_video)),
    }
    if len(np.unique(y_true)) < 2:
        return result
    clipped = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits, y_true)
        result["Calibration_Intercept"] = float(model.intercept_[0])
        result["Calibration_Slope"] = float(model.coef_[0, 0])
    except Exception:
        pass
    return result


def bootstrap_cluster_ci(df_video, n_bootstrap=BOOTSTRAP_REPS, seed=SEED):
    if df_video["video_key"].duplicated().any():
        raise ValueError("Bootstrap 输入必须是一视频一行")
    names = [
        "ACC", "AUC", "PR-AUC", "F1", "Balanced_ACC", "Sensitivity",
        "Specificity", "Brier", "ECE", "MCE",
        "Calibration_Intercept", "Calibration_Slope",
    ]
    rng = np.random.default_rng(seed)
    values = {name: [] for name in names}
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(df_video), size=len(df_video))
        sample = df_video.iloc[index].copy().reset_index(drop=True)
        sample["video_key"] = np.arange(len(sample))
        metric = calculate_metrics(
            sample["label"].to_numpy(),
            sample["prediction"].to_numpy(),
            sample["prob"].to_numpy(),
        )
        metric.update(calibration_metrics(sample))
        for name in names:
            values[name].append(metric.get(name, np.nan))
    point = calculate_metrics(
        df_video["label"].to_numpy(),
        df_video["prediction"].to_numpy(),
        df_video["prob"].to_numpy(),
    )
    point.update(calibration_metrics(df_video))
    rows = []
    for name in names:
        array = np.asarray(values[name], dtype=np.float64)
        array = array[np.isfinite(array)]
        rows.append({
            "metric": name,
            "estimate": float(point[name]),
            "ci_lower": float(np.percentile(array, 2.5)) if len(array) else np.nan,
            "ci_upper": float(np.percentile(array, 97.5)) if len(array) else np.nan,
            "n_valid_bootstrap": int(len(array)),
            "n_bootstrap": int(n_bootstrap),
            "cluster_unit": "video_key (one video = one patient)",
        })
    return pd.DataFrame(rows)


class SegmentAttentionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        d_model = 64
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=128,
            dropout=0.15,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.score = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, x, mask):
        hidden = self.proj(x)
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        score = self.score(hidden).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(score, dim=1).unsqueeze(-1)
        weighted = torch.sum(hidden * weights, dim=1)
        masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), -1e4)
        max_pool = masked_hidden.max(dim=1).values
        return self.head(torch.cat([weighted, max_pool], dim=-1))


def loss_weights(y, class_weight):
    if class_weight is None:
        weight = np.ones(2, dtype=np.float32)
    elif class_weight == "balanced":
        counts = np.bincount(y, minlength=2).astype(float)
        weight = len(y) / (2.0 * np.maximum(counts, 1.0))
    else:
        weight = np.array([
            class_weight.get(0, 1.0),
            class_weight.get(1, 1.0),
        ], dtype=np.float32)
    return torch.tensor(weight, dtype=torch.float32, device=DEVICE)


def make_loader(X, M, y=None, shuffle=False, seed=42):
    tensors = [
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(M.astype(bool)),
    ]
    if y is not None:
        tensors.append(torch.from_numpy(y.astype(np.int64)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def predict(model, X, M):
    model.eval()
    result = []
    for xb, mb in make_loader(X, M):
        probability = torch.softmax(
            model(xb.to(DEVICE), mb.to(DEVICE)), dim=1
        )[:, 1]
        result.append(probability.cpu().numpy())
    return np.concatenate(result)


@torch.no_grad()
def evaluate_validation(model, data, criterion):
    model.eval()
    loader = make_loader(
        data["X_eval"], data["M_eval"], data["y_eval"], False, seed=42
    )
    losses = []
    probabilities = []
    for xb, mb, yb in loader:
        logits = model(xb.to(DEVICE), mb.to(DEVICE))
        losses.append(float(criterion(logits, yb.to(DEVICE)).cpu()))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    val_prob = np.concatenate(probabilities)
    y_true = data["y_eval"]
    val_auc = roc_auc_score(y_true, val_prob) if len(np.unique(y_true)) > 1 else np.nan
    val_bal_acc = balanced_accuracy_score(y_true, (val_prob >= 0.5).astype(np.int64))
    return float(np.mean(losses)), val_prob, float(val_auc), float(val_bal_acc)


def train_with_val(
    data,
    class_weight,
    seed,
    log_path=None,
    log_metadata=None,
):
    seed_everything(seed)
    model = SegmentAttentionNet(data["X_fit"].shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights(data["y_fit"], class_weight)
    )
    loader = make_loader(
        data["X_fit"], data["M_fit"], data["y_fit"], True, seed
    )
    best_state = None
    best_prauc = -float("inf")
    best_epoch = 1
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * int(yb.shape[0])
            train_count += int(yb.shape[0])
        scheduler.step()
        val_loss, val_prob, val_auc, val_bal_acc = evaluate_validation(
            model, data, criterion
        )
        prauc = average_precision_score(data["y_eval"], val_prob)
        checkpoint_saved = int(prauc > best_prauc)
        row = {
            "model": "attention",
            "ablation": (log_metadata or {}).get("ablation", data["ablation_mode"]),
            "outer_fold": (log_metadata or {}).get("outer_fold", np.nan),
            "inner_fold": (log_metadata or {}).get("inner_fold", np.nan),
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_ap": float(prauc),
            "val_balanced_accuracy": val_bal_acc,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "checkpoint_saved": checkpoint_saved,
        }
        history.append(row)
        if log_path is not None:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            pd.DataFrame(history, columns=EPOCH_METRIC_COLUMNS).to_csv(
                log_path,
                index=False,
                encoding="utf-8-sig",
            )
        if prauc > best_prauc:
            best_prauc = prauc
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, predict(model, data["X_eval"], data["M_eval"])


def train_fixed(data, class_weight, seed, epochs):
    seed_everything(seed)
    model = SegmentAttentionNet(data["X_fit"].shape[-1]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs))
    )
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights(data["y_fit"], class_weight)
    )
    loader = make_loader(
        data["X_fit"], data["M_fit"], data["y_fit"], True, seed
    )
    for _ in range(max(1, int(epochs))):
        model.train()
        for xb, mb, yb in loader:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        scheduler.step()
    return model


def inner_oof(df_outer_train, train_meta, fold_idx, class_weight, ablation_mode):
    labels = train_meta["label"].to_numpy(np.int64)
    keys = train_meta["video_key"].to_numpy()
    key_to_index = {key: index for index, key in enumerate(keys)}
    oof = np.full(len(keys), np.nan, dtype=np.float64)
    best_epochs = []
    n_splits = min(INNER_FOLD, int(train_meta["label"].value_counts().min()))
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED + fold_idx * 100,
    )
    for inner_idx, (fit_idx, val_idx) in enumerate(
        cv.split(train_meta, labels, groups=train_meta["video_key"]), 1
    ):
        fit_keys = set(train_meta.iloc[fit_idx]["video_key"])
        val_keys = set(train_meta.iloc[val_idx]["video_key"])
        if fit_keys & val_keys:
            raise AssertionError("inner train/validation video overlap")
        data = prepare_sequences(
            df_outer_train[df_outer_train["video_key"].isin(fit_keys)],
            df_outer_train[df_outer_train["video_key"].isin(val_keys)],
            ablation_mode,
        )
        log_path = os.path.join(
            SAVE_ROOT,
            ablation_mode,
            f"outer_fold_{fold_idx}",
            f"inner_fold_{inner_idx}",
            "epoch_metrics.csv",
        )
        _, best_epoch, val_prob = train_with_val(
            data,
            class_weight,
            SEED + fold_idx * 1000 + inner_idx,
            log_path=log_path,
            log_metadata={
                "ablation": ablation_mode,
                "outer_fold": fold_idx,
                "inner_fold": inner_idx,
            },
        )
        best_epochs.append(best_epoch)
        for key, probability in zip(data["keys_eval"], val_prob):
            oof[key_to_index[key]] = probability
    if np.any(~np.isfinite(oof)):
        raise RuntimeError("inner OOF 未覆盖所有视频")
    return labels, oof, max(1, int(np.median(best_epochs)))


def save_attention_bundle(path, state_dicts, data, fold, threshold, epochs, class_weight):
    scaler = data["scaler"]
    bundle = {
        "format_version": 1,
        "model_type": "SegmentAttentionNet",
        "ablation_mode": data["ablation_mode"],
        "state_dicts": state_dicts,
        "input_dim": int(data["X_fit"].shape[-1]),
        "feature_cols": list(data["feature_cols"]),
        "fill_values": data["fill_values"],
        "max_segments": int(data["max_segments"]),
        "scaler_mean": scaler.mean_.astype(np.float32).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float32).tolist(),
        "threshold": float(threshold),
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "threshold_grid": {
            "min": float(THRESHOLD_GRID.min()),
            "max": float(THRESHOLD_GRID.max()),
            "step": 0.01,
        },
        "seed": int(SEED),
        "final_seeds": [
            int(SEED + fold * 10000 + seed_index)
            for seed_index in range(FINAL_SEEDS)
        ],
        "epochs": int(epochs),
        "class_weight": class_weight,
        "fold": int(fold),
        "test_video_keys": [str(key) for key in data["keys_eval"]],
    }
    joblib.dump(bundle, path)


def load_attention_bundle(path):
    bundle = joblib.load(path)
    models = []
    for state in bundle["state_dicts"]:
        model = SegmentAttentionNet(bundle["input_dim"]).to(DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    return models, bundle


def evaluate_saved_attention_bundle(bundle_path, data_path=DATA_PATH):
    """用保存的指定 outer fold bundle 对同一批视频重新测试。"""
    models, bundle = load_attention_bundle(bundle_path)
    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.strip()
    df["filename"] = df["filename"].astype(str)
    df["video_key"] = df["filename"].apply(get_original_video_key)
    df["label"] = df["label"].astype(int)
    test_keys = set(str(key) for key in bundle["test_video_keys"])
    df_test = df[df["video_key"].astype(str).isin(test_keys)].copy()
    if df_test.empty:
        raise ValueError(f"没有找到 bundle 对应的测试视频: {bundle_path}")
    x_eval, mask, y_eval, keys_eval = prepare_saved_eval_sequences(df_test, bundle)
    probabilities = np.mean(
        np.stack([predict(model, x_eval, mask) for model in models]), axis=0
    )
    threshold = float(bundle["threshold"])
    predictions = (probabilities >= threshold).astype(np.int64)
    result = pd.DataFrame({
        "video_key": keys_eval,
        "label": y_eval,
        "prob": probabilities,
        "pred": predictions,
        "fold": int(bundle["fold"]),
        "fold_threshold": threshold,
        "ablation": bundle["ablation_mode"],
    })
    return calculate_metrics(y_eval, predictions, probabilities), result


def save_curve_plot(df_video, output_root, ablation_mode):
    plot_root = os.path.join(output_root, "paper_auc_pr_curves")
    os.makedirs(plot_root, exist_ok=True)
    y_true = df_video["label"].to_numpy(np.int64)
    y_prob = df_video["prob"].to_numpy(np.float64)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, roc_threshold = roc_curve(y_true, y_prob)
    precision, recall, pr_threshold = precision_recall_curve(y_true, y_prob)
    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": roc_threshold,
    }).to_csv(os.path.join(plot_root, f"{ablation_mode}_roc_curve.csv"), index=False)
    pd.DataFrame({
        "recall": recall,
        "precision": precision,
        "threshold": np.append(pr_threshold, np.nan),
    }).to_csv(os.path.join(plot_root, f"{ablation_mode}_pr_curve.csv"), index=False)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 不可用，仅保存曲线 CSV")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].plot(
        fpr, tpr,
        label=f"{ablation_mode} (AUC={roc_auc_score(y_true, y_prob):.3f})",
    )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Video-level ROC curve")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    axes[1].plot(
        recall, precision,
        label=f"{ablation_mode} (PR-AUC={average_precision_score(y_true, y_prob):.3f})",
    )
    axes[1].axhline(float(y_true.mean()), linestyle="--", color="gray")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Video-level precision-recall curve")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left")
    fig.savefig(os.path.join(plot_root, f"{ablation_mode}_roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plot_root, f"{ablation_mode}_roc_pr_curves.pdf"), bbox_inches="tight")
    plt.close(fig)


def save_test_statistics(df_video, output_root, ablation_mode):
    metrics = calculate_metrics(
        df_video["label"].to_numpy(),
        df_video["prediction"].to_numpy(),
        df_video["prob"].to_numpy(),
    )
    pd.DataFrame([metrics]).to_csv(
        os.path.join(output_root, "pooled_outer_test_metrics.csv"), index=False
    )
    bootstrap = bootstrap_cluster_ci(df_video)
    bootstrap.to_csv(
        os.path.join(output_root, "pooled_outer_test_bootstrap_ci.csv"), index=False
    )
    calibration = calibration_metrics(df_video)
    pd.DataFrame([calibration]).to_csv(
        os.path.join(output_root, "pooled_outer_test_calibration.csv"), index=False
    )
    build_calibration_curve(df_video).to_csv(
        os.path.join(output_root, "pooled_outer_test_calibration_curve.csv"), index=False
    )
    df_video.to_csv(
        os.path.join(output_root, "all_outer_test_vid_pred.csv"), index=False
    )
    save_curve_plot(df_video, output_root, ablation_mode)
    return metrics, bootstrap, calibration


def save_paper_table(metrics, bootstrap, calibration, output_root):
    ci = bootstrap.set_index("metric")
    names = [
        "ACC", "AUC", "PR-AUC", "F1", "Balanced_ACC", "Sensitivity",
        "Specificity", "Brier", "ECE", "MCE",
        "Calibration_Intercept", "Calibration_Slope",
    ]
    rows = []
    for name in names:
        estimate = calibration.get(name, metrics.get(name, np.nan))
        row = ci.loc[name]
        rows.append({
            "metric": name,
            "estimate": float(estimate),
            "95CI": f"{row['estimate']:.4f} ({row['ci_lower']:.4f}, {row['ci_upper']:.4f})",
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_root, "paper_table_pooled_outer_test_ci.csv"), index=False)
    try:
        table.to_latex(os.path.join(output_root, "paper_table_pooled_outer_test_ci.tex"), index=False, escape=True)
    except Exception:
        pass


def save_combined_curves(prediction_frames, output_root):
    plot_root = os.path.join(output_root, "paper_auc_pr_curves")
    os.makedirs(plot_root, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    curve_rows = []
    for frame in prediction_frames:
        mode = str(frame["ablation"].iloc[0])
        y_true = frame["label"].to_numpy(np.int64)
        y_prob = frame["prob"].to_numpy(np.float64)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, roc_threshold = roc_curve(y_true, y_prob)
        precision, recall, pr_threshold = precision_recall_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        curve_rows.extend(
            [{"ablation": mode, "curve": "ROC", **row} for row in pd.DataFrame({
                "fpr": fpr, "tpr": tpr, "threshold": roc_threshold
            }).to_dict("records")]
        )
        curve_rows.extend(
            [{"ablation": mode, "curve": "PR", **row} for row in pd.DataFrame({
                "recall": recall, "precision": precision,
                "threshold": np.append(pr_threshold, np.nan),
            }).to_dict("records")]
        )
        axes[0].plot(fpr, tpr, label=f"{mode} (AUC={roc_auc:.3f})")
        axes[1].plot(recall, precision, label=f"{mode} (PR-AUC={pr_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Video-level ROC curve")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Video-level precision-recall curve")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower left", fontsize=9)
    fig.savefig(os.path.join(plot_root, "ablation_roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(plot_root, "ablation_roc_pr_curves.pdf"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(curve_rows).to_csv(
        os.path.join(plot_root, "ablation_roc_pr_curves.csv"), index=False
    )


def load_input_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    required = set(FEATURE_COLS + ["filename", "label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少字段: {sorted(missing)}")
    df["filename"] = df["filename"].astype(str)
    df["video_key"] = df["filename"].apply(get_original_video_key)
    df["label"] = df["label"].astype(int)
    if df.groupby("video_key")["label"].nunique().max() > 1:
        raise ValueError("同一视频对应多个标签")
    return df


def build_video_table(df):
    return (
        df[["video_key", "label"]]
        .drop_duplicates()
        .sort_values("video_key")
        .reset_index(drop=True)
    )




def save_split_manifest(video_meta, output_path):
    """保存 outer/inner 的固定视频级划分清单。"""
    rows = []
    label_map = dict(
        zip(video_meta["video_key"].astype(str), video_meta["label"].astype(int))
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLD,
        shuffle=True,
        random_state=SEED,
    )

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(
            video_meta,
            video_meta["label"],
            groups=video_meta["video_key"],
        ),
        start=1,
    ):
        outer_train = video_meta.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = video_meta.iloc[outer_test_idx].reset_index(drop=True)

        for key in outer_test["video_key"].astype(str):
            rows.append(
                {
                    "animal_id": key,
                    "video_key": key,
                    "label": label_map[key],
                    "outer_fold": outer_fold,
                    "role": "outer_test",
                    "inner_fold": "",
                    "split_seed": SEED,
                    "split_version": "sgkf_v1",
                }
            )

        n_inner = min(
            INNER_FOLD,
            int(outer_train["label"].value_counts().min()),
        )
        inner_cv = StratifiedGroupKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=SEED + outer_fold * 100,
        )

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(
                outer_train,
                outer_train["label"],
                groups=outer_train["video_key"],
            ),
            start=1,
        ):
            for role, indices in (
                ("inner_train", inner_train_idx),
                ("inner_val", inner_val_idx),
            ):
                for key in outer_train.iloc[indices]["video_key"].astype(str):
                    rows.append(
                        {
                            "animal_id": key,
                            "video_key": key,
                            "label": label_map[key],
                            "outer_fold": outer_fold,
                            "role": role,
                            "inner_fold": inner_fold,
                            "split_seed": SEED + outer_fold * 100,
                            "split_version": "sgkf_v1",
                        }
                    )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def save_run_manifest(summary, output_root, run_start_time):
    """保存每个 outer fold 的运行、阈值和模型文件记录。"""
    end_time = datetime.now().isoformat(timespec="seconds")
    required = pd.DataFrame({
        "run_id": [
            f"attention_{str(ablation)}_outer_fold_{int(fold)}"
            for ablation, fold in zip(summary["ablation"], summary["outer_fold"])
        ],
        "model": "attention",
        "ablation": summary["ablation"].astype(str),
        "outer_fold": summary["outer_fold"].astype(int),
        "seed": SEED,
        "config_file": "",
        "config_sha256": "",
        "code_commit": "",
        "start_time": run_start_time,
        "end_time": end_time,
        "selected_epoch": summary["epochs"],
        "selected_threshold": summary["threshold"],
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "checkpoint_path": [
            os.path.abspath(os.path.join(output_root, path))
            for path in summary["model_bundle"]
        ],
    })
    extras = summary.drop(columns=["threshold"], errors="ignore").reset_index(drop=True)
    manifest = pd.concat([required, extras], axis=1)
    ordered_columns = RUN_MANIFEST_COLUMNS + [
        column for column in manifest.columns if column not in RUN_MANIFEST_COLUMNS
    ]
    logs_dir = os.path.join(output_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    manifest[ordered_columns].to_csv(
        os.path.join(logs_dir, "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    global_path = os.path.join(SAVE_ROOT, "logs", "run_manifest.csv")
    os.makedirs(os.path.dirname(global_path), exist_ok=True)
    header = not os.path.exists(global_path)
    manifest[ordered_columns].to_csv(
        global_path,
        mode="a",
        header=header,
        index=False,
        encoding="utf-8-sig",
    )


def consolidate_epoch_metrics():
    rows = []
    for root, _, filenames in os.walk(SAVE_ROOT):
        if os.path.abspath(root) == os.path.abspath(os.path.join(SAVE_ROOT, "logs")):
            continue
        if "epoch_metrics.csv" not in filenames:
            continue
        path = os.path.join(root, "epoch_metrics.csv")
        try:
            rows.append(pd.read_csv(path))
        except (OSError, pd.errors.EmptyDataError):
            continue
    logs_dir = os.path.join(SAVE_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for column in EPOCH_METRIC_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    combined[EPOCH_METRIC_COLUMNS].to_csv(
        os.path.join(logs_dir, "epoch_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def run_one_ablation(df, videos, outer_splits, ablation_mode):
    output_root = os.path.join(SAVE_ROOT, ablation_mode)
    os.makedirs(output_root, exist_ok=True)
    summary_rows = []
    all_test = []
    for fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        print(f"\n================ {ablation_mode} | Fold {fold}/{OUTER_FOLD} ================")
        fold_root = os.path.join(output_root, f"outer_fold_{fold}")
        os.makedirs(fold_root, exist_ok=True)
        train_meta = videos.iloc[train_idx].reset_index(drop=True)
        test_meta = videos.iloc[test_idx].reset_index(drop=True)
        train_keys = set(train_meta["video_key"])
        test_keys = set(test_meta["video_key"])
        df_train = df[df["video_key"].isin(train_keys)].copy()
        df_test = df[df["video_key"].isin(test_keys)].copy()

        candidates = []
        for class_weight in CLASS_WEIGHT_CANDIDATES:
            labels, probabilities, epochs = inner_oof(
                df_train,
                train_meta,
                fold,
                class_weight,
                ablation_mode,
            )
            threshold = search_best_threshold(labels, probabilities)
            metric = calculate_metrics(
                labels,
                (probabilities >= threshold).astype(np.int64),
                probabilities,
            )
            candidates.append(
                (
                    metric["Balanced_ACC"],
                    metric["F1"],
                    metric["ACC"],
                    metric["AUC"],
                    class_weight,
                    threshold,
                    epochs,
                    labels,
                    probabilities,
                )
            )

        _, _, _, _, class_weight, threshold, epochs, inner_labels, inner_prob = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3] if np.isfinite(item[3]) else -1.0,
            ),
        )
        pd.DataFrame(
            [
                {
                    "class_weight": str(item[4]),
                    "threshold": item[5],
                    "oof_balanced_accuracy": item[0],
                    "oof_f1": item[1],
                    "oof_accuracy": item[2],
                    "oof_auc": item[3],
                    "epochs": item[6],
                    "threshold_objective": THRESHOLD_OBJECTIVE,
                }
                for item in candidates
            ]
        ).to_csv(
            os.path.join(fold_root, "threshold_selection_candidates.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        inner_oof_frame = pd.DataFrame({
            "model": "attention",
            "ablation": ablation_mode,
            "animal_id": train_meta["video_key"].astype(str),
            "video_key": train_meta["video_key"],
            "label": inner_labels,
            "probability": inner_prob,
            "prediction": (inner_prob >= threshold).astype(np.int64),
            "prob": inner_prob,
            "pred": (inner_prob >= threshold).astype(np.int64),
            "outer_fold": fold,
            "fold_threshold": threshold,
            "threshold_objective": THRESHOLD_OBJECTIVE,
        })
        inner_oof_frame.to_csv(
            os.path.join(fold_root, "inner_oof_video_pred.csv"),
            index=False,
        )

        data = prepare_sequences(df_train, df_test, ablation_mode)
        probabilities = []
        state_dicts = []
        for seed_index in range(FINAL_SEEDS):
            model = train_fixed(
                data,
                class_weight,
                SEED + fold * 10000 + seed_index,
                epochs,
            )
            probabilities.append(predict(model, data["X_eval"], data["M_eval"]))
            state_dicts.append({
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            })
            torch.save(
                {
                    "state": state_dicts[-1],
                    "ablation_mode": ablation_mode,
                    "threshold": threshold,
                    "epochs": epochs,
                    "input_dim": data["X_fit"].shape[-1],
                    "feature_cols": list(data["feature_cols"]),
                    "fill_values": data["fill_values"],
                    "max_segments": data["max_segments"],
                    "scaler_mean": data["scaler"].mean_.astype(np.float32).tolist(),
                    "scaler_scale": data["scaler"].scale_.astype(np.float32).tolist(),
                    "test_video_keys": [str(key) for key in data["keys_eval"]],
                },
                os.path.join(fold_root, f"fold{fold}_seed{seed_index}.pt"),
            )
        probability = np.mean(np.stack(probabilities), axis=0)
        prediction = (probability >= threshold).astype(np.int64)
        test_frame = pd.DataFrame({
            "model": "attention",
            "ablation": ablation_mode,
            "animal_id": np.asarray(data["keys_eval"]).astype(str),
            "video_key": data["keys_eval"],
            "label": data["y_eval"],
            "probability": probability,
            "prediction": prediction,
            "prob": probability,
            "pred": prediction,
            "fold": fold,
            "outer_fold": fold,
            "fold_threshold": threshold,
            "threshold_objective": THRESHOLD_OBJECTIVE,
        })
        bundle_path = os.path.join(fold_root, "attention_model_bundle.pkl")
        test_frame["checkpoint_id"] = os.path.relpath(bundle_path, SAVE_ROOT)
        metric = calculate_metrics(
            test_frame["label"].to_numpy(),
            test_frame["pred"].to_numpy(),
            test_frame["prob"].to_numpy(),
        )
        test_frame.to_csv(os.path.join(fold_root, "outer_test_vid_pred.csv"), index=False)
        save_attention_bundle(
            bundle_path,
            state_dicts,
            data,
            fold,
            threshold,
            epochs,
            class_weight,
        )
        all_test.append(test_frame)
        summary_rows.append({
            "ablation": ablation_mode,
            "outer_fold": fold,
            "class_weight": str(class_weight),
            "threshold": threshold,
            "epochs": epochs,
            "test_acc": metric["ACC"],
            "test_auc": metric["AUC"],
            "test_prauc": metric["PR-AUC"],
            "test_f1": metric["F1"],
            "test_bal_acc": metric["Balanced_ACC"],
            "test_sensitivity": metric["Sensitivity"],
            "test_specificity": metric["Specificity"],
            "n_test_videos": len(test_keys),
            "model_bundle": os.path.relpath(bundle_path, output_root),
        })
        print(
            f"Test ACC={metric['ACC']:.4f}, AUC={metric['AUC']:.4f}, "
            f"PR-AUC={metric['PR-AUC']:.4f}, F1={metric['F1']:.4f}"
        )

    summary = pd.DataFrame(summary_rows)
    test = pd.concat(all_test, ignore_index=True)
    summary.to_csv(os.path.join(output_root, "outer_5fold_summary.csv"), index=False)
    test.to_csv(
        os.path.join(output_root, "all_outer_test_vid_pred.csv"),
        index=False,
    )
    test[PREDICTION_COLUMNS].to_csv(
        os.path.join(output_root, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    save_run_manifest(summary, output_root, RUN_START_TIME)
    mean_std = pd.DataFrame([
        {
            "metric": column,
            "mean": summary[column].mean(),
            "std": summary[column].std(ddof=1),
            "mean_pm_std": f"{summary[column].mean():.4f} ± {summary[column].std(ddof=1):.4f}",
        }
        for column in [
            "test_acc", "test_auc", "test_prauc", "test_f1",
            "test_bal_acc", "test_sensitivity", "test_specificity",
        ]
    ])
    mean_std.to_csv(os.path.join(output_root, "paper_outer_test_mean_std_long.csv"), index=False)
    metrics, bootstrap, calibration = save_test_statistics(test, output_root, ablation_mode)
    save_paper_table(metrics, bootstrap, calibration, output_root)
    return summary, metrics, bootstrap, calibration, test


def main():
    global RUN_START_TIME
    RUN_START_TIME = datetime.now().isoformat(timespec="seconds")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "logs"), exist_ok=True)
    pd.DataFrame(columns=RUN_MANIFEST_COLUMNS).to_csv(
        os.path.join(SAVE_ROOT, "logs", "run_manifest.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    seed_everything(SEED)
    df = load_input_data()
    videos = build_video_table(df)
    if int(videos["label"].value_counts().min()) < OUTER_FOLD:
        raise ValueError("最少类别视频数不足以进行五折划分")
    save_split_manifest(
        videos,
        os.path.join(SAVE_ROOT, "split_manifest.csv"),
    )
    print(f"device={DEVICE}, rows={len(df)}, videos={len(videos)}")

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_FOLD,
        shuffle=True,
        random_state=SEED,
    )
    outer_splits = list(outer_cv.split(
        videos,
        videos["label"],
        groups=videos["video_key"],
    ))

    summaries = []
    pooled_metrics = []
    pooled_bootstrap = []
    pooled_calibration = []
    prediction_frames = []
    for ablation_mode in ABLATION_MODES:
        summary, metrics, bootstrap, calibration, predictions = run_one_ablation(
            df,
            videos,
            outer_splits,
            ablation_mode,
        )
        summaries.append(summary)
        pooled_metrics.append({"ablation": ablation_mode, **metrics})
        bootstrap = bootstrap.copy()
        bootstrap.insert(0, "ablation", ablation_mode)
        pooled_bootstrap.append(bootstrap)
        pooled_calibration.append({"ablation": ablation_mode, **calibration})
        prediction_frames.append(predictions)

    pd.concat(summaries, ignore_index=True).to_csv(
        os.path.join(SAVE_ROOT, "ablation_outer_5fold_summary.csv"), index=False
    )
    pd.DataFrame(pooled_metrics).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_metrics.csv"), index=False
    )
    pd.concat(pooled_bootstrap, ignore_index=True).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_bootstrap_ci.csv"), index=False
    )
    pd.DataFrame(pooled_calibration).to_csv(
        os.path.join(SAVE_ROOT, "ablation_pooled_outer_test_calibration.csv"), index=False
    )
    save_combined_curves(prediction_frames, SAVE_ROOT)
    pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS].to_csv(
        os.path.join(SAVE_ROOT, "standardized_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    consolidate_epoch_metrics()
    print("\n================ Feature Ablation Summary ================")
    print(pd.DataFrame(pooled_metrics).to_string(index=False))


if __name__ == "__main__":
    main()
