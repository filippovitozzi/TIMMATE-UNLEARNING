from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from utils.model import DynamicMLP


ID_COL = "user_id"
TARGET_PREFIX = "target__"


def load_data(data_dir: Path | str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_paths = sorted(data_dir.glob("*c000.csv"))
    if not train_paths:
        raise FileNotFoundError(f"No partitioned training CSV files found in {data_dir}")

    train = pd.concat((pd.read_csv(path, sep=";") for path in train_paths), ignore_index=True)
    forget = pd.read_csv(data_dir / "forget_data.csv")
    return train, forget


def get_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target_cols = [col for col in df.columns if col.lower().startswith(TARGET_PREFIX)]
    if not target_cols:
        raise ValueError(f"No target columns found with prefix {TARGET_PREFIX!r}")
    feature_cols = [col for col in df.columns if col not in target_cols and col != ID_COL]
    return feature_cols, target_cols


def frame_to_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[target_cols]
    return (
        np.ascontiguousarray(x.replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)),
        np.ascontiguousarray(y.replace([np.inf, -np.inf], 0).fillna(0).to_numpy(np.float32)),
    )


def compute_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    pos_counts = y.sum(axis=0)
    neg_counts = len(y) - pos_counts
    weights = np.clip(neg_counts / (pos_counts + 1e-6), 0.1, 100.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_retain_frame(train: pd.DataFrame, forget: pd.DataFrame) -> pd.DataFrame:
    forget_ids = set(forget[ID_COL].astype(str))
    retain = train.loc[~train[ID_COL].astype(str).isin(forget_ids)].copy()
    if len(retain) + len(forget) != len(train):
        raise ValueError("Unexpected retain/forget split size. Check user_id overlap.")
    return retain


def choose_validation_ids(
    retain: pd.DataFrame,
    target_cols: list[str],
    size: int = 12000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = min(size, len(retain))
    labels_per_row = retain[target_cols].sum(axis=1).to_numpy()
    positive_idx = np.flatnonzero(labels_per_row > 0)
    zero_idx = np.flatnonzero(labels_per_row == 0)

    n_pos = round(size * len(positive_idx) / len(retain))
    n_pos = min(len(positive_idx), n_pos)
    n_zero = min(len(zero_idx), size - n_pos)
    picked = np.concatenate(
        [
            rng.choice(positive_idx, size=n_pos, replace=False),
            rng.choice(zero_idx, size=n_zero, replace=False),
        ]
    )
    if len(picked) < size:
        remaining = np.setdiff1d(np.arange(len(retain)), picked, assume_unique=False)
        picked = np.concatenate([picked, rng.choice(remaining, size=size - len(picked), replace=False)])
    rng.shuffle(picked)
    return retain.iloc[picked][ID_COL].to_numpy()


def subset_by_ids(df: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    return df.loc[df[ID_COL].astype(str).isin(set(pd.Series(ids).astype(str)))].copy()


def build_splits(
    train: pd.DataFrame,
    forget: pd.DataFrame,
    target_cols: list[str],
    validation_size: int = 12000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    retain = make_retain_frame(train, forget)
    validation_ids = choose_validation_ids(retain, target_cols, validation_size, seed)
    validation = subset_by_ids(retain, validation_ids)
    return retain, validation, validation_ids


def sample_arrays(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or max_rows >= len(x):
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), size=max_rows, replace=False)
    return x[idx], y[idx]


def load_artifact(artifact_path: Path | str, device: torch.device) -> tuple[dict, DynamicMLP]:
    with Path(artifact_path).open("rb") as handle:
        payload = pickle.load(handle)

    architecture = payload["architecture"]
    model = DynamicMLP(
        input_dim=architecture["input_dim"],
        hidden_layers=architecture["hidden_layers"],
        num_outputs=architecture["num_outputs"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return payload, model


def batch_slices(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start, end in batch_slices(len(x), batch_size):
        xb = torch.from_numpy(x[start:end]).to(device)
        outputs.append(model(xb).cpu().numpy())
    return np.vstack(outputs)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def restore_state(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in state.items()})
    model.eval()


def save_submission(
    payload: dict,
    state_dict: dict[str, torch.Tensor],
    validation_ids: np.ndarray,
    output_dir: Path | str,
    execution_time: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "model_artifact").open("wb") as handle:
        pickle.dump({**payload, "state_dict": state_dict}, handle)

    seconds = 0 if execution_time <= 0 else max(1, math.ceil(execution_time))
    (output_dir / "execution_time.txt").write_text(str(seconds), encoding="utf-8")

    with (output_dir / "validation_ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([ID_COL])
        writer.writerows([user_id] for user_id in validation_ids)


def precision_at_k(y_true: np.ndarray, logits: np.ndarray, k: int = 10) -> float:
    top_k = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def per_sample_bce_loss(y_true: np.ndarray, logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    losses = np.maximum(logits, 0) - logits * y_true + np.log1p(np.exp(-np.abs(logits)))
    return losses.mean(axis=1)


def mia_auc_from_losses(loss_forget: np.ndarray, loss_eval: np.ndarray) -> float:
    labels = np.concatenate([np.ones_like(loss_forget), np.zeros_like(loss_eval)])
    scores = -np.concatenate([loss_forget, loss_eval])
    return float(roc_auc_score(labels, scores))


def metric_bundle(
    y_forget: np.ndarray,
    logits_forget: np.ndarray,
    y_validation: np.ndarray,
    logits_validation: np.ndarray,
) -> dict[str, float]:
    loss_forget = per_sample_bce_loss(y_forget, logits_forget)
    loss_validation = per_sample_bce_loss(y_validation, logits_validation)
    precision = precision_at_k(y_validation, logits_validation, k=10)
    mia_auc = mia_auc_from_losses(loss_forget, loss_validation)
    mia_resistance = max(0.0, 1.0 - 2.0 * abs(mia_auc - 0.5))
    return {
        "precision_at_10": precision,
        "mia_auc": mia_auc,
        "mia_resistance": mia_resistance,
        "forget_loss_mean": float(loss_forget.mean()),
        "validation_loss_mean": float(loss_validation.mean()),
        "score_no_time": 0.45 * precision + 0.45 * mia_resistance,
    }
