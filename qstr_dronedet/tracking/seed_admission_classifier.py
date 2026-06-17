from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


SEED_ADMISSION_FEATURES = [
    "detector_score",
    "selector_score",
    "motion_consistency",
    "memory_consistency",
    "jump_px",
    "reacquire_distance_px",
    "bbox_area",
    "appearance_similarity",
    "crop_drone_score",
    "tracklet_filter_applied",
    "tracklet_is_drone",
    "tracklet_classifier_prob",
    "sequence_gate_confirmed",
    "global_small_area_candidate",
    "small_global_since_last_frame",
    "small_global_repeat_distance_px",
    "small_global_repeat_cooldown_active",
]


@dataclass
class SeedAdmissionClassifierResult:
    weights_path: Path
    metrics_json: Path
    threshold_sweep_csv: Path
    summary_json: Path
    summary: dict[str, Any]


class SeedAdmissionMLP(torch.nn.Module):
    def __init__(self, in_dim: int = len(SEED_ADMISSION_FEATURES), hidden: int = 32) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_seed_admission_dataset(
    csv_path: str | Path,
    *,
    features: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    feature_names = features or SEED_ADMISSION_FEATURES
    rows: list[dict[str, Any]] = []
    values: list[list[float]] = []
    labels: list[int] = []
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _parse_label(row)
            if label is None:
                continue
            labels.append(label)
            rows.append(row)
            values.append([_feature_value(row.get(feature, "")) for feature in feature_names])
    if not values:
        raise ValueError("Seed admission dataset has no labeled rows")
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
        rows,
    )


def train_seed_admission_classifier(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 25,
    lr: float = 1e-3,
    hidden: int = 32,
    thresholds: list[float] | None = None,
    smoke: bool = False,
) -> SeedAdmissionClassifierResult:
    x, y, meta = load_seed_admission_dataset(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / "seed_admission_classifier.pt"
    metrics_json = out / "metrics.json"
    threshold_sweep_csv = out / "threshold_sweep.csv"
    summary_json = out / "summary.json"

    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_norm = (x - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeedAdmissionMLP(in_dim=x.shape[1], hidden=hidden).to(device)
    loader = _training_loader(x_norm, y, meta)
    counts = torch.bincount(y, minlength=2).float()
    loss_weights = counts.sum() / counts.clamp_min(1.0) / 2.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history: list[dict[str, Any]] = []
    for epoch in range(int(epochs)):
        total_loss = 0.0
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(by.numel())
            total += int(by.numel())
        history.append({"epoch": epoch + 1, "loss": total_loss / max(1, total)})

    probs = _predict_probs(model.cpu(), x_norm)
    sweep_rows, best = _threshold_sweep(probs, y, thresholds)
    _write_threshold_sweep(threshold_sweep_csv, sweep_rows)
    metrics = {
        "csv_path": str(csv_path),
        "weights": str(weights_path),
        "threshold_sweep_csv": str(threshold_sweep_csv),
        "num_rows": int(len(y)),
        "num_training_rows": int(len(y)),
        "positive_rows": int((y == 1).sum().item()),
        "negative_rows": int((y == 0).sum().item()),
        "smoke": bool(smoke),
        "best": best,
    }
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": SEED_ADMISSION_FEATURES,
            "mean": mean,
            "std": std,
            "hidden": int(hidden),
            "history": history,
            "num_training_rows": int(len(y)),
            "positive_rows": int((y == 1).sum().item()),
            "negative_rows": int((y == 0).sum().item()),
            "smoke": bool(smoke),
        },
        weights_path,
    )
    summary = {
        **metrics,
        "summary_json": str(summary_json),
        "metrics_json": str(metrics_json),
        "threshold_sweep_csv": str(threshold_sweep_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return SeedAdmissionClassifierResult(weights_path, metrics_json, threshold_sweep_csv, summary_json, summary)


def _parse_label(row: dict[str, Any]) -> int | None:
    sample_weight = _feature_value(row.get("sample_weight", "1"))
    if sample_weight <= 0:
        return None
    raw = str(row.get("seed_label", "")).strip()
    if raw == "":
        return None
    return int(float(raw))


def _feature_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip()
    if text.lower() in {"true", "yes"}:
        return 1.0
    if text.lower() in {"false", "no", ""}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _training_loader(x: torch.Tensor, y: torch.Tensor, meta: list[dict[str, Any]]) -> DataLoader:
    group_counts: dict[str, int] = {}
    groups: list[str] = []
    for index, row in enumerate(meta):
        group = f"{row.get('dataset_source', '')}|{int(y[index].item())}"
        groups.append(group)
        group_counts[group] = group_counts.get(group, 0) + 1
    sample_weights = torch.tensor([1.0 / max(1, group_counts[group]) for group in groups], dtype=torch.float32)
    dataset = TensorDataset(x, y)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=min(32, len(dataset)), sampler=sampler)


def _predict_probs(model: SeedAdmissionMLP, x_norm: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(model(x_norm), dim=1)[:, 1]


def _threshold_sweep(
    probs: torch.Tensor,
    y: torch.Tensor,
    thresholds: list[float] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_values = thresholds or [round(i / 20.0, 3) for i in range(1, 20)]
    y_bool = y.bool()
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for threshold in threshold_values:
        pred = probs >= float(threshold)
        tp = int((pred & y_bool).sum().item())
        fp = int((pred & ~y_bool).sum().item())
        fn = int((~pred & y_bool).sum().item())
        tn = int((~pred & ~y_bool).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
        row = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": (tp + tn) / max(1, len(y)),
        }
        rows.append(row)
        if best is None or (row["f1"], row["recall"], row["precision"], -row["fp"]) > (
            best["f1"],
            best["recall"],
            best["precision"],
            -best["fp"],
        ):
            best = row
    return rows, best or {}


def _write_threshold_sweep(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)
