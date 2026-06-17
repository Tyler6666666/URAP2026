import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from qstr_dronedet.tracking.seed_admission_classifier import (
    SEED_ADMISSION_FEATURES,
    load_seed_admission_dataset,
    train_seed_admission_classifier,
)


def _write_seed_dataset(path: Path) -> None:
    fields = [
        "frame_id",
        "dataset_source",
        "seed_label",
        "sample_weight",
        *SEED_ADMISSION_FEATURES,
    ]
    rows = [
        {
            "frame_id": 1,
            "dataset_source": "seg_pos",
            "seed_label": 1,
            "sample_weight": 1,
            "detector_score": 0.85,
            "selector_score": 0.80,
            "motion_consistency": 0.72,
            "memory_consistency": 0.68,
            "jump_px": 8,
            "reacquire_distance_px": 12,
            "bbox_area": 220,
            "appearance_similarity": 0.40,
            "crop_drone_score": 0.91,
            "tracklet_filter_applied": 1,
            "tracklet_is_drone": 1,
            "tracklet_classifier_prob": 0.94,
            "sequence_gate_confirmed": 1,
            "global_small_area_candidate": 1,
            "small_global_since_last_frame": 10,
            "small_global_repeat_distance_px": 18,
            "small_global_repeat_cooldown_active": 1,
        },
        {
            "frame_id": 2,
            "dataset_source": "seg_pos",
            "seed_label": 1,
            "sample_weight": 1,
            "detector_score": 0.82,
            "selector_score": 0.77,
            "motion_consistency": 0.69,
            "memory_consistency": 0.61,
            "jump_px": 11,
            "reacquire_distance_px": 14,
            "bbox_area": 240,
            "appearance_similarity": 0.45,
            "crop_drone_score": 0.88,
            "tracklet_filter_applied": 1,
            "tracklet_is_drone": 1,
            "tracklet_classifier_prob": 0.90,
            "sequence_gate_confirmed": 1,
            "global_small_area_candidate": 1,
            "small_global_since_last_frame": 11,
            "small_global_repeat_distance_px": 20,
            "small_global_repeat_cooldown_active": 1,
        },
        {
            "frame_id": 3,
            "dataset_source": "seg_neg",
            "seed_label": 0,
            "sample_weight": 1,
            "detector_score": 0.35,
            "selector_score": 0.41,
            "motion_consistency": 0.15,
            "memory_consistency": 0.12,
            "jump_px": 210,
            "reacquire_distance_px": 260,
            "bbox_area": 1800,
            "appearance_similarity": 0.05,
            "crop_drone_score": 0.12,
            "tracklet_filter_applied": 1,
            "tracklet_is_drone": 0,
            "tracklet_classifier_prob": 0.04,
            "sequence_gate_confirmed": 0,
            "global_small_area_candidate": 1,
            "small_global_since_last_frame": 6,
            "small_global_repeat_distance_px": 420,
            "small_global_repeat_cooldown_active": 1,
        },
        {
            "frame_id": 4,
            "dataset_source": "seg_ignore",
            "seed_label": "",
            "sample_weight": 0,
            "detector_score": 0.99,
            "selector_score": 0.99,
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_load_seed_admission_dataset_ignores_unlabeled_rows(tmp_path):
    csv_path = tmp_path / "seed_admission.csv"
    _write_seed_dataset(csv_path)

    x, y, meta = load_seed_admission_dataset(csv_path)

    assert x.shape == (3, len(SEED_ADMISSION_FEATURES))
    assert y.tolist() == [1, 1, 0]
    assert [row["frame_id"] for row in meta] == ["1", "2", "3"]


def test_train_seed_admission_classifier_writes_smoke_artifacts(tmp_path):
    csv_path = tmp_path / "seed_admission.csv"
    out_dir = tmp_path / "smoke"
    _write_seed_dataset(csv_path)

    result = train_seed_admission_classifier(
        csv_path,
        out_dir,
        epochs=2,
        hidden=8,
        thresholds=[0.25, 0.5],
        smoke=True,
    )

    assert result.weights_path.exists()
    assert result.metrics_json.exists()
    assert result.threshold_sweep_csv.exists()
    assert result.summary["smoke"] is True
    assert result.summary["num_training_rows"] == 3
    assert result.summary["positive_rows"] == 2
    assert result.summary["negative_rows"] == 1
    assert result.summary["best"]["threshold"] in {0.25, 0.5}
    checkpoint = torch.load(result.weights_path, map_location="cpu", weights_only=False)
    assert checkpoint["features"] == SEED_ADMISSION_FEATURES


def test_modal_seed_admission_train_script_local_smoke(tmp_path):
    csv_path = tmp_path / "seed_admission.csv"
    out_dir = tmp_path / "script_smoke"
    _write_seed_dataset(csv_path)

    result = subprocess.run(
        [
            sys.executable,
            "tools/modal_seed_admission_train.py",
            "--dataset",
            str(csv_path),
            "--out",
            str(out_dir),
            "--epochs",
            "2",
            "--hidden",
            "8",
            "--thresholds",
            "0.25",
            "0.5",
            "--smoke",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["smoke"] is True
    assert Path(summary["weights"]).exists()
    assert Path(summary["metrics_json"]).exists()
    assert Path(summary["threshold_sweep_csv"]).exists()
