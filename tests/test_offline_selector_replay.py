import csv
import json
import subprocess
import sys

from qstr_dronedet.tracking.offline_selector import replay_offline_selector


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_color_video(path, frame_specs, size=(64, 64)):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    assert writer.isOpened()
    for specs in frame_specs:
        frame = np.full((size[1], size[0], 3), 32, dtype=np.uint8)
        for bbox, color in specs:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            frame[y1:y2, x1:x2] = color
        writer.write(frame)
    writer.release()


def test_offline_selector_rejects_jump_and_recovers_from_memory(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 2, "bbox": [90, 90, 100, 100], "objectness": 0.95, "final_drone_score": 0.90, "source": "yolo"},
            {"frame_id": 3, "bbox": [16, 10, 26, 20], "objectness": 0.76, "final_drone_score": 0.66, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=3,
        max_jump_px=30.0,
        max_recover_frames=2,
        min_accept_score=0.20,
        min_motion_consistency=0.20,
        min_memory_consistency=0.20,
    )

    trajectory = _read_csv(result.trajectory_csv)
    debug = _read_csv(result.debug_csv)

    assert result.summary["accepted_candidate_frames"] == 3
    assert result.summary["memory_recover_frames"] == 1
    assert result.summary["fallback_needed_frames"] == 1
    assert trajectory[0]["selected_source"] == "yolo"
    assert trajectory[1]["selected_source"] == "yolo"
    assert trajectory[2]["selected_source"] == "memory_recover"
    assert trajectory[2]["state"] == "RECOVER"
    assert trajectory[2]["guard_reason"] == "jump_or_memory_inconsistent"
    assert float(trajectory[2]["x1"]) < 20.0
    assert trajectory[3]["selected_source"] == "yolo"
    assert trajectory[3]["state"] == "TRACK"
    assert any(row["guard_reason"] == "jump_or_memory_inconsistent" for row in debug)


def test_offline_selector_outputs_missing_frames_until_lost(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [{"frame_id": 0, "bbox": [30, 30, 40, 40], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"}],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        max_frame_id=2,
        max_recover_frames=1,
        min_accept_score=0.20,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert [row["frame_id"] for row in trajectory] == ["0", "1", "2"]
    assert trajectory[0]["selected_source"] == "yolo"
    assert trajectory[1]["selected_source"] == "memory_recover"
    assert trajectory[1]["state"] == "RECOVER"
    assert trajectory[2]["selected"] == "0"
    assert trajectory[2]["selected_source"] == "none"
    assert trajectory[2]["state"] == "LOST"
    assert result.summary["lost_frames"] == 1


def test_offline_selector_uses_lower_ranked_candidate_when_top_candidate_jumps(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [100, 100, 110, 110], "objectness": 0.95, "final_drone_score": 0.90, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.50, "final_drone_score": 0.45, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=3,
        max_jump_px=30.0,
        min_accept_score=0.20,
        min_motion_consistency=0.20,
        min_memory_consistency=0.20,
        detector_weight=0.90,
        motion_weight=0.05,
        memory_weight=0.05,
    )

    trajectory = _read_csv(result.trajectory_csv)
    debug = _read_csv(result.debug_csv)

    assert trajectory[1]["selected_source"] == "yolo"
    assert trajectory[1]["state"] == "TRACK"
    assert trajectory[1]["guard_reason"] == "accepted"
    assert float(trajectory[1]["x1"]) == 12.0
    assert result.summary["accepted_candidate_frames"] == 2
    assert result.summary["memory_recover_frames"] == 0
    rejected_rank1 = [row for row in debug if row["frame_id"] == "1" and row["rank"] == "1"][0]
    selected_rank2 = [row for row in debug if row["frame_id"] == "1" and row["rank"] == "2"][0]
    assert rejected_rank1["guard_reason"] == "jump_or_memory_inconsistent"
    assert selected_rank2["selected"] == "1"


def test_offline_selector_reacquires_from_deep_candidate_after_topk_guard_fails(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 2, "bbox": [100, 100, 110, 110], "objectness": 0.95, "final_drone_score": 0.90, "source": "yolo"},
            {"frame_id": 2, "bbox": [14, 10, 24, 20], "objectness": 0.30, "final_drone_score": 0.25, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_jump_px=30.0,
        min_accept_score=0.20,
        min_motion_consistency=0.20,
        min_memory_consistency=0.20,
        reacquire_top_k=0,
        reacquire_max_distance_px=20.0,
        reacquire_min_detector_score=0.10,
        reacquire_confirm_frames=1,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[2]["state"] == "REACQUIRE"
    assert trajectory[2]["selected_source"] == "yolo"
    assert trajectory[2]["guard_reason"] == "reacquire_memory_proximity"
    assert trajectory[2]["reacquire_reason"] == "deep_candidate_near_prediction"
    assert float(trajectory[2]["reacquire_distance_px"]) <= 20.0
    assert float(trajectory[2]["x1"]) == 14.0
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["memory_recover_frames"] == 0


def test_offline_selector_rejects_reacquire_candidate_with_inconsistent_size(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 2, "bbox": [0, 0, 40, 40], "objectness": 0.95, "final_drone_score": 0.90, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_jump_px=2.0,
        max_recover_frames=1,
        min_accept_score=0.20,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_top_k=0,
        reacquire_max_distance_px=96.0,
        reacquire_min_detector_score=0.10,
        reacquire_min_side_ratio=0.50,
        reacquire_max_side_ratio=2.0,
        reacquire_confirm_frames=1,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[2]["state"] == "RECOVER"
    assert trajectory[2]["selected_source"] == "memory_recover"
    assert trajectory[2]["reacquire_reason"] == ""
    assert result.summary["reacquired_frames"] == 0


def test_offline_selector_waits_for_reacquire_confirmation(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 2, "bbox": [14, 10, 24, 20], "objectness": 0.30, "final_drone_score": 0.25, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_jump_px=2.0,
        max_recover_frames=2,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_top_k=0,
        reacquire_max_distance_px=20.0,
        reacquire_min_detector_score=0.10,
        reacquire_confirm_frames=2,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[2]["state"] == "RECOVER"
    assert trajectory[2]["selected_source"] == "memory_recover"
    assert trajectory[2]["guard_reason"] == "reacquire_pending_confirmation"
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_pending_frames"] == 1


def test_offline_selector_confirms_reacquire_after_repeated_candidates(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 2, "bbox": [14, 10, 24, 20], "objectness": 0.30, "final_drone_score": 0.25, "source": "yolo"},
            {"frame_id": 3, "bbox": [16, 10, 26, 20], "objectness": 0.31, "final_drone_score": 0.26, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_jump_px=2.0,
        max_recover_frames=2,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_top_k=0,
        reacquire_max_distance_px=20.0,
        reacquire_min_detector_score=0.10,
        reacquire_confirm_frames=2,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[2]["state"] == "RECOVER"
    assert trajectory[2]["guard_reason"] == "reacquire_pending_confirmation"
    assert trajectory[3]["state"] == "REACQUIRE"
    assert trajectory[3]["reacquire_reason"] == "confirmed_deep_candidate_near_prediction"
    assert float(trajectory[3]["x1"]) == 16.0
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_confirmed_frames"] == 1


def test_offline_selector_globally_reacquires_after_stale_lost_with_size_band(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [100, 100, 150, 130], "objectness": 0.35, "final_drone_score": 0.30, "source": "yolo"},
            {"frame_id": 4, "bbox": [102, 100, 152, 130], "objectness": 0.36, "final_drone_score": 0.31, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_top_k=0,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[3]["guard_reason"] == "reacquire_pending_confirmation"
    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert float(trajectory[4]["x1"]) == 102.0
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_frames"] == 1


def test_offline_selector_global_reacquire_rejects_size_out_of_band(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [100, 100, 105, 105], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo"},
            {"frame_id": 4, "bbox": [102, 100, 107, 105], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo"},
            {"frame_id": 5, "bbox": [0, 0, 120, 120], "objectness": 0.92, "final_drone_score": 0.87, "source": "yolo"},
            {"frame_id": 6, "bbox": [2, 0, 122, 120], "objectness": 0.93, "final_drone_score": 0.88, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=6,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_top_k=0,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_frames"] == 0


def test_offline_selector_global_reacquire_tracklet_guard_rejects_rejected_tracklet(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.95,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.99,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": "tracklet_rejected",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.95,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.99,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": "tracklet_rejected",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
        reacquire_global_require_tracklet_confirmation=True,
        reacquire_global_min_tracklet_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_frames"] == 0
    assert result.summary["global_tracklet_rejected_frames"] >= 2


def test_offline_selector_global_reacquire_tracklet_guard_accepts_confirmed_tracklet(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.35,
                "final_drone_score": 0.30,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed+sequence_gate_rejected",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.36,
                "final_drone_score": 0.31,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed+sequence_gate_rejected",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_require_tracklet_confirmation=True,
        reacquire_global_min_tracklet_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[3]["guard_reason"] == "reacquire_pending_confirmation"
    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert float(trajectory[4]["x1"]) == 102.0
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_frames"] == 1
    assert result.summary["global_tracklet_rejected_frames"] == 0


def test_offline_selector_global_reacquire_rejected_tracklet_veto_blocks_explicit_rejection(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.95,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": True,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": "tracklet_rejected",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.95,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": True,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": "tracklet_rejected",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_reject_tracklet_rejected=True,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["global_tracklet_rejected_frames"] >= 2


def test_offline_selector_global_reacquire_rejected_tracklet_veto_allows_unapplied_tracklet_score(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": False,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": None,
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "tracklet_filter_applied": False,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
                "diagnostic_cause": None,
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_reject_tracklet_rejected=True,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["global_tracklet_rejected_frames"] == 0


def test_offline_selector_global_reacquire_delayed_confirmation_blocks_short_seed(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.95,
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.96,
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=6,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=60.0,
        reacquire_min_motion_consistency=0.01,
        reacquire_min_memory_consistency=0.01,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[4]["state"] == "LOST"
    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_frames"] == 0
    assert result.summary["reacquire_pending_frames"] == 2


def test_offline_selector_global_reacquire_delayed_confirmation_accepts_persistent_seed(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.95,
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.96,
            },
            {
                "frame_id": 5,
                "bbox": [104, 100, 154, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.97,
            },
            {
                "frame_id": 6,
                "bbox": [106, 100, 156, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.98,
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=6,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=60.0,
        reacquire_min_motion_consistency=0.01,
        reacquire_min_memory_consistency=0.01,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[4]["state"] == "LOST"
    assert trajectory[5]["state"] == "LOST"
    assert trajectory[6]["state"] == "REACQUIRE"
    assert trajectory[6]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_frames"] == 1
    assert result.summary["reacquire_pending_frames"] == 3


def test_offline_selector_global_reacquire_delayed_confirmation_accepts_tracklet_shortcut(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 150, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.95,
                "tracklet_filter_applied": False,
                "tracklet_is_drone": False,
                "tracklet_classifier_prob": 0.02,
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 152, 130],
                "objectness": 0.99,
                "final_drone_score": 0.10,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.96,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=6,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_frames"] == 1
    assert result.summary["reacquire_pending_frames"] == 1


def test_offline_selector_size_adaptive_global_reacquire_accepts_small_confirmed_seed(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.70,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.72,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=5,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_small_area_candidate_after_stale_lost"
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_frames"] == 1
    assert result.summary["reacquire_global_small_area_frames"] == 1


def test_offline_selector_size_adaptive_global_reacquire_rejects_sequence_gate_rejected_seed(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.95,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.97,
                "diagnostic_cause": "tracklet_confirmed+sequence_gate_rejected",
                "sequence_gate_confirmed": False,
                "sequence_gate_reason": "sequence_inconsistent",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.96,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.98,
                "diagnostic_cause": "tracklet_confirmed+sequence_gate_rejected",
                "sequence_gate_confirmed": False,
                "sequence_gate_reason": "sequence_inconsistent",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=5,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_frames"] == 0
    assert result.summary["reacquire_global_small_area_frames"] == 0


def test_offline_selector_size_adaptive_global_reacquire_requires_crop_persistence(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.80,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.10,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=5,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[4]["state"] == "LOST"
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_small_area_frames"] == 0
    assert result.summary["crop_rejected_frames"] >= 1


def test_offline_selector_size_adaptive_global_reacquire_rejects_far_small_seed(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [200, 200, 220, 210],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.80,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [202, 200, 222, 210],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.82,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=5,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_small_max_distance_px=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquired_frames"] == 0
    assert result.summary["reacquire_global_small_area_frames"] == 0


def test_offline_selector_small_global_probation_blocks_memory_near_reacquire(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.80,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.82,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 6,
                "bbox": [106, 100, 126, 110],
                "objectness": 1.00,
                "final_drone_score": 1.00,
                "source": "tracker",
                "crop_drone_score": 0.95,
            },
            {
                "frame_id": 7,
                "bbox": [108, 100, 128, 110],
                "objectness": 1.00,
                "final_drone_score": 1.00,
                "source": "tracker",
                "crop_drone_score": 0.96,
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=7,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.80,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=200.0,
        reacquire_min_detector_score=0.70,
        reacquire_min_motion_consistency=0.01,
        reacquire_min_memory_consistency=0.01,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_global_small_memory_probation_frames=8,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_small_area_candidate_after_stale_lost"
    assert all(row["state"] != "REACQUIRE" for row in trajectory[5:])
    assert result.summary["reacquired_frames"] == 1
    assert result.summary["reacquire_global_small_area_frames"] == 1
    assert result.summary["reacquire_memory_probation_rejected_frames"] >= 2


def test_offline_selector_small_global_probation_decays_memory_recover_after_grace(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.80,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.82,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {"frame_id": 5, "bbox": [148.5, 145, 168.5, 155], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=6,
        max_jump_px=5.0,
        max_recover_frames=1,
        min_accept_score=0.80,
        min_motion_consistency=0.80,
        min_memory_consistency=0.0,
        reacquire_max_distance_px=200.0,
        reacquire_min_detector_score=0.70,
        reacquire_min_motion_consistency=0.01,
        reacquire_min_memory_consistency=0.01,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_global_small_memory_probation_frames=8,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[5]["state"] == "TRACK"
    assert trajectory[6]["state"] == "LOST"
    assert result.summary["memory_recover_probation_rejected_frames"] == 1


def test_offline_selector_small_global_probation_decays_after_configured_frames(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 20, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 22, 20], "objectness": 1.00, "final_drone_score": 1.00, "source": "yolo"},
            {
                "frame_id": 3,
                "bbox": [100, 100, 120, 110],
                "objectness": 0.90,
                "final_drone_score": 0.45,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.80,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.95,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 4,
                "bbox": [102, 100, 122, 110],
                "objectness": 0.91,
                "final_drone_score": 0.46,
                "source": "tracker+yolo_tile",
                "crop_drone_score": 0.82,
                "tracklet_filter_applied": True,
                "tracklet_is_drone": True,
                "tracklet_classifier_prob": 0.96,
                "diagnostic_cause": "tracklet_confirmed",
            },
            {
                "frame_id": 7,
                "bbox": [108, 100, 128, 110],
                "objectness": 1.00,
                "final_drone_score": 1.00,
                "source": "tracker",
                "crop_drone_score": 0.95,
            },
            {
                "frame_id": 8,
                "bbox": [110, 100, 130, 110],
                "objectness": 1.00,
                "final_drone_score": 1.00,
                "source": "tracker",
                "crop_drone_score": 0.96,
            },
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=8,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.90,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=200.0,
        reacquire_min_detector_score=0.70,
        reacquire_min_motion_consistency=0.01,
        reacquire_min_memory_consistency=0.01,
        reacquire_confirm_frames=2,
        reacquire_global_delayed_confirm_frames=4,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=1000.0,
        reacquire_global_small_min_area=100.0,
        reacquire_global_max_area=3000.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_global_min_tracklet_score=0.50,
        reacquire_global_small_memory_probation_frames=1,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[8]["state"] == "REACQUIRE"
    assert trajectory[8]["reacquire_reason"] == "confirmed_deep_candidate_near_prediction"
    assert result.summary["reacquired_frames"] == 2
    assert result.summary["reacquire_memory_frames"] == 1


def test_offline_selector_global_reacquire_accepts_appearance_match(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    video = tmp_path / "appearance.mp4"
    red = (0, 0, 255)
    _write_color_video(
        video,
        [
            [([10, 10, 22, 22], red)],
            [([12, 10, 24, 22], red)],
            [],
            [([40, 30, 56, 46], red)],
            [([42, 30, 58, 46], red)],
        ],
    )
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo"},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_min_appearance_similarity=0.70,
        video=video,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[3]["guard_reason"] == "reacquire_pending_confirmation"
    assert trajectory[4]["state"] == "REACQUIRE"
    assert trajectory[4]["reacquire_reason"] == "confirmed_global_candidate_after_stale_lost"
    assert float(trajectory[4]["appearance_similarity"]) >= 0.70
    assert result.summary["reacquire_global_frames"] == 1
    assert result.summary["appearance_rejected_frames"] == 0


def test_offline_selector_global_reacquire_rejects_appearance_mismatch(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    video = tmp_path / "appearance.mp4"
    red = (0, 0, 255)
    blue = (255, 0, 0)
    _write_color_video(
        video,
        [
            [([10, 10, 22, 22], red)],
            [([12, 10, 24, 22], red)],
            [],
            [([40, 30, 56, 46], blue)],
            [([42, 30, 58, 46], blue)],
        ],
    )
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo"},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_min_appearance_similarity=0.70,
        video=video,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["reacquire_global_frames"] == 0
    assert result.summary["appearance_rejected_frames"] >= 2


def test_offline_selector_appearance_weight_prefers_visual_match(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    video = tmp_path / "appearance_weight.mp4"
    red = (0, 0, 255)
    blue = (255, 0, 0)
    _write_color_video(
        video,
        [
            [([10, 10, 22, 22], red)],
            [([12, 10, 24, 22], red)],
            [],
            [([40, 30, 56, 46], red), ([8, 38, 24, 54], blue)],
            [([42, 30, 58, 46], red), ([10, 38, 26, 54], blue)],
        ],
    )
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [8, 38, 24, 54], "objectness": 0.99, "final_drone_score": 0.95, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.45, "final_drone_score": 0.40, "source": "yolo"},
            {"frame_id": 4, "bbox": [10, 38, 26, 54], "objectness": 0.99, "final_drone_score": 0.95, "source": "yolo"},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.46, "final_drone_score": 0.41, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_appearance_weight=1.0,
        video=video,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert float(trajectory[4]["x1"]) == 42.0
    assert float(trajectory[4]["appearance_similarity"]) > 0.90
    assert result.summary["appearance_soft_scored_frames"] >= 2
    assert result.summary["appearance_rejected_frames"] == 0


def test_offline_selector_appearance_weight_does_not_hard_reject_only_candidate(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    video = tmp_path / "appearance_weight.mp4"
    red = (0, 0, 255)
    blue = (255, 0, 0)
    _write_color_video(
        video,
        [
            [([10, 10, 22, 22], red)],
            [([12, 10, 24, 22], red)],
            [],
            [([40, 30, 56, 46], blue)],
            [([42, 30, 58, 46], blue)],
        ],
    )
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo"},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo"},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_appearance_weight=1.0,
        video=video,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert float(trajectory[4]["x1"]) == 42.0
    assert result.summary["appearance_soft_scored_frames"] >= 2
    assert result.summary["appearance_rejected_frames"] == 0


def test_offline_selector_crop_score_weight_prefers_classifier_positive_candidate(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [8, 38, 24, 54], "objectness": 0.99, "final_drone_score": 0.95, "source": "yolo", "crop_drone_score": 0.05},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.45, "final_drone_score": 0.40, "source": "yolo", "crop_drone_score": 0.95},
            {"frame_id": 4, "bbox": [10, 38, 26, 54], "objectness": 0.99, "final_drone_score": 0.95, "source": "yolo", "crop_drone_score": 0.05},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.46, "final_drone_score": 0.41, "source": "yolo", "crop_drone_score": 0.95},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_crop_weight=1.0,
    )

    trajectory = _read_csv(result.trajectory_csv)
    debug = _read_csv(result.debug_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert float(trajectory[4]["x1"]) == 42.0
    assert float(trajectory[4]["crop_drone_score"]) == 0.95
    assert result.summary["crop_soft_scored_frames"] >= 2
    assert result.summary["crop_rejected_frames"] == 0
    selected_debug = [row for row in debug if row["frame_id"] == "4" and row["selected"] == "1"][-1]
    assert float(selected_debug["crop_drone_score"]) == 0.95


def test_offline_selector_crop_score_weight_does_not_hard_reject_only_candidate(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo", "crop_drone_score": 0.05},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo", "crop_drone_score": 0.05},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_crop_weight=1.0,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[4]["state"] == "REACQUIRE"
    assert float(trajectory[4]["crop_drone_score"]) == 0.05
    assert result.summary["crop_soft_scored_frames"] >= 2
    assert result.summary["crop_rejected_frames"] == 0


def test_offline_selector_crop_score_threshold_rejects_low_classifier_candidate(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 30, 56, 46], "objectness": 0.90, "final_drone_score": 0.85, "source": "yolo", "crop_drone_score": 0.05},
            {"frame_id": 4, "bbox": [42, 30, 58, 46], "objectness": 0.91, "final_drone_score": 0.86, "source": "yolo", "crop_drone_score": 0.05},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=5.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.80,
        min_memory_consistency=0.80,
        reacquire_max_distance_px=20.0,
        reacquire_confirm_frames=2,
        reacquire_stale_after_frames=1,
        reacquire_global_min_area=100.0,
        reacquire_global_max_area=400.0,
        reacquire_global_min_detector_score=0.10,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert all(row["state"] != "REACQUIRE" for row in trajectory[3:])
    assert result.summary["crop_rejected_frames"] >= 2
    assert result.summary["reacquired_frames"] == 0


def test_offline_selector_crop_threshold_applies_to_track_confirmation(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [
            {"frame_id": 0, "bbox": [10, 10, 22, 22], "objectness": 0.80, "final_drone_score": 0.70, "source": "yolo"},
            {"frame_id": 1, "bbox": [12, 10, 24, 22], "objectness": 0.78, "final_drone_score": 0.68, "source": "yolo"},
            {"frame_id": 3, "bbox": [40, 10, 52, 22], "objectness": 0.30, "final_drone_score": 0.25, "source": "yolo", "crop_drone_score": 0.95},
            {"frame_id": 4, "bbox": [42, 10, 54, 22], "objectness": 0.99, "final_drone_score": 0.95, "source": "yolo", "crop_drone_score": 0.05},
        ],
    )

    result = replay_offline_selector(
        pred,
        tmp_path / "selector",
        top_k=1,
        max_frame_id=4,
        max_jump_px=40.0,
        max_recover_frames=0,
        min_accept_score=0.70,
        min_motion_consistency=0.20,
        min_memory_consistency=0.20,
        reacquire_max_distance_px=40.0,
        reacquire_confirm_frames=2,
        reacquire_min_detector_score=0.10,
        reacquire_min_motion_consistency=0.20,
        reacquire_min_memory_consistency=0.20,
        reacquire_crop_score_field="crop_drone_score",
        reacquire_min_crop_drone_score=0.50,
    )

    trajectory = _read_csv(result.trajectory_csv)

    assert trajectory[3]["state"] == "LOST"
    assert trajectory[3]["guard_reason"] == "reacquire_pending_confirmation"
    assert trajectory[4]["state"] != "REACQUIRE"
    assert result.summary["crop_rejected_frames"] >= 1
    assert result.summary["reacquired_frames"] == 0


def test_offline_selector_replay_cli_writes_outputs(tmp_path):
    pred = tmp_path / "predictions.jsonl"
    _write_jsonl(
        pred,
        [{"frame_id": 0, "bbox": [5, 5, 15, 15], "objectness": 0.8, "final_drone_score": 0.7, "source": "yolo"}],
    )
    out = tmp_path / "selector"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qstr_dronedet.cli",
            "offline-selector-replay",
            "--predictions",
            str(pred),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out / "trajectory.csv").exists()
    assert (out / "selector_debug.csv").exists()
    assert (out / "selector_summary.json").exists()
