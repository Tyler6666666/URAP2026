import csv
import json
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.tracking.proposal_tracklets import (
    build_proposal_tracklet_dataset,
    compare_flat_prediction_eval_summaries,
    evaluate_flat_tracklet_predictions,
    export_frame_list_from_gt_csv,
    export_temporal_saliency_tracklets,
    export_tracklet_jsonl_predictions,
    export_yolo_labels_to_gt_csv,
    export_yolo_oracle_tracklets,
    export_yolo_predictions_to_route_b_run,
    merge_tracklet_jsonl,
    scan_route_b_proposal_inputs,
    sweep_flat_tracklet_prediction_nms,
    validate_route_b_proposal_inputs,
    write_route_b_proposal_run_manifest,
)


def _row(frame_id, box, score, track_id=None, source="fallback_yolo", crop=0.5, temporal=0.6, bg=0.3):
    row = {
        "frame_id": frame_id,
        "bbox": box,
        "objectness": score,
        "source": source,
        "final_drone_score": score * temporal,
        "predicted_class": "drone" if temporal > bg else "background",
        "crop_probs": {"drone": crop, "background": 1.0 - crop},
        "temporal_probs": {"drone": temporal, "background": 1.0 - temporal},
        "final_probs": {"drone": temporal, "background": bg},
        "alignment_quality": 0.8,
    }
    if track_id is not None:
        row["track_id"] = track_id
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_proposal_tracklets_relinks_untracked_rows(tmp_path):
    run_root = tmp_path / "runs"
    seq_dir = run_root / "hard_recovery" / "seq001"
    rows = [
        _row(0, [10, 10, 20, 20], 0.20),
        _row(1, [11, 10, 21, 20], 0.22),
        _row(0, [100, 100, 110, 110], 0.40, crop=0.2, temporal=0.2, bg=0.8),
        _row(1, [101, 100, 111, 110], 0.42, crop=0.2, temporal=0.2, bg=0.8),
    ]
    _write_jsonl(seq_dir / "diagnostics_raw.jsonl", rows)
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 1, 11, 10, 21, 20, "drone", "tiny"])

    result = build_proposal_tracklet_dataset([run_root], gt, tmp_path / "proposal_tracklets", min_tracklet_rows=2)

    assert result.summary["num_tracklets"] == 2
    assert result.summary["positives"] == 1
    assert result.summary["bucket_counts"]["hard_tiny_positive"] == 1
    assert result.summary["bucket_counts"]["high_score_detector_fp"] == 1
    assert result.json_path.exists()


def test_merge_tracklet_jsonl_tags_sources(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    item_a = {"meta": {"seq": "s", "track_id": "1", "label": 1, "bucket": "positive"}, "rows": []}
    item_b = {"meta": {"seq": "s", "track_id": "1", "label": 0, "bucket": "high_score_detector_fp"}, "rows": []}
    first.write_text(json.dumps(item_a) + "\n", encoding="utf-8")
    second.write_text(json.dumps(item_b) + "\n", encoding="utf-8")

    result = merge_tracklet_jsonl([first, second], tmp_path / "merged" / "tracklets.jsonl", source_names=["orig", "proposal"])

    lines = [json.loads(line) for line in result.json_path.read_text(encoding="utf-8").splitlines()]
    assert result.summary["num_tracklets"] == 2
    assert {line["meta"]["dataset_source"] for line in lines} == {"orig", "proposal"}


def test_validate_route_b_proposal_inputs(tmp_path):
    run_root = tmp_path / "runs"
    seq_dir = run_root / "hard_recovery" / "seq001"
    _write_jsonl(seq_dir / "diagnostics_raw.jsonl", [_row(0, [10, 10, 20, 20], 0.20)])
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    result = validate_route_b_proposal_inputs(
        [run_root],
        [gt],
        [run_root],
        [gt],
        tmp_path / "proposal_preflight.json",
        train_source_names=["nps"],
        eval_dataset_names=["nps"],
    )

    assert result.out_path.exists()
    assert result.summary["valid"] is True
    assert result.summary["train_bbox_rows"] == 1
    assert result.summary["eval_bbox_rows"] == 1
    assert result.summary["train"][0]["matched_gt_sequences"] == 1


def test_validate_route_b_proposal_inputs_flags_missing_overlap(tmp_path):
    run_root = tmp_path / "runs"
    seq_dir = run_root / "hard_recovery" / "seq001"
    _write_jsonl(seq_dir / "diagnostics_raw.jsonl", [_row(0, [10, 10, 20, 20], 0.20)])
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "other_seq" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    result = validate_route_b_proposal_inputs(
        [run_root],
        [gt],
        [run_root],
        [gt],
        tmp_path / "bad_preflight.json",
    )

    assert result.summary["valid"] is False
    assert any("no sequence names overlap" in issue for issue in result.summary["issues"])


def test_write_route_b_proposal_run_manifest(tmp_path):
    run_root = tmp_path / "runs"
    seq_dir = run_root / "hard_recovery" / "seq001"
    _write_jsonl(seq_dir / "diagnostics_raw.jsonl", [_row(0, [10, 10, 20, 20], 0.20)])
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    result = write_route_b_proposal_run_manifest(
        tmp_path / "manifest",
        [run_root],
        [gt],
        [run_root],
        [gt],
        train_source_names=["nps"],
        eval_dataset_names=["nps"],
        run_id="route_b_smoke",
        past_len=3,
        future_len=2,
        model_types=["mlp"],
        epochs=2,
        hidden=16,
        batch_size=4,
        thresholds=[0.5],
        balance_by=["dataset_source"],
    )

    assert result.out_path.exists()
    assert result.summary["preflight_valid"] is True
    assert result.summary["commands"]["start_detached"].startswith(".\\tools\\start_route_b_proposal_benchmark_detached.ps1")
    assert "-RunId 'route_b_smoke'" in result.summary["commands"]["start_detached"]
    assert "validate-route-b-proposal-inputs" in result.summary["commands"]["preflight"]
    assert (tmp_path / "manifest" / "start_route_b_proposal_benchmark.ps1").exists()
    assert (tmp_path / "manifest" / "monitor_route_b_proposal_benchmark.ps1").exists()
    assert (tmp_path / "manifest" / "proposal_preflight.json").exists()


def test_scan_route_b_proposal_inputs_finds_run_root_and_gt(tmp_path):
    run_root = tmp_path / "nps_runs"
    for seq in ["seq001", "seq002"]:
        _write_jsonl(run_root / "hard_recovery" / seq / "diagnostics_raw.jsonl", [_row(0, [10, 10, 20, 20], 0.20)])
    gt = tmp_path / "nps_gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writerow([str(tmp_path / "seq001" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])
        writer.writerow([str(tmp_path / "seq002" / "visible.mp4"), 0, 10, 10, 20, 20, "drone", "tiny"])

    result = scan_route_b_proposal_inputs([tmp_path], tmp_path / "scan.json", max_depth=5)

    assert result.out_path.exists()
    assert result.summary["num_run_candidates"] == 1
    assert result.summary["num_gt_candidates"] == 1
    candidate = result.summary["run_candidates"][0]
    assert candidate["run_root"] == str(run_root)
    assert candidate["best_gt_csv"] == str(gt)
    assert candidate["best_gt_sequence_overlap"] == 2
    assert result.summary["suggested_manifest_inputs"][0]["gt_csv"] == str(gt)


def test_scan_route_b_proposal_inputs_reports_yolo_dataset_candidates(tmp_path):
    dataset = tmp_path / "YOLOMG_eval" / "NPS_test"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir(parents=True)
    (dataset / "images" / "Clip_1_00001.png").write_bytes(b"fake")
    (dataset / "labels" / "Clip_1_00001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (dataset / "val.txt").write_text(str(dataset / "images" / "Clip_1_00001.png") + "\n", encoding="utf-8")
    (dataset / "nps_test_yolomg.yaml").write_text("path: .\n", encoding="utf-8")

    result = scan_route_b_proposal_inputs([tmp_path], tmp_path / "scan_yolo.json", max_depth=5)

    assert result.summary["num_run_candidates"] == 0
    assert result.summary["num_yolo_dataset_candidates"] == 1
    candidate = result.summary["yolo_dataset_candidates"][0]
    assert candidate["root"] == str(dataset)
    assert candidate["has_images_dir"] is True
    assert candidate["has_labels_dir"] is True
    assert candidate["total_list_rows"] == 1


def test_export_yolo_oracle_tracklets(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_paths = []
    for frame_id in range(4):
        image = images / f"Clip_1_{frame_id:05d}.png"
        image.write_bytes(b"fake")
        label = labels / f"Clip_1_{frame_id:05d}.txt"
        label.write_text("0 0.500000 0.500000 0.100000 0.200000\n", encoding="utf-8")
        image_paths.append(image)
    list_file = dataset / "val.txt"
    list_file.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")

    result = export_yolo_oracle_tracklets(
        [list_file],
        tmp_path / "oracle",
        dataset_source="nps_oracle",
        image_size=(100, 50),
        min_tracklet_rows=2,
    )

    items = [json.loads(line) for line in result.json_path.read_text(encoding="utf-8").splitlines()]
    assert result.summary["images_seen"] == 4
    assert result.summary["labels_seen"] == 4
    assert result.summary["num_tracklets"] == 1
    assert items[0]["meta"]["dataset_source"] == "nps_oracle"
    assert items[0]["meta"]["bucket"] == "oracle_yolo_positive"
    assert len(items[0]["rows"]) == 4
    assert items[0]["rows"][0]["bbox"] == [45.0, 20.0, 55.0, 30.0]

    skipped = export_yolo_oracle_tracklets(
        [list_file],
        tmp_path / "oracle_skip",
        dataset_source="nps_oracle",
        image_size=(100, 50),
        max_images=2,
        skip_images=2,
        min_tracklet_rows=2,
    )
    skipped_items = [json.loads(line) for line in skipped.json_path.read_text(encoding="utf-8").splitlines()]
    assert skipped.summary["images_skipped"] == 2
    assert skipped.summary["images_seen"] == 2
    assert skipped.summary["labels_seen"] == 2
    assert skipped_items[0]["rows"][0]["frame_id"] == 2

    for frame_id in range(4):
        image = images / f"phantom109_{frame_id + 1:04d}.jpg"
        image.write_bytes(b"fake")
        label = labels / f"phantom109_{frame_id + 1:04d}.txt"
        label.write_text("0 0.400000 0.500000 0.100000 0.200000\n", encoding="utf-8")
        image_paths.append(image)
    list_file.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
    capped = export_yolo_oracle_tracklets(
        [list_file],
        tmp_path / "oracle_capped",
        dataset_source="mixed_oracle",
        image_size=(100, 50),
        max_labeled_images_per_seq=2,
        min_tracklet_rows=2,
    )
    capped_items = [json.loads(line) for line in capped.json_path.read_text(encoding="utf-8").splitlines()]
    assert capped.summary["images_seen"] == 4
    assert capped.summary["labels_seen"] == 4
    assert capped.summary["labeled_images_by_seq"] == {"Clip_1": 2, "phantom109": 2}
    assert {item["meta"]["seq"] for item in capped_items} == {"Clip_1", "phantom109"}


def test_export_yolo_predictions_to_route_b_run_and_gt_csv(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    labels = dataset / "labels"
    pred_labels = tmp_path / "pred_labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    pred_labels.mkdir(parents=True)
    image_paths = []
    for frame_id in range(2):
        image = images / f"Clip_9_{frame_id:05d}.png"
        image.write_bytes(b"fake")
        (labels / f"Clip_9_{frame_id:05d}.txt").write_text("0 0.500000 0.500000 0.100000 0.200000\n", encoding="utf-8")
        (pred_labels / f"Clip_9_{frame_id:05d}.txt").write_text("0 0.500000 0.500000 0.100000 0.200000 0.250000\n", encoding="utf-8")
        image_paths.append(image)
    list_file = dataset / "val.txt"
    list_file.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")

    gt = export_yolo_labels_to_gt_csv([list_file], tmp_path / "gt.csv", image_size=(100, 50))
    run = export_yolo_predictions_to_route_b_run(
        [list_file],
        pred_labels,
        tmp_path / "route_b_run",
        image_size=(100, 50),
    )

    gt_rows = list(csv.DictReader(gt.out_path.open("r", encoding="utf-8")))
    diag_path = tmp_path / "route_b_run" / "hard_recovery" / "Clip_9" / "diagnostics_raw.jsonl"
    diag_rows = [json.loads(line) for line in diag_path.read_text(encoding="utf-8").splitlines()]

    assert gt.summary["labels_seen"] == 2
    assert gt_rows[0]["video_path"] == str(Path("Clip_9") / "visible.mp4")
    assert run.summary["prediction_rows"] == 2
    assert run.summary["sequences"] == 1
    assert diag_rows[0]["source"] == "yolomg_lowconf"
    assert diag_rows[0]["objectness"] == 0.25
    assert diag_rows[0]["bbox"] == [45.0, 20.0, 55.0, 30.0]


def test_export_yolo_labels_gt_csv_parses_numeric_frame_stems(tmp_path):
    seq = tmp_path / "dji_train_pool_seq"
    frames = seq / "frames"
    labels = seq / "labels"
    frames.mkdir(parents=True)
    labels.mkdir()
    image = frames / "000123.jpg"
    image.write_bytes(b"fake")
    (labels / "000123.txt").write_text("0 0.500000 0.500000 0.100000 0.200000\n", encoding="utf-8")
    list_file = tmp_path / "images.txt"
    list_file.write_text(str(image) + "\n", encoding="utf-8")

    gt = export_yolo_labels_to_gt_csv([list_file], tmp_path / "gt.csv", image_size=(100, 50))

    rows = list(csv.DictReader(gt.out_path.open("r", encoding="utf-8")))
    assert rows[0]["video_path"] == str(Path("dji_train_pool_seq") / "visible.mp4")
    assert rows[0]["frame_id"] == "123"


def test_export_temporal_saliency_tracklets_from_moving_tiny_blob(tmp_path):
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    image_paths = []
    for frame_id in range(4):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        x = 18 + frame_id * 3
        cv2.rectangle(image, (x, 20), (x + 4, 24), (255, 255, 255), thickness=-1)
        path = images / f"Clip_2_{frame_id:05d}.png"
        cv2.imwrite(str(path), image)
        image_paths.append(path)
    list_file = dataset / "val.txt"
    list_file.write_text("\n".join(str(path) for path in image_paths) + "\n", encoding="utf-8")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        for frame_id in range(1, 4):
            x = 18 + frame_id * 3
            writer.writerow([str(Path("Clip_2") / "visible.mp4"), frame_id, x, 20, x + 5, 25, "drone", "tiny"])

    result = export_temporal_saliency_tracklets(
        [list_file],
        gt,
        tmp_path / "saliency",
        threshold=16.0,
        min_area=1.0,
        max_area=80.0,
        dilate_iters=0,
        min_tracklet_rows=2,
        center_threshold=8.0,
    )

    items = [json.loads(line) for line in result.json_path.read_text(encoding="utf-8").splitlines()]
    assert result.summary["images_seen"] == 4
    assert result.summary["candidate_rows"] >= 2
    assert result.summary["num_tracklets"] >= 1
    assert result.summary["positives"] >= 1
    assert items[0]["meta"]["dataset_source"] == "vatd_temporal_saliency"
    assert items[0]["rows"][0]["source"] == "vatd_temporal_saliency"
    assert items[0]["rows"][0]["image_width"] == 64


def test_export_frame_list_from_gt_csv_matches_flat_frame_root(tmp_path):
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    for name in [
        "Clip_37_00001.png",
        "Clip_37_00002.png",
        "Clip_38_00001.png",
        "Other_00001.png",
    ]:
        (frame_root / name).write_bytes(b"fake")
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writerow(["Clip_37", 1, 1, 1, 2, 2, "Clip_37/Clip_37_00001.png"])
        writer.writerow(["Clip_38", 1, 3, 3, 4, 4, "Clip_38/Clip_38_00001.png"])

    result = export_frame_list_from_gt_csv(gt, frame_root, tmp_path / "lists" / "frames.txt")

    lines = result.out_path.read_text(encoding="utf-8").splitlines()
    assert result.summary["gt_sequences"] == 2
    assert result.summary["matched_sequences"] == 2
    assert result.summary["frames"] == 3
    assert [Path(line).name for line in lines] == ["Clip_37_00001.png", "Clip_37_00002.png", "Clip_38_00001.png"]

    capped = export_frame_list_from_gt_csv(gt, frame_root, tmp_path / "lists" / "frames_capped.txt", max_frames_per_seq=1)
    capped_lines = capped.out_path.read_text(encoding="utf-8").splitlines()
    assert capped.summary["frames_by_seq"] == {"Clip_37": 1, "Clip_38": 1}
    assert [Path(line).name for line in capped_lines] == ["Clip_37_00001.png", "Clip_38_00001.png"]


def test_export_tracklet_jsonl_predictions_writes_flat_and_yolo(tmp_path):
    tracklets = tmp_path / "tracklets.jsonl"
    item = {
        "meta": {"seq": "Clip_1", "track_id": "t1", "vatd_score": 0.75, "dataset_source": "vatd_temporal_saliency"},
        "rows": [
            {
                "seq": "Clip_1",
                "frame_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "image_path": str(tmp_path / "Clip_1_00001.png"),
                "image_width": 100,
                "image_height": 80,
                "source": "vatd_temporal_saliency",
            }
        ],
    }
    tracklets.write_text(json.dumps(item) + "\n", encoding="utf-8")

    result = export_tracklet_jsonl_predictions(tracklets, tmp_path / "predictions", dataset_name="vatd_independent")

    flat_rows = list(csv.DictReader((tmp_path / "predictions" / "flat_xyxy_predictions.csv").open("r", encoding="utf-8")))
    yolo_txt = tmp_path / "predictions" / "yolo_txt" / "vatd_independent" / "labels" / "Clip_1_00001.txt"
    assert result.summary["rows_exported"] == 1
    assert result.summary["yolo_label_files"] == 1
    assert flat_rows[0]["dataset"] == "vatd_independent"
    assert flat_rows[0]["score"] == "0.75"
    assert flat_rows[0]["x1"] == "10.0"
    assert yolo_txt.read_text(encoding="utf-8").strip() == "0 0.20000000 0.37500000 0.20000000 0.25000000 0.75000000"


def test_export_tracklet_jsonl_predictions_can_nms_duplicate_frame_boxes(tmp_path):
    tracklets = tmp_path / "tracklets.jsonl"
    items = [
        {
            "meta": {"seq": "Clip_1", "track_id": "high", "vatd_score": 0.90, "dataset_source": "vatd_temporal_saliency"},
            "rows": [
                {
                    "seq": "Clip_1",
                    "frame_id": 1,
                    "bbox": [10.0, 10.0, 30.0, 30.0],
                    "image_path": str(tmp_path / "Clip_1_00001.png"),
                    "image_width": 100,
                    "image_height": 100,
                    "source": "vatd_temporal_saliency",
                }
            ],
        },
        {
            "meta": {"seq": "Clip_1", "track_id": "low", "vatd_score": 0.60, "dataset_source": "vatd_temporal_saliency"},
            "rows": [
                {
                    "seq": "Clip_1",
                    "frame_id": 1,
                    "bbox": [11.0, 11.0, 31.0, 31.0],
                    "image_path": str(tmp_path / "Clip_1_00001.png"),
                    "image_width": 100,
                    "image_height": 100,
                    "source": "vatd_temporal_saliency",
                }
            ],
        },
    ]
    tracklets.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")

    result = export_tracklet_jsonl_predictions(
        tracklets,
        tmp_path / "predictions",
        dataset_name="vatd_independent",
        nms_iou_threshold=0.5,
    )

    flat_rows = list(csv.DictReader((tmp_path / "predictions" / "flat_xyxy_predictions.csv").open("r", encoding="utf-8")))
    yolo_txt = tmp_path / "predictions" / "yolo_txt" / "vatd_independent" / "labels" / "Clip_1_00001.txt"
    assert result.summary["rows_after_score_filter"] == 2
    assert result.summary["rows_exported"] == 1
    assert result.summary["rows_suppressed_nms"] == 1
    assert flat_rows[0]["track_id"] == "high"
    assert flat_rows[0]["score"] == "0.9"
    assert len(yolo_txt.read_text(encoding="utf-8").splitlines()) == 1


def test_evaluate_flat_tracklet_predictions_sweeps_same_fp_recall(tmp_path):
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writerow(["Clip_1", 1, 10, 10, 20, 20, "Clip_1/Clip_1_00001.png"])
        writer.writerow(["Clip_1", 2, 30, 30, 40, 40, "Clip_1/Clip_1_00002.png"])

    pred = tmp_path / "pred.csv"
    with pred.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "seq", "frame_id", "image_stem", "class_id", "score", "x1", "y1", "x2", "y2", "track_id", "image_path", "source"])
        writer.writerow(["vatd", "Clip_1", 1, "Clip_1_00001", 0, 0.90, 10, 10, 20, 20, "t1", "", "vatd_temporal_saliency"])
        writer.writerow(["vatd", "Clip_1", 2, "Clip_1_00002", 0, 0.40, 30, 30, 40, 40, "t1", "", "vatd_temporal_saliency"])
        writer.writerow(["vatd", "Clip_1", 2, "Clip_1_00002", 0, 0.35, 70, 70, 80, 80, "fp", "", "vatd_temporal_saliency"])

    result = evaluate_flat_tracklet_predictions(
        gt,
        pred,
        tmp_path / "eval",
        thresholds=[0.9, 0.4, 0.35],
        iou_threshold=0.5,
        fp_limit=0,
        fp_limits=[0, 1],
        max_fppis=[0.0, 0.5],
    )

    rows = list(csv.DictReader(result.csv_path.open("r", encoding="utf-8")))
    assert result.summary["gt_boxes"] == 2
    assert result.summary["prediction_rows"] == 3
    assert result.summary["best_f1"]["threshold"] == 0.4
    assert result.summary["best_under_budget"]["threshold"] == 0.4
    assert result.summary["best_under_budget"]["recall"] == 1.0
    assert result.summary["fp_budget_curve"][0]["fp_limit"] == 0
    assert result.summary["fp_budget_curve"][0]["best"]["recall"] == 1.0
    assert result.summary["fp_budget_curve"][1]["fp_limit"] == 1
    assert result.summary["fp_budget_curve"][1]["best"]["fp"] == 0
    assert result.summary["fppi_budget_curve"][0]["max_fppi"] == 0.0
    assert result.summary["fppi_budget_curve"][0]["best"]["fp"] == 0
    fp_curve_rows = list(csv.DictReader((tmp_path / "eval" / "flat_prediction_fp_budget_curve.csv").open("r", encoding="utf-8")))
    fppi_curve_rows = list(csv.DictReader((tmp_path / "eval" / "flat_prediction_fppi_budget_curve.csv").open("r", encoding="utf-8")))
    assert fp_curve_rows[0]["fp_limit"] == "0"
    assert fp_curve_rows[0]["recall"] == "1.0"
    assert fppi_curve_rows[0]["max_fppi"] == "0.0"
    assert fppi_curve_rows[0]["available"] == "True"
    assert rows[0]["threshold"] == "0.9"
    assert rows[-1]["fp"] == "1"


def test_sweep_flat_tracklet_prediction_nms_selects_budgeted_duplicate_suppression(tmp_path):
    gt = tmp_path / "gt.csv"
    with gt.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writerow(["Clip_1", 1, 10, 10, 30, 30, "Clip_1/Clip_1_00001.png"])

    tracklets = tmp_path / "tracklets.jsonl"
    items = [
        {
            "meta": {"seq": "Clip_1", "track_id": "tp", "vatd_score": 0.90, "dataset_source": "vatd_temporal_saliency"},
            "rows": [
                {
                    "seq": "Clip_1",
                    "frame_id": 1,
                    "bbox": [10.0, 10.0, 30.0, 30.0],
                    "image_width": 100,
                    "image_height": 100,
                    "source": "vatd_temporal_saliency",
                }
            ],
        },
        {
            "meta": {"seq": "Clip_1", "track_id": "dup", "vatd_score": 0.80, "dataset_source": "vatd_temporal_saliency"},
            "rows": [
                {
                    "seq": "Clip_1",
                    "frame_id": 1,
                    "bbox": [12.0, 10.0, 32.0, 30.0],
                    "image_width": 100,
                    "image_height": 100,
                    "source": "vatd_temporal_saliency",
                }
            ],
        },
    ]
    tracklets.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")

    result = sweep_flat_tracklet_prediction_nms(
        tracklets,
        gt,
        tmp_path / "sweep",
        dataset_name="vatd_independent",
        iou_thresholds=[None],
        center_thresholds=[None, 6.0],
        score_thresholds=[0.0],
        fp_limit=0,
    )

    rows = list(csv.DictReader(result.csv_path.open("r", encoding="utf-8")))
    assert result.summary["num_runs"] == 2
    assert result.summary["best_under_budget"]["nms_center_threshold"] == 6.0
    assert result.summary["best_under_budget"]["tp"] == 1
    assert result.summary["best_under_budget"]["fp"] == 0
    assert result.summary["selected_final"]["nms_center_threshold"] == 6.0
    assert result.summary["final_prediction_export"]["rows_exported"] == 1
    assert result.summary["final_eval"]["best_under_budget"]["fp"] == 0
    assert (tmp_path / "sweep" / "final_predictions" / "flat_xyxy_predictions.csv").exists()
    assert (tmp_path / "sweep" / "final_predictions" / "yolo_txt" / "vatd_independent" / "labels").exists()
    assert {row["best_under_budget_available"] for row in rows} == {"False", "True"}


def test_compare_flat_prediction_eval_summaries_writes_csv_and_markdown(tmp_path):
    first = tmp_path / "first" / "flat_prediction_eval_summary.json"
    second = tmp_path / "second" / "flat_prediction_eval_summary.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    summary_a = {
        "best_f1": {"threshold": 0.5, "tp": 2, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1.0, "f1": 0.8, "fppi": 0.5},
        "best_under_budget": {"threshold": 0.8, "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3, "fppi": 0.0},
        "fp_budget_curve": [{"fp_limit": 0, "best": {"threshold": 0.8, "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3, "fppi": 0.0}}],
        "fppi_budget_curve": [{"max_fppi": 0.0, "best": {"threshold": 0.8, "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3, "fppi": 0.0}}],
    }
    summary_b = {
        "best_f1": {"threshold": 0.4, "tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fppi": 0.0},
        "best_under_budget": {"threshold": 0.4, "tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fppi": 0.0},
        "fp_budget_curve": [{"fp_limit": 0, "best": {"threshold": 0.4, "tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "fppi": 0.0}}],
    }
    first.write_text(json.dumps(summary_a), encoding="utf-8")
    second.write_text(json.dumps(summary_b), encoding="utf-8")

    result = compare_flat_prediction_eval_summaries(
        [first, second],
        tmp_path / "comparison",
        method_names=["baseline", "vatd"],
    )

    rows = list(csv.DictReader(result.csv_path.open("r", encoding="utf-8")))
    assert result.summary["rows"] == 7
    assert rows[0]["method"] == "baseline"
    assert rows[0]["budget_type"] == "best_f1"
    vatd_fp = next(row for row in rows if row["method"] == "vatd" and row["budget_type"] == "fp_limit")
    assert vatd_fp["recall"] == "1.0"
    assert float(vatd_fp["delta_recall_vs_baseline"]) > 0.0
    assert vatd_fp["baseline_method"] == "baseline"
    assert vatd_fp["recall_verdict_vs_baseline"] == "win"
    assert vatd_fp["recall_win_vs_baseline"] == "True"
    assert result.summary["verdict_summary"]["vatd"]["win"] == 2
    assert result.summary["verdict_summary"]["vatd"]["tie"] == 1
    assert result.summary["verdict_summary"]["vatd"]["by_budget_type"]["fp_limit"]["win"] == 1
    assert result.summary["fixed_budget_verdict_summary"]["vatd"]["win"] == 2
    assert result.summary["fixed_budget_verdict_summary"]["vatd"]["tie"] == 0
    assert "best_f1" not in result.summary["fixed_budget_verdict_summary"]["vatd"]["by_budget_type"]
    assert result.summary["fixed_budget_verdict_summary"]["vatd"]["by_budget_type"]["best_under_budget"]["win"] == 1
    claim_budget_types = {row["budget_type"] for row in result.summary["paper_claim_rows"]}
    assert claim_budget_types == {"best_under_budget", "fp_limit"}
    assert all(row["method"] == "vatd" for row in result.summary["paper_claim_rows"])
    assert not any(row["budget_type"] == "best_f1" for row in result.summary["paper_claim_rows"])
    claim_csv = Path(result.summary["paper_claim_rows_csv"])
    claim_md = Path(result.summary["paper_claim_rows_markdown"])
    claim_rows = list(csv.DictReader(claim_csv.open("r", encoding="utf-8")))
    assert claim_csv.exists()
    assert claim_md.exists()
    assert {row["budget_type"] for row in claim_rows} == {"best_under_budget", "fp_limit"}
    assert not any(row["budget_type"] == "best_f1" for row in claim_rows)
    assert "| vatd | baseline | fp_limit | 0 | win |" in claim_md.read_text(encoding="utf-8")
    wins_csv = Path(result.summary["paper_claim_wins_csv"])
    wins_md = Path(result.summary["paper_claim_wins_markdown"])
    win_rows = list(csv.DictReader(wins_csv.open("r", encoding="utf-8")))
    assert wins_csv.exists()
    assert wins_md.exists()
    assert len(result.summary["paper_claim_wins"]) == 2
    assert len(win_rows) == 2
    assert {row["budget_type"] for row in win_rows} == {"best_under_budget", "fp_limit"}
    assert all(row["verdict"] == "win" for row in win_rows)
    assert "| vatd | baseline | fp_limit | 0 | 1.0 | 0.5 |" in wins_md.read_text(encoding="utf-8")
    fixed_report = Path(result.summary["fixed_budget_report_markdown"])
    fixed_report_text = fixed_report.read_text(encoding="utf-8")
    assert fixed_report.exists()
    assert "- Baseline method: baseline" in fixed_report_text
    assert "- Claim gate: pass (at least one fixed-budget recall win over the baseline)" in fixed_report_text
    assert "- Fixed-budget wins: 2" in fixed_report_text
    assert "| vatd | 2 | 0 | 0 |" in fixed_report_text
    paper_summary = result.summary["paper_result_summary"]
    paper_summary_json = Path(result.summary["paper_result_summary_json"])
    loaded_paper_summary = json.loads(paper_summary_json.read_text(encoding="utf-8"))
    assert paper_summary_json.exists()
    assert paper_summary["baseline_method"] == "baseline"
    assert paper_summary["compared_methods"] == ["vatd"]
    assert paper_summary["claim_gate"]["status"] == "pass"
    assert paper_summary["claim_gate"]["requires"] == "fixed-budget recall win over baseline at best_under_budget, fp_limit, or max_fppi"
    assert paper_summary["fixed_budget_wins"] == 2
    assert paper_summary["best_fixed_budget_win"]["method"] == "vatd"
    assert paper_summary["best_fixed_budget_win"]["delta_recall_vs_baseline"] == 0.5
    assert loaded_paper_summary["fixed_budget_wins"] == 2
    claim_gate_json = Path(result.summary["claim_gate_json"])
    loaded_claim_gate = json.loads(claim_gate_json.read_text(encoding="utf-8"))
    assert claim_gate_json.exists()
    assert result.summary["claim_gate"]["claim_gate"]["status"] == "pass"
    assert loaded_claim_gate["claim_gate"]["status"] == "pass"
    assert loaded_claim_gate["fixed_budget_wins"] == 2
    assert loaded_claim_gate["best_fixed_budget_win"]["method"] == "vatd"
    assert loaded_claim_gate["paper_result_summary_json"] == str(paper_summary_json)
    assert "| baseline | best_f1 | best_f1 |" in result.markdown_path.read_text(encoding="utf-8")
    assert "| vatd | fp_limit | 0 | True |" in result.markdown_path.read_text(encoding="utf-8")
