from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qstr_dronedet.candidates.merge import bbox_iou, center_distance
from qstr_dronedet.tracking.tracklet_classifier import TRACKLET_FEATURES, _features, _load_gt_csv, _prob


@dataclass
class ProposalTrackletDatasetResult:
    csv_path: Path
    json_path: Path
    summary: dict[str, Any]


@dataclass
class MergedTrackletJsonlResult:
    json_path: Path
    summary: dict[str, Any]


@dataclass
class ProposalInputValidationResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class RouteBProposalRunManifestResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class RouteBProposalInputScanResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class YoloRouteBExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class FrameListExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class TrackletPredictionExportResult:
    out_path: Path
    summary: dict[str, Any]


@dataclass
class FlatPredictionEvaluationResult:
    csv_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass
class FlatPredictionNmsSweepResult:
    csv_path: Path
    summary_path: Path
    summary: dict[str, Any]


@dataclass
class FlatPredictionEvalComparisonResult:
    csv_path: Path
    markdown_path: Path
    summary_path: Path
    summary: dict[str, Any]


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _bbox(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in row.get("bbox", [0.0, 0.0, 0.0, 0.0]))


def _safe_seq_frame_from_image(path: Path) -> tuple[str, int]:
    stem = path.stem
    if stem.isdigit():
        seq_path = path.parent.parent if path.parent.name.lower() in {"frames", "images", "images2"} else path.parent
        return seq_path.name, int(stem)
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            return prefix, int(suffix)
    return path.parent.name, 0


def _yolo_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() in {"images", "images2"}:
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def _read_image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
            return int(img.shape[1]), int(img.shape[0])
    except Exception:
        pass
    raise ValueError(f"could not read image size: {path}")


def _read_yolo_label_rows(
    image_path: Path,
    image_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    label_path = _yolo_label_path(image_path)
    if not label_path.exists():
        return []
    width, height = image_size or _read_image_size(image_path)
    seq, frame_id = _safe_seq_frame_from_image(image_path)
    rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8-sig") as f:
        for label_index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height
            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(width), cx + bw / 2.0)
            y2 = min(float(height), cy + bh / 2.0)
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "bbox": [x1, y1, x2, y2],
                    "objectness": 1.0,
                    "final_drone_score": 1.0,
                    "source": "oracle_yolo_label",
                    "class_id": cls,
                    "label_index": label_index,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "visible": True,
                    "image_width": width,
                    "image_height": height,
                }
            )
    return rows


def _read_yolo_prediction_rows(
    image_path: Path,
    pred_label_dir: str | Path,
    image_size: tuple[int, int] | None,
    source: str,
) -> list[dict[str, Any]]:
    label_path = Path(pred_label_dir) / f"{image_path.stem}.txt"
    if not label_path.exists():
        return []
    width, height = image_size or _read_image_size(image_path)
    seq, frame_id = _safe_seq_frame_from_image(image_path)
    rows: list[dict[str, Any]] = []
    with label_path.open("r", encoding="utf-8-sig") as f:
        for pred_index, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            bw = float(parts[3]) * width
            bh = float(parts[4]) * height
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            x1 = max(0.0, cx - bw / 2.0)
            y1 = max(0.0, cy - bh / 2.0)
            x2 = min(float(width), cx + bw / 2.0)
            y2 = min(float(height), cy + bh / 2.0)
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "bbox": [x1, y1, x2, y2],
                    "objectness": conf,
                    "final_drone_score": conf,
                    "source": source,
                    "class_id": cls,
                    "prediction_index": pred_index,
                    "pred_label_path": str(label_path),
                    "image_path": str(image_path),
                    "visible": True,
                    "image_width": width,
                    "image_height": height,
                    "predicted_class": "drone",
                    "final_probs": {"drone": conf, "background": max(0.0, 1.0 - conf)},
                }
            )
    return rows


def _read_image_list_files(list_files: list[str | Path], max_images: int | None = None) -> list[Path]:
    image_paths: list[Path] = []
    for list_file in list_files:
        base = Path(list_file).parent
        with Path(list_file).open("r", encoding="utf-8-sig") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                path = Path(text)
                if not path.is_absolute():
                    path = base / path
                image_paths.append(path)
                if max_images is not None and len(image_paths) >= max_images:
                    return image_paths
    return image_paths


def _gt_sequences(gt_csv: str | Path) -> set[str]:
    sequences: set[str] = set()
    with Path(gt_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = str(row.get("seq") or "").strip()
            if not seq:
                video_path = str(row.get("video_path") or "").strip()
                if video_path:
                    seq = Path(video_path).parent.name or Path(video_path).stem
            if seq:
                sequences.add(seq)
    return sequences


def export_frame_list_from_gt_csv(
    gt_csv: str | Path,
    frame_root: str | Path,
    out: str | Path,
    extensions: list[str] | None = None,
    recursive: bool = False,
    max_frames: int | None = None,
    max_frames_per_seq: int | None = None,
) -> FrameListExportResult:
    frame_root_obj = Path(frame_root)
    if not frame_root_obj.exists():
        raise FileNotFoundError(frame_root_obj)
    sequences = _gt_sequences(gt_csv)
    if not sequences:
        raise ValueError("GT CSV has no usable seq/video_path values")
    exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in (extensions or [".png", ".jpg", ".jpeg"])}
    iterator = frame_root_obj.rglob("*") if recursive else frame_root_obj.glob("*")
    selected: list[Path] = []
    scanned_files = 0
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        scanned_files += 1
        seq, _frame_id = _safe_seq_frame_from_image(path)
        parent_seq = path.parent.name
        if seq in sequences or parent_seq in sequences:
            selected.append(path.resolve())
    selected = sorted(selected, key=lambda p: (_safe_seq_frame_from_image(p)[0], _safe_seq_frame_from_image(p)[1], str(p)))
    if max_frames_per_seq is not None and max_frames_per_seq > 0:
        capped: list[Path] = []
        counts: dict[str, int] = {}
        for path in selected:
            seq, _frame_id = _safe_seq_frame_from_image(path)
            if seq not in sequences and path.parent.name in sequences:
                seq = path.parent.name
            if counts.get(seq, 0) >= max_frames_per_seq:
                continue
            counts[seq] = counts.get(seq, 0) + 1
            capped.append(path)
        selected = capped
    if max_frames is not None and max_frames > 0:
        selected = selected[:max_frames]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(str(path) for path in selected) + ("\n" if selected else ""), encoding="utf-8")
    by_seq: dict[str, int] = {}
    for path in selected:
        seq, _frame_id = _safe_seq_frame_from_image(path)
        if seq not in sequences and path.parent.name in sequences:
            seq = path.parent.name
        by_seq[seq] = by_seq.get(seq, 0) + 1
    summary = {
        "gt_csv": str(gt_csv),
        "frame_root": str(frame_root_obj),
        "out": str(out_path),
        "recursive": recursive,
        "extensions": sorted(exts),
        "gt_sequences": len(sequences),
        "matched_sequences": len(by_seq),
        "scanned_files": scanned_files,
        "frames": len(selected),
        "max_frames": max_frames,
        "max_frames_per_seq": max_frames_per_seq,
        "frames_by_seq": by_seq,
    }
    (out_path.parent / f"{out_path.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return FrameListExportResult(out_path=out_path, summary=summary)


def _temporal_saliency_rows_for_sequence(
    image_paths: list[Path],
    threshold: float,
    min_area: float,
    max_area: float,
    dilate_iters: int,
    source: str,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for temporal saliency proposal export") from exc

    rows: list[dict[str, Any]] = []
    prev_gray = None
    for image_path in sorted(image_paths, key=lambda p: _safe_seq_frame_from_image(p)[1]):
        seq, frame_id = _safe_seq_frame_from_image(image_path)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            prev_gray = None
            continue
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue
        diff = cv2.absdiff(gray, prev_gray)
        if threshold <= 0:
            _thr, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _thr, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        if dilate_iters > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iters)
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour_index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            x1 = float(max(0, x))
            y1 = float(max(0, y))
            x2 = float(min(width, x + w))
            y2 = float(min(height, y + h))
            roi = diff[int(y1) : int(y2), int(x1) : int(x2)]
            objectness = float(np.clip((float(np.mean(roi)) if roi.size else 0.0) / 255.0, 0.0, 1.0))
            objectness = max(objectness, float(np.clip(area / max(1.0, min_area * 4.0), 0.0, 1.0)) * 0.25)
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "bbox": [x1, y1, x2, y2],
                    "objectness": objectness,
                    "final_drone_score": objectness,
                    "source": source,
                    "saliency_area": area,
                    "saliency_mean_absdiff": float(np.mean(roi)) if roi.size else 0.0,
                    "saliency_contour_index": contour_index,
                    "image_path": str(image_path),
                    "frame_path": str(image_path),
                    "visible": True,
                    "image_width": int(width),
                    "image_height": int(height),
                    "predicted_class": "motion_candidate",
                    "final_probs": {"drone": objectness, "background": max(0.0, 1.0 - objectness)},
                }
            )
        prev_gray = gray
    return rows


def _box_side(row: dict[str, Any]) -> float:
    x1, y1, x2, y2 = _bbox(row)
    return max(0.0, max(x2 - x1, y2 - y1))


def _row_score(row: dict[str, Any]) -> float:
    return max(float(row.get("objectness", 0.0)), float(row.get("final_drone_score", 0.0)))


def _source_has_detector(row: dict[str, Any]) -> bool:
    source = str(row.get("source", ""))
    return any(token in source for token in ("yolo", "fallback", "motion", "seed", "oracle"))


def _artifact_score(row: dict[str, Any]) -> float:
    return max(
        _prob(row, "crop_probs", "alignment_artifact"),
        _prob(row, "feature_probs", "alignment_artifact"),
        _prob(row, "temporal_probs", "alignment_artifact"),
        _prob(row, "final_probs", "alignment_artifact"),
    )


def _load_run_diagnostics(
    run_roots: list[str | Path],
    profile: str,
    diagnostics_name: str,
    max_frames: int | None,
) -> dict[str, list[dict[str, Any]]]:
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for run_root in run_roots:
        profile_root = Path(run_root) / profile
        for seq_dir in sorted(p for p in profile_root.glob("*") if p.is_dir()):
            diag_path = seq_dir / diagnostics_name
            if not diag_path.exists() and diagnostics_name == "diagnostics_raw.jsonl":
                diag_path = seq_dir / "diagnostics.jsonl"
            if not diag_path.exists():
                continue
            seq = seq_dir.name
            for row in _load_jsonl(diag_path):
                frame_id = int(row.get("frame_id", -1))
                if max_frames is not None and frame_id >= max_frames:
                    continue
                if "bbox" not in row:
                    continue
                item = dict(row)
                item["seq"] = seq
                by_seq.setdefault(seq, []).append(item)
    return by_seq


def _inspect_proposal_source(
    run_root: str | Path,
    gt_csv: str | Path,
    name: str,
    profile: str,
    diagnostics_name: str,
    max_frames: int | None,
    min_bbox_rows: int,
) -> dict[str, Any]:
    run_root_obj = Path(run_root)
    gt_obj = Path(gt_csv)
    summary: dict[str, Any] = {
        "name": name,
        "run_root": str(run_root_obj),
        "gt_csv": str(gt_obj),
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "exists": run_root_obj.exists(),
        "profile_exists": (run_root_obj / profile).exists(),
        "gt_exists": gt_obj.exists(),
        "sequences": 0,
        "diagnostics_files": 0,
        "diagnostic_rows": 0,
        "bbox_rows": 0,
        "gt_frames": 0,
        "gt_sequences": 0,
        "matched_gt_sequences": 0,
        "missing_gt_sequences": [],
        "diagnostic_sequences_without_gt": [],
        "issues": [],
        "warnings": [],
    }
    if not run_root_obj.exists():
        summary["issues"].append("run root does not exist")
    if not gt_obj.exists():
        summary["issues"].append("GT CSV does not exist")
    profile_root = run_root_obj / profile
    if run_root_obj.exists() and not profile_root.exists():
        summary["issues"].append(f"profile directory does not exist: {profile}")
    if summary["issues"]:
        return summary

    try:
        gt_by_key = _load_gt_csv(gt_obj, max_frames=max_frames)
    except Exception as exc:
        summary["issues"].append(f"failed to parse GT CSV: {exc}")
        return summary

    gt_sequences = {seq for seq, _frame_id in gt_by_key}
    summary["gt_frames"] = len(gt_by_key)
    summary["gt_sequences"] = len(gt_sequences)
    if not gt_by_key:
        summary["issues"].append("GT CSV has zero usable frames")

    diagnostic_sequences: set[str] = set()
    seq_dirs = sorted(p for p in profile_root.glob("*") if p.is_dir())
    summary["sequences"] = len(seq_dirs)
    if not seq_dirs:
        summary["issues"].append("profile directory has no sequence subdirectories")

    for seq_dir in seq_dirs:
        diag_path = seq_dir / diagnostics_name
        if not diag_path.exists() and diagnostics_name == "diagnostics_raw.jsonl":
            diag_path = seq_dir / "diagnostics.jsonl"
        if not diag_path.exists():
            summary["warnings"].append(f"{seq_dir.name}: diagnostics file missing")
            continue
        summary["diagnostics_files"] += 1
        seq = seq_dir.name
        diagnostic_sequences.add(seq)
        try:
            rows = _load_jsonl(diag_path)
        except Exception as exc:
            summary["issues"].append(f"{seq}: failed to parse diagnostics: {exc}")
            continue
        row_count = 0
        bbox_count = 0
        for row in rows:
            frame_id = int(row.get("frame_id", -1))
            if max_frames is not None and frame_id >= max_frames:
                continue
            row_count += 1
            if "bbox" in row:
                bbox_count += 1
        summary["diagnostic_rows"] += row_count
        summary["bbox_rows"] += bbox_count
        if bbox_count == 0:
            summary["warnings"].append(f"{seq}: diagnostics has zero bbox rows")

    summary["matched_gt_sequences"] = len(diagnostic_sequences & gt_sequences)
    summary["missing_gt_sequences"] = sorted(gt_sequences - diagnostic_sequences)
    summary["diagnostic_sequences_without_gt"] = sorted(diagnostic_sequences - gt_sequences)
    if summary["diagnostics_files"] == 0:
        summary["issues"].append("no diagnostics files found")
    if summary["bbox_rows"] < min_bbox_rows:
        summary["issues"].append(f"only {summary['bbox_rows']} bbox rows found, expected at least {min_bbox_rows}")
    if gt_sequences and summary["matched_gt_sequences"] == 0:
        summary["issues"].append("no sequence names overlap between diagnostics and GT CSV")
    if summary["missing_gt_sequences"]:
        summary["warnings"].append("some GT sequences have no diagnostics")
    if summary["diagnostic_sequences_without_gt"]:
        summary["warnings"].append("some diagnostic sequences have no GT rows")
    return summary


def validate_route_b_proposal_inputs(
    train_run_roots: list[str | Path],
    train_gt_csvs: list[str | Path],
    eval_run_roots: list[str | Path],
    eval_gt_csvs: list[str | Path],
    out: str | Path,
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    max_frames: int | None = None,
    min_bbox_rows: int = 1,
) -> ProposalInputValidationResult:
    issues: list[str] = []
    warnings: list[str] = []
    if not train_run_roots:
        issues.append("at least one train run root is required")
    if not eval_run_roots:
        issues.append("at least one eval run root is required")
    if len(train_run_roots) != len(train_gt_csvs):
        issues.append("train_run_roots and train_gt_csvs must have the same length")
    if len(eval_run_roots) != len(eval_gt_csvs):
        issues.append("eval_run_roots and eval_gt_csvs must have the same length")
    if train_source_names is not None and len(train_source_names) != len(train_run_roots):
        issues.append("train_source_names must have the same length as train_run_roots")
    if eval_dataset_names is not None and len(eval_dataset_names) != len(eval_run_roots):
        issues.append("eval_dataset_names must have the same length as eval_run_roots")
    if min_bbox_rows < 0:
        issues.append("min_bbox_rows must be nonnegative")

    train_summaries: list[dict[str, Any]] = []
    eval_summaries: list[dict[str, Any]] = []
    if not issues:
        for index, (run_root, gt_csv) in enumerate(zip(train_run_roots, train_gt_csvs)):
            name = train_source_names[index] if train_source_names is not None else Path(run_root).name
            summary = _inspect_proposal_source(run_root, gt_csv, str(name), profile, diagnostics_name, max_frames, min_bbox_rows)
            train_summaries.append(summary)
            issues.extend(f"train/{name}: {issue}" for issue in summary["issues"])
            warnings.extend(f"train/{name}: {warning}" for warning in summary["warnings"])
        for index, (run_root, gt_csv) in enumerate(zip(eval_run_roots, eval_gt_csvs)):
            name = eval_dataset_names[index] if eval_dataset_names is not None else Path(run_root).name
            summary = _inspect_proposal_source(run_root, gt_csv, str(name), profile, diagnostics_name, max_frames, min_bbox_rows)
            eval_summaries.append(summary)
            issues.extend(f"eval/{name}: {issue}" for issue in summary["issues"])
            warnings.extend(f"eval/{name}: {warning}" for warning in summary["warnings"])

    train_bbox_rows = sum(int(s.get("bbox_rows", 0)) for s in train_summaries)
    eval_bbox_rows = sum(int(s.get("bbox_rows", 0)) for s in eval_summaries)
    summary = {
        "valid": not issues,
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "max_frames": max_frames,
        "min_bbox_rows": min_bbox_rows,
        "train": train_summaries,
        "eval": eval_summaries,
        "train_bbox_rows": train_bbox_rows,
        "eval_bbox_rows": eval_bbox_rows,
        "issues": issues,
        "warnings": warnings,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ProposalInputValidationResult(out_path=out_path, summary=summary)


def _ps_quote(value: str | Path) -> str:
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _ps_array(values: list[str | Path]) -> str:
    return "@(" + ", ".join(_ps_quote(value) for value in values) + ")"


def _compact_args(args: list[str]) -> str:
    return " ".join(_ps_quote(arg) if any(ch.isspace() for ch in arg) else arg for arg in args)


def write_route_b_proposal_run_manifest(
    out_dir: str | Path,
    train_run_roots: list[str | Path],
    train_gt_csvs: list[str | Path],
    eval_run_roots: list[str | Path],
    eval_gt_csvs: list[str | Path],
    train_source_names: list[str] | None = None,
    eval_dataset_names: list[str] | None = None,
    run_id: str = "route_b_proposal_benchmark",
    benchmark_out_dir: str | Path | None = None,
    runner_output_root: str | Path | None = None,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    max_frames: int | None = None,
    past_len: int = 8,
    future_len: int = 8,
    model_types: list[str] | None = None,
    epochs: int = 50,
    hidden: int = 128,
    batch_size: int = 64,
    thresholds: list[float] | None = None,
    balance_by: list[str] | None = None,
    baseline_csv: str | Path | None = None,
    baseline_metric: str = "best_f1",
    baseline_lower_is_better: bool = False,
    validate_inputs: bool = True,
    preflight_min_bbox_rows: int = 1,
) -> RouteBProposalRunManifestResult:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    benchmark_out = Path(benchmark_out_dir) if benchmark_out_dir is not None else out_root / "run"
    runner_root = Path(runner_output_root) if runner_output_root is not None else out_root / "runner"
    preflight_json = out_root / "proposal_preflight.json"

    validation_summary = None
    if validate_inputs:
        validation = validate_route_b_proposal_inputs(
            train_run_roots,
            train_gt_csvs,
            eval_run_roots,
            eval_gt_csvs,
            preflight_json,
            train_source_names=train_source_names,
            eval_dataset_names=eval_dataset_names,
            profile=profile,
            diagnostics_name=diagnostics_name,
            max_frames=max_frames,
            min_bbox_rows=preflight_min_bbox_rows,
        )
        validation_summary = validation.summary

    model_types = model_types or ["mlp", "diffusion"]
    thresholds = thresholds or []
    balance_by = balance_by if balance_by is not None else ["dataset_source"]
    train_source_names = train_source_names or []
    eval_dataset_names = eval_dataset_names or []

    start_parts = [
        ".\\tools\\start_route_b_proposal_benchmark_detached.ps1",
        "-TrainRunRoots",
        _ps_array(train_run_roots),
        "-TrainGtCsvs",
        _ps_array(train_gt_csvs),
        "-EvalRunRoots",
        _ps_array(eval_run_roots),
        "-EvalGtCsvs",
        _ps_array(eval_gt_csvs),
        "-OutDir",
        _ps_quote(benchmark_out),
        "-OutputRoot",
        _ps_quote(runner_root),
        "-RunId",
        _ps_quote(run_id),
        "-Profile",
        _ps_quote(profile),
        "-DiagnosticsName",
        _ps_quote(diagnostics_name),
        "-PastLen",
        str(past_len),
        "-FutureLen",
        str(future_len),
        "-ModelTypes",
        _ps_array(model_types),
        "-Epochs",
        str(epochs),
        "-Hidden",
        str(hidden),
        "-BatchSize",
        str(batch_size),
        "-PreflightOut",
        _ps_quote(preflight_json),
        "-PreflightMinBBoxRows",
        str(preflight_min_bbox_rows),
    ]
    if max_frames is not None:
        start_parts.extend(["-MaxFrames", str(max_frames)])
    if train_source_names:
        start_parts.extend(["-TrainSourceNames", _ps_array(train_source_names)])
    if eval_dataset_names:
        start_parts.extend(["-EvalDatasetNames", _ps_array(eval_dataset_names)])
    if thresholds:
        start_parts.extend(["-Thresholds", "@(" + ", ".join(str(value) for value in thresholds) + ")"])
    if balance_by:
        start_parts.extend(["-BalanceBy", _ps_array(balance_by)])
    if baseline_csv is not None:
        start_parts.extend(["-BaselineCsv", _ps_quote(baseline_csv), "-BaselineMetric", _ps_quote(baseline_metric)])
        if baseline_lower_is_better:
            start_parts.append("-BaselineLowerIsBetter")
    start_command = " ".join(start_parts)

    preflight_args = [
        "python",
        "-m",
        "qstr_dronedet.cli",
        "validate-route-b-proposal-inputs",
        "--train-run-roots",
        *[str(path) for path in train_run_roots],
        "--train-gt-csvs",
        *[str(path) for path in train_gt_csvs],
        "--eval-run-roots",
        *[str(path) for path in eval_run_roots],
        "--eval-gt-csvs",
        *[str(path) for path in eval_gt_csvs],
        "--out",
        str(preflight_json),
        "--profile",
        profile,
        "--diagnostics-name",
        diagnostics_name,
        "--min-bbox-rows",
        str(preflight_min_bbox_rows),
        "--strict",
    ]
    if max_frames is not None:
        preflight_args.extend(["--max-frames", str(max_frames)])
    if train_source_names:
        preflight_args.extend(["--train-source-names", *train_source_names])
    if eval_dataset_names:
        preflight_args.extend(["--eval-dataset-names", *eval_dataset_names])
    monitor_command = (
        ".\\tools\\monitor_route_b_proposal_benchmark.ps1 "
        + "-OutputRoot "
        + _ps_quote(runner_root)
        + " -RunId "
        + _ps_quote(run_id)
    )
    manifest = {
        "kind": "route_b_proposal_policy_benchmark_manifest",
        "run_id": run_id,
        "out_dir": str(out_root),
        "benchmark_out_dir": str(benchmark_out),
        "runner_output_root": str(runner_root),
        "preflight_json": str(preflight_json),
        "train_run_roots": [str(path) for path in train_run_roots],
        "train_gt_csvs": [str(path) for path in train_gt_csvs],
        "eval_run_roots": [str(path) for path in eval_run_roots],
        "eval_gt_csvs": [str(path) for path in eval_gt_csvs],
        "train_source_names": train_source_names,
        "eval_dataset_names": eval_dataset_names,
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "max_frames": max_frames,
        "past_len": past_len,
        "future_len": future_len,
        "model_types": model_types,
        "epochs": epochs,
        "hidden": hidden,
        "batch_size": batch_size,
        "thresholds": thresholds,
        "balance_by": balance_by,
        "baseline_csv": str(baseline_csv) if baseline_csv is not None else None,
        "baseline_metric": baseline_metric,
        "baseline_lower_is_better": baseline_lower_is_better,
        "preflight_min_bbox_rows": preflight_min_bbox_rows,
        "preflight_valid": None if validation_summary is None else bool(validation_summary.get("valid")),
        "preflight_summary": validation_summary,
        "commands": {
            "preflight": _compact_args(preflight_args),
            "start_detached": start_command,
            "monitor": monitor_command,
        },
    }
    manifest_path = out_root / "route_b_proposal_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "start_route_b_proposal_benchmark.ps1").write_text(start_command + "\n", encoding="utf-8")
    (out_root / "monitor_route_b_proposal_benchmark.ps1").write_text(monitor_command + "\n", encoding="utf-8")
    (out_root / "preflight_route_b_proposal_benchmark.txt").write_text(manifest["commands"]["preflight"] + "\n", encoding="utf-8")
    return RouteBProposalRunManifestResult(out_path=manifest_path, summary=manifest)


def _iter_files_limited(root: Path, max_depth: int, max_files: int, candidate_names: set[str]) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    root = root.resolve()
    for dirpath, _dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        try:
            depth = len(current.resolve().relative_to(root).parts)
        except ValueError:
            continue
        if depth > max_depth:
            _dirnames[:] = []
            continue
        for filename in filenames:
            if filename not in candidate_names and not filename.lower().endswith(".csv"):
                continue
            out.append(current / filename)
            if len(out) >= max_files:
                return out
    return out


def _looks_like_gt_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
    except Exception:
        return False
    return {"frame_id", "x1", "y1", "x2", "y2"}.issubset(columns) and ("video_path" in columns or "seq" in columns)


def _safe_gt_sequences(path: Path, max_frames: int | None) -> set[str]:
    try:
        gt = _load_gt_csv(path, max_frames=max_frames)
        return {seq for seq, _frame_id in gt}
    except Exception:
        return set()


def _count_text_lines(path: Path, max_lines: int = 100000) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
            for count, _line in enumerate(f, start=1):
                if count >= max_lines:
                    return count
    except Exception:
        return 0
    return count


def _scan_yolo_dataset_candidates(files: list[Path]) -> list[dict[str, Any]]:
    by_root: dict[Path, dict[str, Any]] = {}
    for path in files:
        name = path.name.lower()
        parent = path.parent
        root: Path | None = None
        role = ""
        if name in {"train.txt", "val.txt", "test.txt", "train2.txt", "val2.txt", "test2.txt"}:
            root = parent
            role = name
        elif path.suffix.lower() in {".yaml", ".yml"}:
            root = parent
            role = "yaml"
        if root is None:
            continue
        item = by_root.setdefault(
            root,
            {
                "root": str(root),
                "has_images_dir": (root / "images").exists() or (root / "images2").exists(),
                "has_labels_dir": (root / "labels").exists(),
                "list_files": [],
                "yaml_files": [],
                "total_list_rows": 0,
            },
        )
        if role == "yaml":
            item["yaml_files"].append(str(path))
        else:
            rows = _count_text_lines(path)
            item["list_files"].append({"path": str(path), "name": name, "rows": rows})
            item["total_list_rows"] += rows
    candidates = [
        item
        for item in by_root.values()
        if item["has_images_dir"] and (item["has_labels_dir"] or item["list_files"] or item["yaml_files"])
    ]
    return sorted(candidates, key=lambda row: int(row["total_list_rows"]), reverse=True)


def _sample_bbox_rows(diag_paths: list[Path], max_files: int, max_rows_per_file: int) -> tuple[int, int]:
    sampled_files = 0
    bbox_rows = 0
    for path in diag_paths[:max_files]:
        sampled_files += 1
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                for index, line in enumerate(f):
                    if index >= max_rows_per_file:
                        break
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and "bbox" in row:
                        bbox_rows += 1
        except Exception:
            continue
    return sampled_files, bbox_rows


def scan_route_b_proposal_inputs(
    scan_roots: list[str | Path],
    out: str | Path,
    profiles: list[str] | None = None,
    diagnostics_names: list[str] | None = None,
    max_depth: int = 8,
    max_files: int = 20000,
    max_diag_sample_files: int = 20,
    max_rows_per_diag_file: int = 200,
    max_frames: int | None = None,
) -> RouteBProposalInputScanResult:
    profiles = profiles or ["hard_recovery"]
    diagnostics_names = diagnostics_names or ["diagnostics_raw.jsonl", "diagnostics.jsonl"]
    profile_set = set(profiles)
    diagnostics_set = set(diagnostics_names)
    issues: list[str] = []
    files: list[Path] = []
    candidate_names = set(diagnostics_names) | {
        "train.txt",
        "val.txt",
        "test.txt",
        "train2.txt",
        "val2.txt",
        "test2.txt",
    }
    for root_like in scan_roots:
        root = Path(root_like)
        if not root.exists():
            issues.append(f"scan root does not exist: {root}")
            continue
        files.extend(_iter_files_limited(root, max_depth=max_depth, max_files=max_files, candidate_names=candidate_names))
        if len(files) >= max_files:
            issues.append(f"stopped after max_files={max_files} candidate files")
            files = files[:max_files]
            break

    yolo_dataset_candidates = _scan_yolo_dataset_candidates(files)
    run_map: dict[tuple[str, str], dict[str, Any]] = {}
    gt_candidates: list[dict[str, Any]] = []
    for path in files:
        name = path.name
        if name in diagnostics_set and path.parent.parent.name in profile_set:
            profile_dir = path.parent.parent
            run_root = profile_dir.parent
            profile = profile_dir.name
            key = (str(run_root), profile)
            item = run_map.setdefault(
                key,
                {
                    "run_root": str(run_root),
                    "profile": profile,
                    "diagnostics_name": name,
                    "diagnostics_files": 0,
                    "sequences": set(),
                    "diagnostic_paths": [],
                },
            )
            item["diagnostics_files"] += 1
            item["sequences"].add(path.parent.name)
            item["diagnostic_paths"].append(path)
        elif path.suffix.lower() == ".csv" and _looks_like_gt_csv(path):
            seqs = _safe_gt_sequences(path, max_frames=max_frames)
            gt_candidates.append(
                {
                    "path": str(path),
                    "gt_sequences": len(seqs),
                    "sequences": sorted(seqs),
                }
            )

    run_candidates: list[dict[str, Any]] = []
    for item in run_map.values():
        seqs = sorted(item["sequences"])
        sampled_files, sampled_bbox_rows = _sample_bbox_rows(
            item["diagnostic_paths"],
            max_files=max_diag_sample_files,
            max_rows_per_file=max_rows_per_diag_file,
        )
        best_gt = None
        best_overlap = -1
        seq_set = set(seqs)
        for gt in gt_candidates:
            gt_seq_set = set(gt["sequences"])
            overlap = len(seq_set & gt_seq_set)
            if overlap > best_overlap:
                best_gt = gt
                best_overlap = overlap
        run_candidates.append(
            {
                "run_root": item["run_root"],
                "profile": item["profile"],
                "diagnostics_name": item["diagnostics_name"],
                "diagnostics_files": int(item["diagnostics_files"]),
                "sequences": len(seqs),
                "sequence_names": seqs[:50],
                "sampled_diagnostics_files": sampled_files,
                "sampled_bbox_rows": sampled_bbox_rows,
                "best_gt_csv": None if best_gt is None else best_gt["path"],
                "best_gt_sequence_overlap": max(0, best_overlap),
                "best_gt_sequences": 0 if best_gt is None else int(best_gt["gt_sequences"]),
            }
        )

    run_candidates = sorted(
        run_candidates,
        key=lambda row: (int(row["best_gt_sequence_overlap"]), int(row["sampled_bbox_rows"]), int(row["diagnostics_files"])),
        reverse=True,
    )
    summary = {
        "scan_roots": [str(path) for path in scan_roots],
        "profiles": profiles,
        "diagnostics_names": diagnostics_names,
        "max_depth": max_depth,
        "max_files": max_files,
        "candidate_files_scanned": len(files),
        "files_scanned": len(files),
        "run_candidates": run_candidates,
        "gt_candidates": gt_candidates,
        "yolo_dataset_candidates": yolo_dataset_candidates,
        "num_run_candidates": len(run_candidates),
        "num_gt_candidates": len(gt_candidates),
        "num_yolo_dataset_candidates": len(yolo_dataset_candidates),
        "issues": issues,
        "suggested_manifest_inputs": [
            {
                "run_root": row["run_root"],
                "gt_csv": row["best_gt_csv"],
                "name": Path(row["run_root"]).name,
                "profile": row["profile"],
                "diagnostics_name": row["diagnostics_name"],
                "overlap": row["best_gt_sequence_overlap"],
            }
            for row in run_candidates
            if row["best_gt_csv"] and int(row["best_gt_sequence_overlap"]) > 0 and int(row["sampled_bbox_rows"]) > 0
        ],
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RouteBProposalInputScanResult(out_path=out_path, summary=summary)


def export_yolo_oracle_tracklets(
    list_files: list[str | Path],
    out: str | Path,
    dataset_source: str = "yolo_oracle",
    image_size: tuple[int, int] | None = None,
    max_images: int | None = None,
    skip_images: int = 0,
    max_labeled_images_per_seq: int | None = None,
    max_gap: int = 3,
    base_radius: float = 18.0,
    radius_per_side: float = 0.75,
    min_iou: float = 0.05,
    min_tracklet_rows: int = 2,
) -> ProposalTrackletDatasetResult:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_seq: dict[str, list[dict[str, Any]]] = {}
    images_seen = 0
    images_skipped = 0
    labels_seen = 0
    labeled_images_by_seq: dict[str, int] = {}
    for list_file in list_files:
        with Path(list_file).open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if images_skipped < skip_images:
                    images_skipped += 1
                    continue
                if max_images is not None and images_seen >= max_images:
                    break
                image_path = Path(text)
                if not image_path.exists():
                    continue
                seq, _ = _safe_seq_frame_from_image(image_path)
                if max_labeled_images_per_seq is not None and labeled_images_by_seq.get(seq, 0) >= max_labeled_images_per_seq:
                    continue
                label_rows = _read_yolo_label_rows(image_path, image_size=image_size)
                if max_labeled_images_per_seq is not None and not label_rows:
                    continue
                images_seen += 1
                labels_seen += len(label_rows)
                if label_rows:
                    labeled_images_by_seq[seq] = labeled_images_by_seq.get(seq, 0) + 1
                for row in label_rows:
                    rows_by_seq.setdefault(str(row["seq"]), []).append(row)
                if images_seen == 1 or images_seen % 5000 == 0:
                    print(
                        json.dumps(
                            {
                                "kind": "export_yolo_oracle_tracklets_progress",
                                "dataset_source": dataset_source,
                                "images_seen": images_seen,
                                "labels_seen": labels_seen,
                                "sequences": len(rows_by_seq),
                            }
                        ),
                        flush=True,
                    )
        if max_images is not None and images_seen >= max_images:
            break

    csv_path = out_dir / "oracle_tracklets.csv"
    json_path = out_dir / "oracle_tracklets.jsonl"
    fields = ["seq", "track_id", "label", "bucket", "best_iou", "matched_frames", "num_rows_raw", "dataset_source"] + TRACKLET_FEATURES
    total = 0
    skipped_short = 0
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for seq, rows in sorted(rows_by_seq.items()):
            for tracklet in _relink_sequence_rows(
                rows,
                max_gap=max_gap,
                base_radius=base_radius,
                radius_per_side=radius_per_side,
                min_iou=min_iou,
                min_score=0.0,
                detector_only=False,
            ):
                if len(tracklet) < min_tracklet_rows:
                    skipped_short += 1
                    continue
                tracklet = sorted(tracklet, key=lambda r: int(r.get("frame_id", 0)))
                track_id = str(tracklet[0].get("proposal_track_id", tracklet[0].get("track_id", f"oracle_{total + 1}")))
                for row in tracklet:
                    row["track_id"] = track_id
                    row["proposal_track_id"] = track_id
                feats = _features(tracklet)
                meta = {
                    "seq": seq,
                    "track_id": track_id,
                    "label": 1,
                    "bucket": "oracle_yolo_positive",
                    "best_iou": 1.0,
                    "matched_frames": len(tracklet),
                    "num_rows_raw": len(tracklet),
                    "dataset_source": dataset_source,
                    **feats,
                }
                writer.writerow({key: meta.get(key, "") for key in fields})
                f_json.write(json.dumps({"meta": meta, "rows": tracklet}, ensure_ascii=False) + "\n")
                total += 1

    summary = {
        "list_files": [str(path) for path in list_files],
        "dataset_source": dataset_source,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "images_skipped": images_skipped,
        "images_seen": images_seen,
        "labels_seen": labels_seen,
        "max_labeled_images_per_seq": max_labeled_images_per_seq,
        "labeled_images_by_seq": dict(sorted(labeled_images_by_seq.items())),
        "num_sequences": len(rows_by_seq),
        "num_tracklets": total,
        "positives": total,
        "negatives": 0,
        "skipped_short_tracklets": skipped_short,
        "min_tracklet_rows": min_tracklet_rows,
        "note": "Oracle YOLO labels are for action-dynamics pretraining/sanity only; do not report as detector proposals.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ProposalTrackletDatasetResult(csv_path=csv_path, json_path=json_path, summary=summary)


def export_yolo_labels_to_gt_csv(
    list_files: list[str | Path],
    out: str | Path,
    image_size: tuple[int, int] | None = None,
    max_images: int | None = None,
) -> YoloRouteBExportResult:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_seen = 0
    labels_seen = 0
    seq_counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["video_path", "frame_id", "x1", "y1", "x2", "y2", "class", "tag"])
        writer.writeheader()
        for list_file in list_files:
            with Path(list_file).open("r", encoding="utf-8-sig", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    if max_images is not None and images_seen >= max_images:
                        break
                    image_path = Path(text)
                    if not image_path.exists():
                        continue
                    images_seen += 1
                    for row in _read_yolo_label_rows(image_path, image_size=image_size):
                        seq = str(row["seq"])
                        labels_seen += 1
                        seq_counts[seq] = seq_counts.get(seq, 0) + 1
                        x1, y1, x2, y2 = row["bbox"]
                        writer.writerow(
                            {
                                "video_path": str(Path(seq) / "visible.mp4"),
                                "frame_id": int(row["frame_id"]),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "class": "drone",
                                "tag": "yolo_label",
                            }
                        )
            if max_images is not None and images_seen >= max_images:
                break
    summary = {
        "out": str(out_path),
        "list_files": [str(path) for path in list_files],
        "images_seen": images_seen,
        "labels_seen": labels_seen,
        "sequences": len(seq_counts),
        "sequence_label_counts": dict(sorted(seq_counts.items())),
    }
    out_path.with_suffix(out_path.suffix + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return YoloRouteBExportResult(out_path=out_path, summary=summary)


def export_yolo_predictions_to_route_b_run(
    list_files: list[str | Path],
    pred_label_dir: str | Path,
    out_run_root: str | Path,
    image_size: tuple[int, int] | None = None,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    source: str = "yolomg_lowconf",
    max_images: int | None = None,
) -> YoloRouteBExportResult:
    out_root = Path(out_run_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows_by_seq: dict[str, list[dict[str, Any]]] = {}
    images_seen = 0
    prediction_rows = 0
    missing_prediction_files = 0
    for list_file in list_files:
        with Path(list_file).open("r", encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                if max_images is not None and images_seen >= max_images:
                    break
                image_path = Path(text)
                if not image_path.exists():
                    continue
                images_seen += 1
                pred_path = Path(pred_label_dir) / f"{image_path.stem}.txt"
                if not pred_path.exists():
                    missing_prediction_files += 1
                    continue
                rows = _read_yolo_prediction_rows(image_path, pred_label_dir, image_size=image_size, source=source)
                prediction_rows += len(rows)
                for row in rows:
                    rows_by_seq.setdefault(str(row["seq"]), []).append(row)
                if images_seen == 1 or images_seen % 5000 == 0:
                    print(
                        json.dumps(
                            {
                                "kind": "export_yolo_predictions_route_b_progress",
                                "images_seen": images_seen,
                                "prediction_rows": prediction_rows,
                                "sequences": len(rows_by_seq),
                                "missing_prediction_files": missing_prediction_files,
                            }
                        ),
                        flush=True,
                    )
        if max_images is not None and images_seen >= max_images:
            break

    sequence_summaries: dict[str, dict[str, Any]] = {}
    for seq, rows in sorted(rows_by_seq.items()):
        seq_dir = out_root / profile / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        path = seq_dir / diagnostics_name
        rows = sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), -float(r.get("objectness", 0.0))))
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        sequence_summaries[seq] = {"diagnostics": str(path), "rows": len(rows)}

    summary = {
        "out_run_root": str(out_root),
        "list_files": [str(path) for path in list_files],
        "pred_label_dir": str(pred_label_dir),
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "source": source,
        "images_seen": images_seen,
        "prediction_rows": prediction_rows,
        "missing_prediction_files": missing_prediction_files,
        "sequences": len(rows_by_seq),
        "sequence_summaries": sequence_summaries,
    }
    summary_path = out_root / "route_b_yolo_prediction_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return YoloRouteBExportResult(out_path=summary_path, summary=summary)


def _relink_sequence_rows(
    rows: list[dict[str, Any]],
    max_gap: int,
    base_radius: float,
    radius_per_side: float,
    min_iou: float,
    min_score: float,
    detector_only: bool,
) -> list[list[dict[str, Any]]]:
    rows = [r for r in rows if _row_score(r) >= min_score and (not detector_only or _source_has_detector(r))]
    rows = sorted(rows, key=lambda r: (int(r.get("frame_id", 0)), -_row_score(r)))
    active: list[dict[str, Any]] = []
    tracklets: list[list[dict[str, Any]]] = []

    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault(int(row.get("frame_id", 0)), []).append(row)

    next_id = 1
    for frame_id in sorted(by_frame):
        frame_rows = by_frame[frame_id]
        used_track_indices: set[int] = set()
        for row in frame_rows:
            best_idx = None
            best_score = -1e9
            side = _box_side(row)
            for idx, tr in enumerate(active):
                if idx in used_track_indices:
                    continue
                gap = frame_id - int(tr["last_frame"])
                if gap <= 0 or gap > max_gap:
                    continue
                last_box = tr["last_bbox"]
                dist = center_distance(last_box, _bbox(row))
                radius = base_radius + radius_per_side * max(side, max(last_box[2] - last_box[0], last_box[3] - last_box[1])) + 4.0 * max(0, gap - 1)
                ov = bbox_iou(last_box, _bbox(row))
                if dist > radius and ov < min_iou:
                    continue
                score = 2.0 * ov - dist / max(radius, 1e-6) + 0.1 * _row_score(row)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is None:
                item = dict(row)
                item["proposal_track_id"] = f"proposal_{next_id}"
                item["track_id"] = item["proposal_track_id"]
                next_id += 1
                tracklets.append([item])
                active.append({"tracklet": tracklets[-1], "last_bbox": _bbox(item), "last_frame": frame_id})
            else:
                tr = active[best_idx]
                item = dict(row)
                item["proposal_track_id"] = tr["tracklet"][0]["proposal_track_id"]
                item["track_id"] = item["proposal_track_id"]
                tr["tracklet"].append(item)
                tr["last_bbox"] = _bbox(item)
                tr["last_frame"] = frame_id
                used_track_indices.add(best_idx)
        active = [tr for tr in active if frame_id - int(tr["last_frame"]) <= max_gap]
    return tracklets


def _label_tracklet(
    seq: str,
    rows: list[dict[str, Any]],
    gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]],
    iou_threshold: float,
    center_threshold: float,
) -> tuple[int, float, int]:
    best_iou = 0.0
    matched_frames = 0
    for row in rows:
        frame_id = int(row.get("frame_id", -1))
        box = _bbox(row)
        matched = False
        for gt in gt_by_key.get((seq, frame_id), []):
            ov = bbox_iou(box, gt)
            best_iou = max(best_iou, ov)
            if ov >= iou_threshold or center_distance(box, gt) <= center_threshold:
                matched = True
        matched_frames += int(matched)
    return int(matched_frames > 0), best_iou, matched_frames


def _bucket(label: int, rows: list[dict[str, Any]], hard_tiny_side: float, hard_low_score: float) -> str:
    scores = [_row_score(r) for r in rows]
    sides = [_box_side(r) for r in rows]
    artifact = max([_artifact_score(r) for r in rows] or [0.0])
    low_alignment = np.mean([float(r.get("alignment_quality", 1.0)) < 0.3 for r in rows]) if rows else 0.0
    source = "+".join(str(r.get("source", "")) for r in rows)
    if label:
        if (np.mean(sides) if sides else 0.0) <= hard_tiny_side or max(scores or [0.0]) <= hard_low_score:
            return "hard_tiny_positive"
        return "positive"
    if artifact >= 0.35 or ("motion" in source and low_alignment >= 0.5):
        return "motion_alignment_artifact"
    if max(scores or [0.0]) >= hard_low_score or "fallback" in source:
        return "high_score_detector_fp"
    return "easy_background"


def build_proposal_tracklet_dataset(
    run_roots: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    profile: str = "hard_recovery",
    diagnostics_name: str = "diagnostics_raw.jsonl",
    max_frames: int | None = None,
    max_gap: int = 3,
    base_radius: float = 18.0,
    radius_per_side: float = 0.75,
    min_iou: float = 0.05,
    min_score: float = 0.0,
    detector_only: bool = False,
    min_tracklet_rows: int = 1,
    iou_threshold: float = 0.3,
    center_threshold: float = 24.0,
    hard_tiny_side: float = 24.0,
    hard_low_score: float = 0.25,
) -> ProposalTrackletDatasetResult:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_key = _load_gt_csv(gt_csv, max_frames=max_frames)
    rows_by_seq = _load_run_diagnostics(run_roots, profile, diagnostics_name, max_frames)
    csv_path = out_dir / "proposal_tracklets.csv"
    json_path = out_dir / "proposal_tracklets.jsonl"
    fields = ["seq", "track_id", "label", "bucket", "best_iou", "matched_frames", "num_rows_raw"] + TRACKLET_FEATURES
    counts: dict[str, int] = {}
    positives = 0
    total = 0

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        for seq, rows in sorted(rows_by_seq.items()):
            for tracklet in _relink_sequence_rows(rows, max_gap, base_radius, radius_per_side, min_iou, min_score, detector_only):
                if len(tracklet) < min_tracklet_rows:
                    continue
                tracklet = sorted(tracklet, key=lambda r: int(r.get("frame_id", 0)))
                label, best_iou, matched_frames = _label_tracklet(seq, tracklet, gt_by_key, iou_threshold, center_threshold)
                bucket = _bucket(label, tracklet, hard_tiny_side, hard_low_score)
                feats = _features(tracklet)
                track_id = str(tracklet[0].get("proposal_track_id", tracklet[0].get("track_id", "")))
                meta = {
                    "seq": seq,
                    "track_id": track_id,
                    "label": label,
                    "bucket": bucket,
                    "best_iou": best_iou,
                    "matched_frames": matched_frames,
                    "num_rows_raw": len(tracklet),
                    **feats,
                }
                writer.writerow(meta)
                f_json.write(json.dumps({"meta": meta, "rows": tracklet}, ensure_ascii=False) + "\n")
                counts[bucket] = counts.get(bucket, 0) + 1
                positives += int(label)
                total += 1

    summary = {
        "run_roots": [str(p) for p in run_roots],
        "gt_csv": str(gt_csv),
        "profile": profile,
        "diagnostics_name": diagnostics_name,
        "num_sequences": len(rows_by_seq),
        "num_tracklets": total,
        "positives": positives,
        "negatives": total - positives,
        "bucket_counts": counts,
        "params": {
            "max_frames": max_frames,
            "max_gap": max_gap,
            "base_radius": base_radius,
            "radius_per_side": radius_per_side,
            "min_iou": min_iou,
            "min_score": min_score,
            "detector_only": detector_only,
            "min_tracklet_rows": min_tracklet_rows,
            "iou_threshold": iou_threshold,
            "center_threshold": center_threshold,
            "hard_tiny_side": hard_tiny_side,
            "hard_low_score": hard_low_score,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ProposalTrackletDatasetResult(csv_path=csv_path, json_path=json_path, summary=summary)


def export_temporal_saliency_tracklets(
    list_files: list[str | Path],
    gt_csv: str | Path,
    out: str | Path,
    dataset_source: str = "vatd_temporal_saliency",
    max_images: int | None = None,
    threshold: float = 24.0,
    min_area: float = 2.0,
    max_area: float = 400.0,
    dilate_iters: int = 1,
    max_gap: int = 3,
    base_radius: float = 18.0,
    radius_per_side: float = 0.75,
    min_iou: float = 0.0,
    min_tracklet_rows: int = 2,
    iou_threshold: float = 0.3,
    center_threshold: float = 24.0,
    hard_tiny_side: float = 24.0,
    hard_low_score: float = 0.25,
    progress_every_sequences: int = 10,
) -> ProposalTrackletDatasetResult:
    """Export VATD-owned weak motion proposals from frame differencing.

    This is an internal proposal source for the independent VATD path. It is not
    a detector prediction export: candidate boxes come from temporal saliency,
    then the same tracklet labeling/export format feeds the video-action model.
    """
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if max_area < min_area:
        raise ValueError("max_area must be >= min_area")
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_by_key = _load_gt_csv(gt_csv, max_frames=None)
    image_paths = _read_image_list_files(list_files, max_images=max_images)
    by_seq_paths: dict[str, list[Path]] = {}
    for image_path in image_paths:
        seq, _frame_id = _safe_seq_frame_from_image(image_path)
        by_seq_paths.setdefault(seq, []).append(image_path)

    csv_path = out_dir / "proposal_tracklets.csv"
    json_path = out_dir / "proposal_tracklets.jsonl"
    fields = ["seq", "track_id", "label", "bucket", "best_iou", "matched_frames", "num_rows_raw", "dataset_source"] + TRACKLET_FEATURES
    counts: dict[str, int] = {}
    positives = 0
    total = 0
    candidate_rows = 0
    missing_images = 0

    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, json_path.open("w", encoding="utf-8") as f_json:
        writer = csv.DictWriter(f_csv, fieldnames=fields)
        writer.writeheader()
        seq_items = sorted(by_seq_paths.items())
        for seq_index, (seq, paths) in enumerate(seq_items, start=1):
            missing_images += sum(1 for path in paths if not path.exists())
            rows = _temporal_saliency_rows_for_sequence(
                paths,
                threshold=threshold,
                min_area=min_area,
                max_area=max_area,
                dilate_iters=dilate_iters,
                source=dataset_source,
            )
            candidate_rows += len(rows)
            for tracklet in _relink_sequence_rows(rows, max_gap, base_radius, radius_per_side, min_iou, min_score=0.0, detector_only=False):
                if len(tracklet) < min_tracklet_rows:
                    continue
                tracklet = sorted(tracklet, key=lambda r: int(r.get("frame_id", 0)))
                label, best_iou, matched_frames = _label_tracklet(seq, tracklet, gt_by_key, iou_threshold, center_threshold)
                bucket = _bucket(label, tracklet, hard_tiny_side, hard_low_score)
                feats = _features(tracklet)
                track_id = str(tracklet[0].get("proposal_track_id", tracklet[0].get("track_id", "")))
                meta = {
                    "seq": seq,
                    "track_id": track_id,
                    "label": label,
                    "bucket": bucket,
                    "best_iou": best_iou,
                    "matched_frames": matched_frames,
                    "num_rows_raw": len(tracklet),
                    "dataset_source": dataset_source,
                    **feats,
                }
                writer.writerow(meta)
                f_json.write(json.dumps({"meta": meta, "rows": tracklet}, ensure_ascii=False) + "\n")
                counts[bucket] = counts.get(bucket, 0) + 1
                positives += int(label)
                total += 1
            if progress_every_sequences > 0 and (seq_index == 1 or seq_index % progress_every_sequences == 0 or seq_index == len(seq_items)):
                f_json.flush()
                print(
                    json.dumps(
                        {
                            "kind": "export_temporal_saliency_tracklets_progress",
                            "dataset_source": dataset_source,
                            "sequences_done": seq_index,
                            "sequences_total": len(seq_items),
                            "images_seen": len(image_paths),
                            "candidate_rows": candidate_rows,
                            "tracklets": total,
                            "positives": positives,
                            "last_seq": seq,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "list_files": [str(path) for path in list_files],
        "gt_csv": str(gt_csv),
        "dataset_source": dataset_source,
        "images_seen": len(image_paths),
        "missing_images": missing_images,
        "num_sequences": len(by_seq_paths),
        "candidate_rows": candidate_rows,
        "num_tracklets": total,
        "positives": positives,
        "negatives": total - positives,
        "bucket_counts": counts,
        "params": {
            "max_images": max_images,
            "threshold": threshold,
            "min_area": min_area,
            "max_area": max_area,
            "dilate_iters": dilate_iters,
            "max_gap": max_gap,
            "base_radius": base_radius,
            "radius_per_side": radius_per_side,
            "min_iou": min_iou,
            "min_tracklet_rows": min_tracklet_rows,
            "iou_threshold": iou_threshold,
            "center_threshold": center_threshold,
            "hard_tiny_side": hard_tiny_side,
            "hard_low_score": hard_low_score,
            "progress_every_sequences": progress_every_sequences,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ProposalTrackletDatasetResult(csv_path=csv_path, json_path=json_path, summary=summary)


def _row_prediction_score(row: dict[str, Any], meta: dict[str, Any], score_field: str | None) -> float:
    if score_field:
        value = row.get(score_field, meta.get(score_field))
        if value is not None:
            return float(value)
    return float(max(row.get("vatd_score", 0.0), row.get("motion_action_score", 0.0), row.get("final_drone_score", 0.0), row.get("objectness", 0.0), meta.get("vatd_score", 0.0)))


def _row_image_stem(row: dict[str, Any], seq: str, frame_id: int) -> str:
    image_path = row.get("image_path") or row.get("frame_path")
    if image_path:
        return Path(str(image_path)).stem
    return f"{seq}_{frame_id:05d}"


def _suppress_flat_prediction_duplicates(
    rows: list[dict[str, Any]],
    nms_iou_threshold: float | None,
    nms_center_threshold: float | None,
) -> tuple[list[dict[str, Any]], int]:
    if nms_iou_threshold is None and nms_center_threshold is None:
        return rows, 0
    kept: list[dict[str, Any]] = []
    suppressed = 0
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_frame.setdefault((str(row["seq"]), int(row["frame_id"])), []).append(row)
    for frame_rows in by_frame.values():
        frame_kept: list[dict[str, Any]] = []
        for row in sorted(frame_rows, key=lambda item: float(item["score"]), reverse=True):
            box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            duplicate = False
            for kept_row in frame_kept:
                kept_box = (float(kept_row["x1"]), float(kept_row["y1"]), float(kept_row["x2"]), float(kept_row["y2"]))
                if nms_iou_threshold is not None and bbox_iou(box, kept_box) >= nms_iou_threshold:
                    duplicate = True
                    break
                if nms_center_threshold is not None and center_distance(box, kept_box) <= nms_center_threshold:
                    duplicate = True
                    break
            if duplicate:
                suppressed += 1
            else:
                frame_kept.append(row)
        kept.extend(frame_kept)
    kept.sort(key=lambda row: (str(row["seq"]), int(row["frame_id"]), str(row["image_stem"]), -float(row["score"])))
    return kept, suppressed


def export_tracklet_jsonl_predictions(
    tracklet_jsonl: str | Path,
    out_dir: str | Path,
    dataset_name: str = "vatd",
    score_field: str | None = "vatd_score",
    min_score: float = 0.0,
    class_id: int = 0,
    formats: list[str] | None = None,
    nms_iou_threshold: float | None = None,
    nms_center_threshold: float | None = None,
) -> TrackletPredictionExportResult:
    if nms_iou_threshold is not None and not (0.0 <= nms_iou_threshold <= 1.0):
        raise ValueError("nms_iou_threshold must be in [0, 1]")
    if nms_center_threshold is not None and nms_center_threshold < 0.0:
        raise ValueError("nms_center_threshold must be nonnegative")
    requested_formats = set(formats or ["flat_csv", "yolo_txt"])
    allowed_formats = {"flat_csv", "yolo_txt"}
    unknown_formats = sorted(requested_formats - allowed_formats)
    if unknown_formats:
        raise ValueError(f"unsupported formats: {unknown_formats}")
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset_name.replace("\\", "_").replace("/", "_").replace(" ", "_")
    flat_rows: list[dict[str, Any]] = []
    yolo_rows_by_file: dict[Path, list[str]] = {}
    total_tracklets = 0
    rows_seen = 0
    rows_below_score = 0
    missing_image_size = 0
    invalid_rows = 0

    for item in _load_jsonl(tracklet_jsonl):
        total_tracklets += 1
        meta = dict(item.get("meta") or {})
        seq = str(meta.get("seq", ""))
        track_id = str(meta.get("track_id", ""))
        for row in item.get("rows") or []:
            rows_seen += 1
            row = dict(row)
            try:
                frame_id = int(float(row.get("frame_id", 0) or 0))
                x1, y1, x2, y2 = _bbox(row)
                score = _row_prediction_score(row, meta, score_field)
            except (TypeError, ValueError):
                invalid_rows += 1
                continue
            if score < min_score:
                rows_below_score += 1
                continue
            if x2 <= x1 or y2 <= y1:
                invalid_rows += 1
                continue
            row_seq = str(row.get("seq", seq))
            image_stem = _row_image_stem(row, row_seq, frame_id)
            image_path = str(row.get("image_path") or row.get("frame_path") or "")
            flat_rows.append(
                {
                    "dataset": dataset_name,
                    "seq": row_seq,
                    "frame_id": frame_id,
                    "image_stem": image_stem,
                    "class_id": class_id,
                    "score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "track_id": track_id or str(row.get("track_id", "")),
                    "image_path": image_path,
                    "source": row.get("source", meta.get("dataset_source", "")),
                    "_image_width": row.get("image_width", meta.get("image_width")),
                    "_image_height": row.get("image_height", meta.get("image_height")),
                }
            )

    rows_after_score_filter = len(flat_rows)
    flat_rows, rows_suppressed_nms = _suppress_flat_prediction_duplicates(flat_rows, nms_iou_threshold, nms_center_threshold)
    rows_exported = len(flat_rows)

    if "yolo_txt" in requested_formats:
        for flat_row in flat_rows:
            width = flat_row.get("_image_width")
            height = flat_row.get("_image_height")
            if width is None or height is None:
                missing_image_size += 1
                continue
            width_f = float(width)
            height_f = float(height)
            if width_f <= 0.0 or height_f <= 0.0:
                missing_image_size += 1
                continue
            x1 = float(flat_row["x1"])
            y1 = float(flat_row["y1"])
            x2 = float(flat_row["x2"])
            y2 = float(flat_row["y2"])
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0
            score = float(flat_row["score"])
            label_path = out_root / "yolo_txt" / safe_dataset / "labels" / f"{flat_row['image_stem']}.txt"
            yolo_rows_by_file.setdefault(label_path, []).append(
                f"{class_id} {cx / width_f:.8f} {cy / height_f:.8f} {bw / width_f:.8f} {bh / height_f:.8f} {score:.8f}"
            )

    flat_csv = out_root / "flat_xyxy_predictions.csv"
    fieldnames = ["dataset", "seq", "frame_id", "image_stem", "class_id", "score", "x1", "y1", "x2", "y2", "track_id", "image_path", "source"]
    if "flat_csv" in requested_formats:
        with flat_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in flat_rows])
    for label_path, lines in yolo_rows_by_file.items():
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "out_dir": str(out_root),
        "dataset_name": dataset_name,
        "score_field": score_field,
        "min_score": min_score,
        "class_id": class_id,
        "formats": sorted(requested_formats),
        "nms_iou_threshold": nms_iou_threshold,
        "nms_center_threshold": nms_center_threshold,
        "flat_csv": str(flat_csv) if "flat_csv" in requested_formats else None,
        "yolo_txt_dir": str(out_root / "yolo_txt") if "yolo_txt" in requested_formats else None,
        "total_tracklets": total_tracklets,
        "rows_seen": rows_seen,
        "rows_after_score_filter": rows_after_score_filter,
        "rows_exported": rows_exported,
        "rows_suppressed_nms": rows_suppressed_nms,
        "rows_below_score": rows_below_score,
        "invalid_rows": invalid_rows,
        "missing_image_size_rows": missing_image_size,
        "yolo_label_files": len(yolo_rows_by_file),
    }
    summary_path = out_root / "tracklet_prediction_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return TrackletPredictionExportResult(out_path=summary_path, summary=summary)


def _gt_seq_from_row(row: dict[str, Any]) -> str:
    seq = str(row.get("seq") or "").strip()
    if seq:
        return seq
    video_path = str(row.get("video_path") or "").strip()
    if video_path:
        return Path(video_path).parent.name or Path(video_path).stem
    image_path = str(row.get("image_path") or row.get("frame_path") or "").strip()
    if image_path:
        parsed_seq, _frame_id = _safe_seq_frame_from_image(Path(image_path))
        return parsed_seq
    return ""


def _load_gt_boxes_for_eval(gt_csv: str | Path) -> dict[tuple[str, int], list[tuple[float, float, float, float]]]:
    gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]] = {}
    with Path(gt_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                seq = _gt_seq_from_row(row)
                frame_id = int(float(row.get("frame_id", 0) or 0))
                box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not seq or box[2] <= box[0] or box[3] <= box[1]:
                continue
            gt_by_key.setdefault((seq, frame_id), []).append(box)
    return gt_by_key


def _load_flat_predictions_for_eval(prediction_csv: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(prediction_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                seq = str(row.get("seq") or "").strip()
                frame_id = int(float(row.get("frame_id", 0) or 0))
                score = float(row.get("score", 0.0) or 0.0)
                box = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not seq or box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows.append(
                {
                    "seq": seq,
                    "frame_id": frame_id,
                    "score": score,
                    "bbox": box,
                    "track_id": row.get("track_id", ""),
                    "source": row.get("source", ""),
                }
            )
    return rows


def _match_flat_predictions_at_threshold(
    gt_by_key: dict[tuple[str, int], list[tuple[float, float, float, float]]],
    predictions: list[dict[str, Any]],
    threshold: float,
    iou_threshold: float,
    center_threshold: float | None,
) -> dict[str, Any]:
    preds_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in predictions:
        if float(row["score"]) >= threshold:
            preds_by_key.setdefault((str(row["seq"]), int(row["frame_id"])), []).append(row)

    tp = 0
    fp = 0
    fn = 0
    matched_scores: list[float] = []
    keys = set(gt_by_key) | set(preds_by_key)
    for key in keys:
        gt_boxes = gt_by_key.get(key, [])
        pred_rows = sorted(preds_by_key.get(key, []), key=lambda item: float(item["score"]), reverse=True)
        used_gt: set[int] = set()
        for pred in pred_rows:
            pred_box = tuple(float(v) for v in pred["bbox"])
            best_index = None
            best_iou = -1.0
            best_center = float("inf")
            for gt_index, gt_box in enumerate(gt_boxes):
                if gt_index in used_gt:
                    continue
                iou = bbox_iou(pred_box, gt_box)
                dist = center_distance(pred_box, gt_box)
                matched = iou >= iou_threshold or (center_threshold is not None and dist <= center_threshold)
                if matched and (iou, -dist) > (best_iou, -best_center):
                    best_index = gt_index
                    best_iou = iou
                    best_center = dist
            if best_index is None:
                fp += 1
            else:
                used_gt.add(best_index)
                tp += 1
                matched_scores.append(float(pred["score"]))
        fn += max(0, len(gt_boxes) - len(used_gt))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    frame_count = max(1, len({key for key in keys}))
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fppi": fp / frame_count,
        "detections": tp + fp,
        "gt": tp + fn,
        "frames": frame_count,
        "matched_score_mean": float(np.mean(matched_scores)) if matched_scores else 0.0,
    }


def evaluate_flat_tracklet_predictions(
    gt_csv: str | Path,
    prediction_csv: str | Path,
    out_dir: str | Path,
    thresholds: list[float] | None = None,
    iou_threshold: float = 0.5,
    center_threshold: float | None = None,
    fp_limit: int | None = None,
    max_fppi: float | None = None,
    fp_limits: list[int] | None = None,
    max_fppis: list[float] | None = None,
) -> FlatPredictionEvaluationResult:
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if center_threshold is not None and center_threshold < 0.0:
        raise ValueError("center_threshold must be nonnegative")
    gt_by_key = _load_gt_boxes_for_eval(gt_csv)
    predictions = _load_flat_predictions_for_eval(prediction_csv)
    if thresholds is None:
        unique_scores = sorted({float(row["score"]) for row in predictions}, reverse=True)
        thresholds = unique_scores + [0.0]
    thresholds = sorted({float(value) for value in thresholds}, reverse=True)

    sweep_rows = [
        _match_flat_predictions_at_threshold(gt_by_key, predictions, threshold, iou_threshold, center_threshold)
        for threshold in thresholds
    ]
    best_f1 = max(sweep_rows, key=lambda row: (float(row["f1"]), float(row["recall"]), -float(row["fp"])), default=None)
    budget_rows = sweep_rows
    if fp_limit is not None:
        budget_rows = [row for row in budget_rows if int(row["fp"]) <= fp_limit]
    if max_fppi is not None:
        budget_rows = [row for row in budget_rows if float(row["fppi"]) <= max_fppi]
    best_under_budget = max(
        budget_rows,
        key=lambda row: (float(row["recall"]), float(row["precision"]), float(row["f1"]), -float(row["fp"])),
        default=None,
    )
    fp_budget_curve = []
    for limit in sorted({int(value) for value in (fp_limits or [])}):
        candidates = [row for row in sweep_rows if int(row["fp"]) <= limit]
        best = max(
            candidates,
            key=lambda row: (float(row["recall"]), float(row["precision"]), float(row["f1"]), -float(row["fp"])),
            default=None,
        )
        fp_budget_curve.append({"fp_limit": limit, "best": best})
    fppi_budget_curve = []
    for limit in sorted({float(value) for value in (max_fppis or [])}):
        candidates = [row for row in sweep_rows if float(row["fppi"]) <= limit]
        best = max(
            candidates,
            key=lambda row: (float(row["recall"]), float(row["precision"]), float(row["f1"]), -float(row["fp"])),
            default=None,
        )
        fppi_budget_curve.append({"max_fppi": limit, "best": best})

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "flat_prediction_threshold_sweep.csv"
    fields = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "fppi", "detections", "gt", "frames", "matched_score_mean"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sweep_rows)
    fp_budget_curve_csv = out_root / "flat_prediction_fp_budget_curve.csv"
    with fp_budget_curve_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["fp_limit", "available", *fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in fp_budget_curve:
            best = item.get("best")
            row = {"fp_limit": item["fp_limit"], "available": best is not None}
            if best is not None:
                row.update(best)
            writer.writerow(row)
    fppi_budget_curve_csv = out_root / "flat_prediction_fppi_budget_curve.csv"
    with fppi_budget_curve_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["max_fppi", "available", *fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in fppi_budget_curve:
            best = item.get("best")
            row = {"max_fppi": item["max_fppi"], "available": best is not None}
            if best is not None:
                row.update(best)
            writer.writerow(row)

    summary = {
        "gt_csv": str(gt_csv),
        "prediction_csv": str(prediction_csv),
        "out_dir": str(out_root),
        "iou_threshold": iou_threshold,
        "center_threshold": center_threshold,
        "fp_limit": fp_limit,
        "max_fppi": max_fppi,
        "fp_limits": fp_limits or [],
        "max_fppis": max_fppis or [],
        "gt_frames": len(gt_by_key),
        "gt_boxes": int(sum(len(value) for value in gt_by_key.values())),
        "prediction_rows": len(predictions),
        "thresholds": thresholds,
        "best_f1": best_f1,
        "best_under_budget": best_under_budget,
        "fp_budget_curve": fp_budget_curve,
        "fppi_budget_curve": fppi_budget_curve,
        "sweep_csv": str(csv_path),
        "fp_budget_curve_csv": str(fp_budget_curve_csv),
        "fppi_budget_curve_csv": str(fppi_budget_curve_csv),
    }
    summary_path = out_root / "flat_prediction_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return FlatPredictionEvaluationResult(csv_path=csv_path, summary_path=summary_path, summary=summary)


def _nms_value_slug(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def sweep_flat_tracklet_prediction_nms(
    tracklet_jsonl: str | Path,
    gt_csv: str | Path,
    out_dir: str | Path,
    dataset_name: str = "vatd",
    score_field: str | None = "vatd_score",
    min_score: float = 0.0,
    class_id: int = 0,
    iou_thresholds: list[float | None] | None = None,
    center_thresholds: list[float | None] | None = None,
    score_thresholds: list[float] | None = None,
    eval_iou_threshold: float = 0.5,
    eval_center_threshold: float | None = None,
    fp_limit: int | None = None,
    max_fppi: float | None = None,
    fp_limits: list[int] | None = None,
    max_fppis: list[float] | None = None,
) -> FlatPredictionNmsSweepResult:
    iou_values = iou_thresholds if iou_thresholds is not None else [None, 0.3, 0.5]
    center_values = center_thresholds if center_thresholds is not None else [None, 6.0, 12.0]
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for nms_iou in iou_values:
        for nms_center in center_values:
            combo_slug = f"iou_{_nms_value_slug(nms_iou)}__center_{_nms_value_slug(nms_center)}"
            combo_dir = out_root / combo_slug
            prediction_dir = combo_dir / "predictions"
            eval_dir = combo_dir / "eval"
            export_result = export_tracklet_jsonl_predictions(
                tracklet_jsonl,
                prediction_dir,
                dataset_name=dataset_name,
                score_field=score_field,
                min_score=min_score,
                class_id=class_id,
                formats=["flat_csv"],
                nms_iou_threshold=nms_iou,
                nms_center_threshold=nms_center,
            )
            flat_csv = export_result.summary.get("flat_csv")
            eval_result = evaluate_flat_tracklet_predictions(
                gt_csv,
                flat_csv,
                eval_dir,
                thresholds=score_thresholds,
                iou_threshold=eval_iou_threshold,
                center_threshold=eval_center_threshold,
                fp_limit=fp_limit,
                max_fppi=max_fppi,
                fp_limits=fp_limits,
                max_fppis=max_fppis,
            )
            best = eval_result.summary.get("best_under_budget") or eval_result.summary.get("best_f1") or {}
            rows.append(
                {
                    "nms_iou_threshold": "" if nms_iou is None else nms_iou,
                    "nms_center_threshold": "" if nms_center is None else nms_center,
                    "rows_exported": export_result.summary.get("rows_exported", 0),
                    "rows_suppressed_nms": export_result.summary.get("rows_suppressed_nms", 0),
                    "best_threshold": best.get("threshold", ""),
                    "tp": best.get("tp", 0),
                    "fp": best.get("fp", 0),
                    "fn": best.get("fn", 0),
                    "precision": best.get("precision", 0.0),
                    "recall": best.get("recall", 0.0),
                    "f1": best.get("f1", 0.0),
                    "fppi": best.get("fppi", 0.0),
                    "best_under_budget_available": eval_result.summary.get("best_under_budget") is not None,
                    "prediction_csv": flat_csv,
                    "eval_summary": str(eval_result.summary_path),
                }
            )

    best_under_budget = max(
        [row for row in rows if row["best_under_budget_available"]],
        key=lambda row: (float(row["recall"]), float(row["precision"]), float(row["f1"]), -float(row["fp"])),
        default=None,
    )
    best_f1 = max(rows, key=lambda row: (float(row["f1"]), float(row["recall"]), -float(row["fp"])), default=None)
    selected = best_under_budget or best_f1
    final_export_summary = None
    final_eval_summary = None
    if selected is not None:
        selected_iou = None if selected["nms_iou_threshold"] == "" else float(selected["nms_iou_threshold"])
        selected_center = None if selected["nms_center_threshold"] == "" else float(selected["nms_center_threshold"])
        final_prediction_dir = out_root / "final_predictions"
        final_eval_dir = out_root / "final_eval"
        final_export = export_tracklet_jsonl_predictions(
            tracklet_jsonl,
            final_prediction_dir,
            dataset_name=dataset_name,
            score_field=score_field,
            min_score=min_score,
            class_id=class_id,
            formats=["flat_csv", "yolo_txt"],
            nms_iou_threshold=selected_iou,
            nms_center_threshold=selected_center,
        )
        final_eval = evaluate_flat_tracklet_predictions(
            gt_csv,
            final_export.summary["flat_csv"],
            final_eval_dir,
            thresholds=score_thresholds,
            iou_threshold=eval_iou_threshold,
            center_threshold=eval_center_threshold,
            fp_limit=fp_limit,
            max_fppi=max_fppi,
            fp_limits=fp_limits,
            max_fppis=max_fppis,
        )
        final_export_summary = final_export.summary
        final_eval_summary = final_eval.summary
    csv_path = out_root / "flat_prediction_nms_sweep.csv"
    fields = [
        "nms_iou_threshold",
        "nms_center_threshold",
        "rows_exported",
        "rows_suppressed_nms",
        "best_threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fppi",
        "best_under_budget_available",
        "prediction_csv",
        "eval_summary",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "tracklet_jsonl": str(tracklet_jsonl),
        "gt_csv": str(gt_csv),
        "out_dir": str(out_root),
        "dataset_name": dataset_name,
        "score_field": score_field,
        "min_score": min_score,
        "class_id": class_id,
        "iou_thresholds": iou_values,
        "center_thresholds": center_values,
        "score_thresholds": score_thresholds,
        "eval_iou_threshold": eval_iou_threshold,
        "eval_center_threshold": eval_center_threshold,
        "fp_limit": fp_limit,
        "max_fppi": max_fppi,
        "fp_limits": fp_limits or [],
        "max_fppis": max_fppis or [],
        "num_runs": len(rows),
        "best_under_budget": best_under_budget,
        "best_f1": best_f1,
        "selected_final": selected,
        "final_prediction_export": final_export_summary,
        "final_eval": final_eval_summary,
        "sweep_csv": str(csv_path),
    }
    summary_path = out_root / "flat_prediction_nms_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return FlatPredictionNmsSweepResult(csv_path=csv_path, summary_path=summary_path, summary=summary)


def _metric_row_from_eval_best(
    method: str,
    summary_path: str | Path,
    budget_type: str,
    budget: int | float | str,
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {
        "method": method,
        "budget_type": budget_type,
        "budget": budget,
        "available": best is not None,
        "threshold": "",
        "tp": "",
        "fp": "",
        "fn": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "fppi": "",
        "detections": "",
        "gt": "",
        "frames": "",
        "summary_path": str(summary_path),
    }
    if best is not None:
        for key in ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "fppi", "detections", "gt", "frames"]:
            row[key] = best.get(key, "")
    return row


def compare_flat_prediction_eval_summaries(
    summaries: list[str | Path],
    out_dir: str | Path,
    method_names: list[str] | None = None,
) -> FlatPredictionEvalComparisonResult:
    if not summaries:
        raise ValueError("at least one summary is required")
    if method_names is not None and len(method_names) != len(summaries):
        raise ValueError("method_names must have the same length as summaries")
    methods = method_names or [Path(path).parent.name for path in summaries]
    rows: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for method, summary_path in zip(methods, summaries):
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        loaded.append({"method": method, "summary_path": str(summary_path), "summary": summary})
        rows.append(_metric_row_from_eval_best(method, summary_path, "best_f1", "best_f1", summary.get("best_f1")))
        rows.append(_metric_row_from_eval_best(method, summary_path, "best_under_budget", "selected", summary.get("best_under_budget")))
        for item in summary.get("fp_budget_curve") or []:
            rows.append(_metric_row_from_eval_best(method, summary_path, "fp_limit", item.get("fp_limit", ""), item.get("best")))
        for item in summary.get("fppi_budget_curve") or []:
            rows.append(_metric_row_from_eval_best(method, summary_path, "max_fppi", item.get("max_fppi", ""), item.get("best")))

    baseline_by_budget: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["budget_type"]), str(row["budget"]))
        if key not in baseline_by_budget and row["available"]:
            baseline_by_budget[key] = row
    for row in rows:
        baseline = baseline_by_budget.get((str(row["budget_type"]), str(row["budget"])))
        row["baseline_method"] = baseline.get("method", "") if baseline else ""
        for metric in ["precision", "recall", "f1", "fp", "fppi"]:
            delta_key = f"delta_{metric}_vs_baseline"
            if not baseline or row.get(metric, "") == "" or baseline.get(metric, "") == "":
                row[delta_key] = ""
            else:
                row[delta_key] = float(row[metric]) - float(baseline[metric])
        if not row["available"]:
            row["recall_verdict_vs_baseline"] = "unavailable"
            row["recall_win_vs_baseline"] = ""
        elif not baseline:
            row["recall_verdict_vs_baseline"] = "no_baseline"
            row["recall_win_vs_baseline"] = ""
        elif row["method"] == baseline.get("method"):
            row["recall_verdict_vs_baseline"] = "baseline"
            row["recall_win_vs_baseline"] = "False"
        elif row["delta_recall_vs_baseline"] == "":
            row["recall_verdict_vs_baseline"] = "no_comparable_metric"
            row["recall_win_vs_baseline"] = ""
        else:
            delta_recall = float(row["delta_recall_vs_baseline"])
            if delta_recall > 1e-12:
                row["recall_verdict_vs_baseline"] = "win"
                row["recall_win_vs_baseline"] = "True"
            elif delta_recall < -1e-12:
                row["recall_verdict_vs_baseline"] = "loss"
                row["recall_win_vs_baseline"] = "False"
            else:
                row["recall_verdict_vs_baseline"] = "tie"
                row["recall_win_vs_baseline"] = "False"

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "flat_prediction_eval_comparison.csv"
    fields = [
        "method",
        "budget_type",
        "budget",
        "available",
        "threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "fppi",
        "detections",
        "gt",
        "frames",
        "baseline_method",
        "delta_precision_vs_baseline",
        "delta_recall_vs_baseline",
        "delta_f1_vs_baseline",
        "delta_fp_vs_baseline",
        "delta_fppi_vs_baseline",
        "recall_verdict_vs_baseline",
        "recall_win_vs_baseline",
        "summary_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = out_root / "flat_prediction_eval_comparison.md"
    md_lines = [
        "# Flat Prediction Eval Comparison",
        "",
        "| method | budget_type | budget | available | precision | recall | f1 | fp | fppi | delta_recall | delta_fp | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {budget_type} | {budget} | {available} | {precision} | {recall} | {f1} | {fp} | {fppi} | {delta_recall} | {delta_fp} | {verdict} |".format(
                method=row["method"],
                budget_type=row["budget_type"],
                budget=row["budget"],
                available=row["available"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
                fp=row["fp"],
                fppi=row["fppi"],
                delta_recall=row["delta_recall_vs_baseline"],
                delta_fp=row["delta_fp_vs_baseline"],
                verdict=row["recall_verdict_vs_baseline"],
            )
        )
    markdown_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    verdict_summary: dict[str, Any] = {}
    paper_claim_rows: list[dict[str, Any]] = []
    for row in rows:
        verdict = str(row["recall_verdict_vs_baseline"])
        if verdict in {"baseline", "unavailable", "no_baseline", "no_comparable_metric"}:
            continue
        method = str(row["method"])
        budget_type = str(row["budget_type"])
        method_summary = verdict_summary.setdefault(
            method,
            {"win": 0, "tie": 0, "loss": 0, "by_budget_type": {}},
        )
        method_summary[verdict] += 1
        budget_summary = method_summary["by_budget_type"].setdefault(
            budget_type,
            {"win": 0, "tie": 0, "loss": 0},
        )
        budget_summary[verdict] += 1
        if row["budget_type"] in {"best_under_budget", "fp_limit", "max_fppi"}:
            delta_recall = float(row["delta_recall_vs_baseline"]) if row["delta_recall_vs_baseline"] != "" else 0.0
            paper_claim_rows.append(
                {
                    "method": method,
                    "baseline_method": row["baseline_method"],
                    "budget_type": row["budget_type"],
                    "budget": row["budget"],
                    "verdict": verdict,
                    "recall": row["recall"],
                    "baseline_recall": float(row["recall"]) - delta_recall if row["recall"] != "" else "",
                    "delta_recall_vs_baseline": row["delta_recall_vs_baseline"],
                    "fp": row["fp"],
                    "delta_fp_vs_baseline": row["delta_fp_vs_baseline"],
                    "fppi": row["fppi"],
                    "delta_fppi_vs_baseline": row["delta_fppi_vs_baseline"],
                    "precision": row["precision"],
                    "f1": row["f1"],
                    "threshold": row["threshold"],
                }
            )

    claim_csv_path = out_root / "flat_prediction_paper_claim_rows.csv"
    claim_fields = [
        "method",
        "baseline_method",
        "budget_type",
        "budget",
        "verdict",
        "recall",
        "baseline_recall",
        "delta_recall_vs_baseline",
        "fp",
        "delta_fp_vs_baseline",
        "fppi",
        "delta_fppi_vs_baseline",
        "precision",
        "f1",
        "threshold",
    ]
    with claim_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerows(paper_claim_rows)

    claim_markdown_path = out_root / "flat_prediction_paper_claim_rows.md"
    claim_md_lines = [
        "# Flat Prediction Paper Claim Rows",
        "",
        "| method | baseline | budget_type | budget | verdict | recall | baseline_recall | delta_recall | fp | fppi |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in paper_claim_rows:
        claim_md_lines.append(
            "| {method} | {baseline_method} | {budget_type} | {budget} | {verdict} | {recall} | {baseline_recall} | {delta_recall} | {fp} | {fppi} |".format(
                method=row["method"],
                baseline_method=row["baseline_method"],
                budget_type=row["budget_type"],
                budget=row["budget"],
                verdict=row["verdict"],
                recall=row["recall"],
                baseline_recall=row["baseline_recall"],
                delta_recall=row["delta_recall_vs_baseline"],
                fp=row["fp"],
                fppi=row["fppi"],
            )
        )
    claim_markdown_path.write_text("\n".join(claim_md_lines) + "\n", encoding="utf-8")

    paper_claim_wins = [row for row in paper_claim_rows if row.get("verdict") == "win"]
    fixed_budget_verdict_summary: dict[str, Any] = {}
    for row in paper_claim_rows:
        method = str(row["method"])
        verdict = str(row["verdict"])
        budget_type = str(row["budget_type"])
        method_summary = fixed_budget_verdict_summary.setdefault(
            method,
            {"win": 0, "tie": 0, "loss": 0, "by_budget_type": {}},
        )
        method_summary[verdict] += 1
        budget_summary = method_summary["by_budget_type"].setdefault(
            budget_type,
            {"win": 0, "tie": 0, "loss": 0},
        )
        budget_summary[verdict] += 1
    claim_gate_status = "pass" if paper_claim_wins else "insufficient_evidence"
    claim_gate_reason = (
        "at least one fixed-budget recall win over the baseline"
        if paper_claim_wins
        else "no fixed-budget recall win over the baseline"
    )

    claim_wins_csv_path = out_root / "flat_prediction_paper_claim_wins.csv"
    with claim_wins_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerows(paper_claim_wins)

    claim_wins_markdown_path = out_root / "flat_prediction_paper_claim_wins.md"
    claim_win_md_lines = [
        "# Flat Prediction Paper Claim Wins",
        "",
        "| method | baseline | budget_type | budget | recall | baseline_recall | delta_recall | fp | fppi |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper_claim_wins:
        claim_win_md_lines.append(
            "| {method} | {baseline_method} | {budget_type} | {budget} | {recall} | {baseline_recall} | {delta_recall} | {fp} | {fppi} |".format(
                method=row["method"],
                baseline_method=row["baseline_method"],
                budget_type=row["budget_type"],
                budget=row["budget"],
                recall=row["recall"],
                baseline_recall=row["baseline_recall"],
                delta_recall=row["delta_recall_vs_baseline"],
                fp=row["fp"],
                fppi=row["fppi"],
            )
        )
    claim_wins_markdown_path.write_text("\n".join(claim_win_md_lines) + "\n", encoding="utf-8")

    fixed_budget_report_path = out_root / "flat_prediction_fixed_budget_report.md"
    report_lines = [
        "# Fixed-Budget Flat Prediction Comparison",
        "",
        f"- Baseline method: {methods[0]}",
        f"- Compared methods: {', '.join(methods[1:]) if len(methods) > 1 else ''}",
        f"- Claim gate: {claim_gate_status} ({claim_gate_reason})",
        f"- Fixed-budget claim rows: {len(paper_claim_rows)}",
        f"- Fixed-budget wins: {len(paper_claim_wins)}",
        f"- Claim rows CSV: `{claim_csv_path.name}`",
        f"- Claim wins CSV: `{claim_wins_csv_path.name}`",
        "",
        "## Fixed-Budget Verdict Summary",
        "",
        "| method | win | tie | loss |",
        "|---|---:|---:|---:|",
    ]
    for method, method_summary in fixed_budget_verdict_summary.items():
        report_lines.append(
            "| {method} | {win} | {tie} | {loss} |".format(
                method=method,
                win=method_summary.get("win", 0),
                tie=method_summary.get("tie", 0),
                loss=method_summary.get("loss", 0),
            )
        )
    report_lines.extend(["", "## Paper Claim Wins", ""])
    if paper_claim_wins:
        report_lines.extend(
            [
                "| method | baseline | budget_type | budget | recall | baseline_recall | delta_recall | fp | fppi |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in paper_claim_wins:
            report_lines.append(
                "| {method} | {baseline_method} | {budget_type} | {budget} | {recall} | {baseline_recall} | {delta_recall} | {fp} | {fppi} |".format(
                    method=row["method"],
                    baseline_method=row["baseline_method"],
                    budget_type=row["budget_type"],
                    budget=row["budget"],
                    recall=row["recall"],
                    baseline_recall=row["baseline_recall"],
                    delta_recall=row["delta_recall_vs_baseline"],
                    fp=row["fp"],
                    fppi=row["fppi"],
                )
            )
    else:
        report_lines.append("No fixed-budget recall wins over the baseline were found in this comparison.")
    fixed_budget_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    best_fixed_budget_win = None
    if paper_claim_wins:
        best_fixed_budget_win = max(
            paper_claim_wins,
            key=lambda row: (
                float(row["delta_recall_vs_baseline"]) if row["delta_recall_vs_baseline"] != "" else float("-inf"),
                float(row["recall"]) if row["recall"] != "" else float("-inf"),
            ),
        )
    paper_result_summary = {
        "baseline_method": methods[0],
        "compared_methods": methods[1:],
        "claim_gate": {
            "status": claim_gate_status,
            "reason": claim_gate_reason,
            "requires": "fixed-budget recall win over baseline at best_under_budget, fp_limit, or max_fppi",
        },
        "fixed_budget_claim_rows": len(paper_claim_rows),
        "fixed_budget_wins": len(paper_claim_wins),
        "fixed_budget_verdict_summary": fixed_budget_verdict_summary,
        "best_fixed_budget_win": best_fixed_budget_win,
        "claim_rows_csv": str(claim_csv_path),
        "claim_wins_csv": str(claim_wins_csv_path),
        "fixed_budget_report_markdown": str(fixed_budget_report_path),
    }
    paper_result_summary_path = out_root / "flat_prediction_paper_result_summary.json"
    paper_result_summary_path.write_text(json.dumps(paper_result_summary, indent=2), encoding="utf-8")
    claim_gate_summary = {
        "claim_gate": paper_result_summary["claim_gate"],
        "baseline_method": paper_result_summary["baseline_method"],
        "compared_methods": paper_result_summary["compared_methods"],
        "fixed_budget_claim_rows": paper_result_summary["fixed_budget_claim_rows"],
        "fixed_budget_wins": paper_result_summary["fixed_budget_wins"],
        "best_fixed_budget_win": paper_result_summary["best_fixed_budget_win"],
        "claim_rows_csv": paper_result_summary["claim_rows_csv"],
        "claim_wins_csv": paper_result_summary["claim_wins_csv"],
        "fixed_budget_report_markdown": paper_result_summary["fixed_budget_report_markdown"],
        "paper_result_summary_json": str(paper_result_summary_path),
    }
    claim_gate_summary_path = out_root / "flat_prediction_claim_gate.json"
    claim_gate_summary_path.write_text(json.dumps(claim_gate_summary, indent=2), encoding="utf-8")

    summary = {
        "summaries": [str(path) for path in summaries],
        "method_names": methods,
        "out_dir": str(out_root),
        "rows": len(rows),
        "comparison_csv": str(csv_path),
        "comparison_markdown": str(markdown_path),
        "verdict_summary": verdict_summary,
        "fixed_budget_verdict_summary": fixed_budget_verdict_summary,
        "paper_claim_rows": paper_claim_rows,
        "paper_claim_rows_csv": str(claim_csv_path),
        "paper_claim_rows_markdown": str(claim_markdown_path),
        "paper_claim_wins": paper_claim_wins,
        "paper_claim_wins_csv": str(claim_wins_csv_path),
        "paper_claim_wins_markdown": str(claim_wins_markdown_path),
        "fixed_budget_report_markdown": str(fixed_budget_report_path),
        "paper_result_summary": paper_result_summary,
        "paper_result_summary_json": str(paper_result_summary_path),
        "claim_gate": claim_gate_summary,
        "claim_gate_json": str(claim_gate_summary_path),
        "inputs": loaded,
    }
    summary_path = out_root / "flat_prediction_eval_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return FlatPredictionEvalComparisonResult(csv_path=csv_path, markdown_path=markdown_path, summary_path=summary_path, summary=summary)


def merge_tracklet_jsonl(
    inputs: list[str | Path],
    out: str | Path,
    source_names: list[str] | None = None,
) -> MergedTrackletJsonlResult:
    if not inputs:
        raise ValueError("At least one input JSONL is required")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source_names = source_names or [Path(path).parent.name or f"source_{idx + 1}" for idx, path in enumerate(inputs)]
    if len(source_names) != len(inputs):
        raise ValueError("--source-names must have the same length as --inputs")
    seen: set[tuple[str, str, str]] = set()
    counts: dict[str, int] = {}
    labels = {0: 0, 1: 0}
    total = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for source, path in zip(source_names, inputs):
            for item in _load_jsonl(path):
                meta = dict(item.get("meta") or {})
                seq = str(meta.get("seq", ""))
                track_id = str(meta.get("track_id", ""))
                key = (source, seq, track_id)
                if key in seen:
                    continue
                seen.add(key)
                meta["dataset_source"] = source
                label = int(float(meta.get("label", 0)))
                labels[label] = labels.get(label, 0) + 1
                bucket = str(meta.get("bucket", "unbucketed"))
                counts[f"{source}:{bucket}"] = counts.get(f"{source}:{bucket}", 0) + 1
                item["meta"] = meta
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1
    summary = {
        "inputs": [str(path) for path in inputs],
        "source_names": source_names,
        "json_path": str(out_path),
        "num_tracklets": total,
        "positives": labels.get(1, 0),
        "negatives": labels.get(0, 0),
        "source_bucket_counts": counts,
    }
    (out_path.parent / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return MergedTrackletJsonlResult(json_path=out_path, summary=summary)
