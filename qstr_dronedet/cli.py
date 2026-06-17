from __future__ import annotations

import argparse
import bisect
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qstr_dronedet.candidates import candidates_from_motion, candidates_from_yolo, candidates_from_yolo_tiled, merge_candidates
from qstr_dronedet.candidates.merge import bbox_iou, center_distance, nms_candidates
from qstr_dronedet.candidates.yolo_p2_train import (
    build_class_agnostic_yolo_dataset,
    build_tiled_class_agnostic_yolo_dataset,
    train_yolo_p2,
    write_yolov8_p2_model_yaml,
)
from qstr_dronedet.data_augment import (
    build_static_hover_crop_dataset,
    build_static_hover_temporal_dataset,
    build_detector_proposal_stage_b_dataset,
    export_hard_negative_feature_csv,
    export_static_hover_feature_csv,
    export_static_hover_frames_csv,
    make_speed_augmented_frame_csv,
    make_speed_augmented_video_dir,
    make_speed_augmented_videos,
    make_moving_target_sample,
    make_static_hover_sample,
    mine_hard_negative_temporal_dataset,
    mine_motion_hard_negative_crops,
    split_feature_csv,
    split_folder_dataset,
)
from qstr_dronedet.evaluation.metrics import evaluate_predictions
from qstr_dronedet.evaluation.fusion_calibration import calibrate_fusion_from_diagnostics
from qstr_dronedet.evaluation.frame_failure_analysis import analyze_frame_failures
from qstr_dronedet.evaluation.proposal_stage_b import evaluate_crop_recognizer_on_proposals
from qstr_dronedet.evaluation.stage_b_benchmark import run_stage_b_oracle_benchmark
from qstr_dronedet.evaluation.tracker_benchmark import run_tracker_oracle_benchmark
from qstr_dronedet.evaluation.tracklet_filter_sweep import run_tracklet_filter_sweep
from qstr_dronedet.evaluation.tracklet_model_selection import run_tracklet_model_selection
from qstr_dronedet.evaluation.tracklet_sequence_model_selection import run_tracklet_sequence_model_selection
from qstr_dronedet.features.roi import crop_with_context, extract_temporal_tube
from qstr_dronedet.fusion.modes import determine_mode
from qstr_dronedet.fusion.rule_fusion import fuse_rule_based
from qstr_dronedet.motion.difference import compute_multik_motion, motion_score_in_bbox
from qstr_dronedet.recognition.train import train_crop_recognizer, train_feature_recognizer, train_temporal_recognizer
from qstr_dronedet.real_data import (
    build_real_stage_b_datasets,
    build_real_detector_proposal_stage_b_dataset,
    build_real_yolo_candidate_dataset,
    ensure_real_data_layout,
    export_ard100_annotations,
    export_anti_uav300_subset_from_zip,
    extract_real_annotated_frames,
)
from qstr_dronedet.tracking.kalman import ConstantVelocityTracker
from qstr_dronedet.tracking.action_chunk import (
    attach_frame_priors_to_tracklets,
    build_frame_prior_index_from_heatmaps,
    export_action_chunk_dataset_from_tracklets,
    export_action_prior_heatmaps_from_sample_scores,
    merge_action_chunk_datasets,
    split_action_chunk_dataset,
)
from qstr_dronedet.tracking.action_policy import (
    attach_action_dynamics_scores_to_tracklets,
    attach_tracklet_confidence_fusion_scores,
    build_route_b_baseline_report,
    collect_route_b_result_summaries,
    compare_route_b_results_to_baselines,
    evaluate_action_dynamics_thresholds,
    evaluate_action_chunk_policy,
    export_route_b_baseline_markdown_table,
    run_action_policy_ablation,
    run_action_policy_split_selection,
    run_multisource_action_policy_experiment,
    run_multisource_proposal_policy_benchmark,
    run_multisource_tracklet_action_policy_experiment,
    run_multisource_tracklet_policy_benchmark,
    run_action_dynamics_tracklet_ablation,
    run_action_dynamics_tracklet_pipeline,
    score_tracklets_with_action_policy,
    score_tracklets_with_constant_velocity,
    train_action_chunk_policy,
    validate_route_b_tracklet_inputs,
    validate_route_b_baseline_csv,
    write_route_b_baseline_template,
    write_route_b_official_baseline_seed,
)
from qstr_dronedet.tracking.action_prior_fusion import (
    fuse_action_frame_prior_predictions,
    sweep_action_frame_prior_fusion,
    sweep_action_frame_prior_fusion_run_root,
)
from qstr_dronedet.tracking.offline_selector import build_seed_admission_dataset, replay_offline_selector
from qstr_dronedet.tracking.proposal_tracklets import (
    build_proposal_tracklet_dataset,
    compare_flat_prediction_eval_summaries,
    evaluate_flat_tracklet_predictions,
    export_temporal_saliency_tracklets,
    export_frame_list_from_gt_csv,
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
from qstr_dronedet.tracking.tracklet_classifier import (
    apply_tracklet_filter_to_infer_outputs,
    build_tracklet_classifier_official_eval_bundle,
    build_tracklet_dataset,
    evaluate_tracklet_classifier,
    evaluate_tracklet_classifier_thresholds,
    export_aot_prediction_parts_to_tracklets,
    filter_aot_prediction_parts_by_tracklets,
    rescore_aot_prediction_parts_by_tracklets,
    export_tracklet_classifier_aot_prediction_parts,
    export_tracklet_classifier_official_predictions,
    export_tracklet_jsonl_classifier_dataset,
    merge_tracklet_classifier_datasets,
    run_tracklet_classifier_frame_benchmark,
    run_tracklet_classifier_mixture_benchmark,
    train_tracklet_classifier,
    validate_tracklet_classifier_aot_eval_inputs,
    validate_tracklet_classifier_frame_benchmark_inputs,
    validate_tracklet_classifier_mixture_inputs,
)
from qstr_dronedet.tracking.video_action_policy import (
    attach_vatd_scores_to_tracklets,
    score_tracklets_with_ego_adaptive_vatd_policy,
    score_tracklets_with_vatd_motion_action_policy,
    score_tracklets_with_video_action_multihead_policy,
    score_tracklets_with_video_action_policy,
    train_ego_adaptive_vatd_policy,
    train_vatd_motion_action_policy,
    train_video_action_chunk_policy,
    train_video_action_multihead_policy,
)
from qstr_dronedet.types import CLASSES, AlignmentResult, DetectionCandidate, RecognitionResult, normalize_probs
from qstr_dronedet.visualization.draw import draw_overlay, make_side_by_side


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, AlignmentResult):
        d = asdict(obj)
        d["transform"] = _jsonable(obj.transform)
        return d
    if isinstance(obj, DetectionCandidate):
        return asdict(obj)
    if isinstance(obj, RecognitionResult):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "map"}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    return cap


def _limit_candidates(candidates: list[DetectionCandidate], limit: int | None) -> list[DetectionCandidate]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    return sorted(candidates, key=lambda c: c.objectness, reverse=True)[:limit]


def _budget_candidates(
    candidates: list[DetectionCandidate],
    nms_iou: float | None = None,
    top_k: int | None = None,
) -> list[DetectionCandidate]:
    out = list(candidates)
    if nms_iou is not None and nms_iou > 0:
        out = nms_candidates(out, iou_threshold=nms_iou)
    if top_k is not None and top_k > 0:
        out = _limit_candidates(out, top_k)
    return out


def _filter_candidates_by_box_size(
    candidates: list[DetectionCandidate],
    max_box_side: float | None = None,
    min_box_side: float | None = None,
) -> list[DetectionCandidate]:
    max_side_enabled = max_box_side is not None and max_box_side > 0
    min_side_enabled = min_box_side is not None and min_box_side > 0
    if not max_side_enabled and not min_side_enabled:
        return list(candidates)
    kept: list[DetectionCandidate] = []
    for cand in candidates:
        x1, y1, x2, y2 = cand.bbox_xyxy
        side = max(float(x2 - x1), float(y2 - y1))
        if max_side_enabled and side > float(max_box_side):
            continue
        if min_side_enabled and side < float(min_box_side):
            continue
        kept.append(cand)
    return kept


def _candidate_source_priority(candidate: DetectionCandidate) -> tuple[int, float]:
    source = candidate.source
    if "seed" in source:
        return (0, -candidate.objectness)
    if "track" in source:
        return (1, -candidate.objectness)
    if "yolo" in source:
        return (2, -candidate.objectness)
    if "motion" in source:
        return (3, -candidate.objectness)
    return (4, -candidate.objectness)


def _limit_merged_candidates(candidates: list[DetectionCandidate], limit: int | None) -> list[DetectionCandidate]:
    if limit is None or limit <= 0 or len(candidates) <= limit:
        return list(candidates)
    return sorted(candidates, key=_candidate_source_priority)[:limit]


def _run_yolo_candidates(
    frame: np.ndarray,
    weights: str | None,
    conf: float,
    tile_size: int,
    tile_stride: int,
    device: str | None,
    max_det: int,
    source_suffix: str | None = None,
) -> list[DetectionCandidate]:
    if tile_size and tile_size > 0:
        cands = candidates_from_yolo_tiled(
            frame,
            weights,
            tile_size=tile_size,
            stride=tile_stride,
            conf=conf,
            device=device,
            max_det=max_det,
        )
    else:
        cands = candidates_from_yolo(frame, weights, conf=conf)
    if source_suffix:
        for cand in cands:
            cand.extra["base_source"] = cand.source
            cand.source = f"{cand.source}_{source_suffix}"
    return cands


def _should_run_fallback_yolo(
    primary_candidates: list[DetectionCandidate],
    min_primary_candidates: int,
    trigger_objectness: float,
) -> bool:
    if len(primary_candidates) < min_primary_candidates:
        return True
    best = max((c.objectness for c in primary_candidates), default=0.0)
    return best < trigger_objectness


def _should_run_fallback_after_recognition(
    recognitions: list[RecognitionResult],
    trigger_final_score: float,
    primary_candidates: list[DetectionCandidate] | None = None,
    max_primary_objectness: float = 1.0,
) -> bool:
    if trigger_final_score <= 0:
        return False
    if primary_candidates and max_primary_objectness < 1.0:
        best_primary_objectness = max((c.objectness for c in primary_candidates), default=0.0)
        if best_primary_objectness > max_primary_objectness:
            return False
    best = max((r.final_drone_score for r in recognitions), default=0.0)
    return best < trigger_final_score


def _safe_default_probs(candidate: DetectionCandidate, tube: np.ndarray | None = None) -> dict[str, float]:
    probs = {c: 0.02 for c in CLASSES}
    probs["unknown"] = 0.55
    probs["background"] = 0.18
    if candidate.objectness > 0.45:
        probs["drone"] += 0.08
    if candidate.track_score > 0.55:
        probs["drone"] += 0.08
        probs["unknown"] -= 0.05
    if candidate.motion_score > 0.12 and candidate.alignment_quality > 0.3:
        probs["drone"] += 0.06
    if candidate.alignment_quality < 0.25 and candidate.motion_score > 0.15:
        probs["alignment_artifact"] += 0.18
    if candidate.mode == "static_or_hovering" and candidate.track_score > 0.6:
        probs["drone"] += 0.10
        probs["unknown"] -= 0.08
    return normalize_probs(probs)


def _load_crop_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.crop_recognizer import CropRecognizer

        ckpt = torch.load(weights, map_location="cpu")
        model = CropRecognizer()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load crop recognizer weights {weights}: {exc}")
        return None


def _load_feature_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.feature_recognizer import FeatureRecognitionModel

        ckpt = torch.load(weights, map_location="cpu")
        model = FeatureRecognitionModel()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load feature recognizer weights {weights}: {exc}")
        return None


def _load_temporal_model(weights: str | None):
    if not weights:
        return None
    try:
        import torch

        from qstr_dronedet.recognition.temporal_recognizer import TemporalRecognizer

        ckpt = torch.load(weights, map_location="cpu")
        model = TemporalRecognizer()
        model.load_state_dict(ckpt["state_dict"])
        model.qstr_target_mode = ckpt.get("target_mode", "multiclass")
        model.eval()
        return model
    except Exception as exc:
        print(f"Warning: could not load temporal recognizer weights {weights}: {exc}")
        return None


def _predict_crop_probs(model, crop_bgr: np.ndarray) -> dict[str, float]:
    try:
        import torch

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: crop recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _predict_feature_probs(model, frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> dict[str, float]:
    try:
        import torch

        img = cv2.resize(frame_bgr, (640, 640))
        sx = 640.0 / frame_bgr.shape[1]
        sy = 640.0 / frame_bgr.shape[0]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        box = torch.tensor([[bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x, box)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: feature recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _predict_temporal_probs(model, tube: np.ndarray) -> dict[str, float]:
    try:
        import torch

        x = torch.from_numpy(tube).unsqueeze(0).float()
        with torch.no_grad():
            logits = model(x)
            if getattr(model, "qstr_target_mode", "multiclass") == "drone_binary":
                p = logits[:, [CLASSES.index("drone"), CLASSES.index("background")]].softmax(1)[0].cpu().numpy().tolist()
                return normalize_probs({"drone": float(p[0]), "background": float(p[1])})
            probs = logits.softmax(1)[0].cpu().numpy().tolist()
        return normalize_probs({cls: float(probs[i]) for i, cls in enumerate(CLASSES)})
    except Exception as exc:
        print(f"Warning: temporal recognizer inference failed: {exc}")
        return normalize_probs({"unknown": 1.0})


def _read_video_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = _open_video(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, float(fps)


def cmd_motion_debug(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, _ = _read_video_frames(args.video)
    diag_path = out / "diagnostics.jsonl"
    with diag_path.open("w", encoding="utf-8") as log:
        for i in range(len(frames)):
            if i == 0:
                continue
            motion = compute_multik_motion(frames[: i + 1], i, tuple(args.k_values))
            motion_map = motion["motion_map"]
            cv2.imwrite(str(out / f"motion_{i:06d}.png"), motion_map)
            cands = candidates_from_motion(motion_map)
            for c in cands:
                c.alignment_quality = motion["best_quality"]
            overlay = draw_overlay(frames[i], cands)
            cv2.imwrite(str(out / f"overlay_{i:06d}.jpg"), make_side_by_side(frames[i], motion_map, overlay))
            row = {
                "frame_id": i,
                "best_k": motion["best_k"],
                "best_quality": motion["best_quality"],
                "per_k": {k: _jsonable(v) for k, v in motion["per_k"].items()},
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_infer(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "frames"
    vis_dir.mkdir(exist_ok=True)
    frames, fps = _read_video_frames(args.video)
    fusion_weights = _load_fusion_calibration(args.fusion_calibration)
    crop_model = _load_crop_model(args.crop_weights)
    feature_model = _load_feature_model(args.feature_weights)
    temporal_model = _load_temporal_model(args.temporal_weights)
    tracker = ConstantVelocityTracker()
    pred_f = (out / "predictions.jsonl").open("w", encoding="utf-8")
    diag_f = (out / "diagnostics.jsonl").open("w", encoding="utf-8")
    writer = None
    def prepare_candidates(raw_candidates: list[DetectionCandidate]) -> list[DetectionCandidate]:
        for cand in raw_candidates:
            cand.motion_score = max(cand.motion_score, motion_score_in_bbox(motion_map, cand.bbox_xyxy))
            cand.alignment_quality = float(motion["best_quality"])
        return _limit_merged_candidates(merge_candidates(raw_candidates), args.max_candidates_per_frame)

    def recognize_candidates(candidates: list[DetectionCandidate]) -> list[RecognitionResult]:
        recognitions: list[RecognitionResult] = []
        for cand in candidates:
            track_speed = tracker.compute_track_speed()
            track_conf = max(cand.track_score, tracker.track_confidence())
            blur_score = 0.0
            if motion["best_k"] is not None:
                aln = motion["per_k"][motion["best_k"]]["alignment"]
                blur_score = getattr(aln, "blur_score", 0.0) if aln is not None else 0.0
            cand.mode = determine_mode(cand.motion_score, cand.alignment_quality, track_speed, blur_score, track_conf)
            crop = crop_with_context(frame, cand.bbox_xyxy, scale=args.recognition_crop_scale)
            tube = extract_temporal_tube(frames[: i + 1], cand.bbox_xyxy, scale=args.recognition_tube_scale)
            crop_probs = _predict_crop_probs(crop_model, crop) if crop_model is not None else _safe_default_probs(cand, tube)
            feature_probs = _predict_feature_probs(feature_model, frame, cand.bbox_xyxy) if feature_model is not None else _safe_default_probs(cand, tube)
            temporal_probs = _predict_temporal_probs(temporal_model, tube) if temporal_model is not None else _safe_default_probs(cand, tube)
            if cand.mode == "static_or_hovering":
                temporal_probs["drone"] += 0.08
                temporal_probs = normalize_probs(temporal_probs)
            recognitions.append(
                fuse_rule_based(
                    cand.objectness,
                    crop_probs,
                    feature_probs,
                    temporal_probs,
                    cand.motion_score,
                    cand.alignment_quality,
                    cand.track_score,
                    cand.mode,
                candidate_source=cand.source,
                fusion_weights=fusion_weights,
                fallback_gate=not args.disable_fallback_gate,
                fallback_min_branch_drone=args.fallback_gate_min_branch_drone,
                fallback_min_crop_temporal_mean=args.fallback_gate_min_crop_temporal_mean,
                fallback_max_negative_evidence=args.fallback_gate_max_negative_evidence,
                verified_objectness=not args.disable_verified_objectness,
                verified_objectness_mode=args.verified_objectness_mode,
                verified_min_branch_drone=args.verified_min_branch_drone,
                verified_min_crop_temporal_mean=args.verified_min_crop_temporal_mean,
                verified_max_negative_evidence=args.verified_max_negative_evidence,
                verified_objectness_floor=args.verified_objectness_floor,
                hard_tiny_recovery=args.enable_hard_tiny_recovery,
                hard_tiny_min_crop_drone=args.hard_tiny_min_crop_drone,
                hard_tiny_min_temporal_drone=args.hard_tiny_min_temporal_drone,
                hard_tiny_min_temporal_crop_delta=args.hard_tiny_min_temporal_crop_delta,
                hard_tiny_max_bg_minus_drone=args.hard_tiny_max_bg_minus_drone,
                hard_tiny_min_support=args.hard_tiny_min_support,
                hard_tiny_score_floor=args.hard_tiny_score_floor,
                hard_tiny_allow_tracker_only=args.hard_tiny_allow_tracker_only,
                hard_tiny_require_validated_track=not args.hard_tiny_disable_track_validation,
                hard_tiny_max_track_frames_since_detector=args.hard_tiny_max_track_frames_since_detector,
                hard_tiny_min_track_detector_updates=args.hard_tiny_min_track_detector_updates,
                hard_tiny_max_track_drift=args.hard_tiny_max_track_drift,
                hard_tiny_min_track_history=args.hard_tiny_min_track_history,
                candidate_extra=cand.extra,
            )
        )
        return recognitions

    for i, frame in enumerate(frames):
        if args.max_frames is not None and i >= args.max_frames:
            break
        if i == 0:
            motion = {"motion_map": np.zeros(frame.shape[:2], np.uint8), "best_quality": 0.0, "best_k": None, "per_k": {}}
        else:
            motion = compute_multik_motion(frames[: i + 1], i, tuple(args.k_values))
        motion_map = motion["motion_map"]
        motion_cands = [] if args.disable_motion_candidates else candidates_from_motion(motion_map, min_area=args.min_area, max_area=args.max_area)
        yolo_cands = _run_yolo_candidates(
            frame,
            args.yolo_weights,
            args.yolo_conf,
            args.yolo_tile_size,
            args.yolo_tile_stride,
            args.yolo_device,
            args.yolo_max_det,
        )
        yolo_cands = _limit_candidates(yolo_cands, args.max_yolo_candidates_per_frame)
        fallback_ran = False
        if args.fallback_yolo_weights and _should_run_fallback_yolo(
            yolo_cands,
            args.fallback_min_primary_candidates,
            args.fallback_trigger_objectness,
        ):
            fallback_cands = _run_yolo_candidates(
                frame,
                args.fallback_yolo_weights,
                args.fallback_yolo_conf,
                args.fallback_yolo_tile_size or args.yolo_tile_size,
                args.fallback_yolo_tile_stride or args.yolo_tile_stride,
                args.yolo_device,
                args.yolo_max_det,
                source_suffix="fallback",
            )
            fallback_cands = _filter_candidates_by_box_size(
                fallback_cands,
                max_box_side=args.fallback_max_box_side,
                min_box_side=args.fallback_min_box_side,
            )
            fallback_cands = _limit_candidates(fallback_cands, args.max_fallback_yolo_candidates_per_frame)
            for cand in fallback_cands:
                cand.extra["fallback_reason"] = {
                    "primary_count": len(yolo_cands),
                    "primary_best_objectness": max((c.objectness for c in yolo_cands), default=0.0),
                    "min_primary_candidates": args.fallback_min_primary_candidates,
                    "trigger_objectness": args.fallback_trigger_objectness,
                }
            yolo_cands = yolo_cands + fallback_cands
            fallback_ran = True
        track_cands = tracker.get_track_candidates()
        seed_cands: list[DetectionCandidate] = []
        if args.seed_box is not None:
            seed_cands.append(
                DetectionCandidate(
                    bbox_xyxy=tuple(float(v) for v in args.seed_box),
                    objectness=float(args.seed_objectness),
                    source="seed",
                    track_score=float(args.seed_track_score),
                    extra={"oracle_seed": True},
                )
            )
        raw_candidates = motion_cands + yolo_cands + track_cands + seed_cands
        candidates = prepare_candidates(raw_candidates)
        recognitions = recognize_candidates(candidates)
        if args.fallback_yolo_weights and not fallback_ran and _should_run_fallback_after_recognition(
            recognitions,
            args.fallback_trigger_final_score,
            primary_candidates=yolo_cands,
            max_primary_objectness=args.fallback_post_trigger_max_primary_objectness,
        ):
            fallback_cands = _run_yolo_candidates(
                frame,
                args.fallback_yolo_weights,
                args.fallback_yolo_conf,
                args.fallback_yolo_tile_size or args.yolo_tile_size,
                args.fallback_yolo_tile_stride or args.yolo_tile_stride,
                args.yolo_device,
                args.yolo_max_det,
                source_suffix="fallback",
            )
            fallback_cands = _filter_candidates_by_box_size(
                fallback_cands,
                max_box_side=args.fallback_max_box_side,
                min_box_side=args.fallback_min_box_side,
            )
            fallback_cands = _limit_candidates(fallback_cands, args.max_fallback_yolo_candidates_per_frame)
            for cand in fallback_cands:
                cand.extra["fallback_reason"] = {
                    "primary_best_final_drone_score": max((r.final_drone_score for r in recognitions), default=0.0),
                    "trigger_final_score": args.fallback_trigger_final_score,
                    "primary_best_objectness": max((c.objectness for c in yolo_cands), default=0.0),
                    "post_trigger_max_primary_objectness": args.fallback_post_trigger_max_primary_objectness,
                }
            raw_candidates = raw_candidates + fallback_cands
            candidates = prepare_candidates(raw_candidates)
            recognitions = recognize_candidates(candidates)
            fallback_ran = True
        external_raw_candidates = [
            c for c in raw_candidates
            if c.source != "tracker" and not (c.source == "motion" and c.objectness < 0.15)
        ]
        merged_track_updates = [
            c for c in candidates
            if "tracker" in c.source and c.source != "tracker" and any(s in c.source for s in ("yolo", "fallback", "motion", "seed"))
        ]
        tracker_update_candidates = external_raw_candidates + merged_track_updates
        tracker.update(tracker_update_candidates, alignment_quality=float(motion["best_quality"]))
        for cand, rec in zip(candidates, recognitions):
            if cand.extra.get("track_id") is not None:
                tracker.update_evidence(cand.extra.get("track_id"), rec.crop_probs, rec.temporal_probs, rec.final_probs)
        for cand, rec in zip(candidates, recognitions):
            row = {
                "frame_id": i,
                "bbox": list(cand.bbox_xyxy),
                "objectness": cand.objectness,
                "source": cand.source,
                "motion_score": cand.motion_score,
                "alignment_quality": cand.alignment_quality,
                "track_score": cand.track_score,
                "track_id": cand.extra.get("track_id"),
                "track_age": cand.extra.get("track_age"),
                "track_history_len": cand.extra.get("track_history_len"),
                "track_detector_updates": cand.extra.get("track_detector_updates"),
                "track_last_detector_source": cand.extra.get("track_last_detector_source"),
                "track_frames_since_detector_update": cand.extra.get("track_frames_since_detector_update"),
                "track_drift": cand.extra.get("track_drift"),
                "track_validated": cand.extra.get("track_validated"),
                "track_evidence_len": cand.extra.get("track_evidence_len"),
                "track_crop_drone_mean": cand.extra.get("track_crop_drone_mean"),
                "track_temporal_drone_mean": cand.extra.get("track_temporal_drone_mean"),
                "track_background_mean": cand.extra.get("track_background_mean"),
                "track_temporal_gain_rate": cand.extra.get("track_temporal_gain_rate"),
                "track_recognition_confirmed": cand.extra.get("track_recognition_confirmed"),
                "mode": cand.mode,
                "final_drone_score": rec.final_drone_score,
                "predicted_class": rec.predicted_class,
                "diagnostic_cause": rec.diagnostic_cause,
                "final_probs": rec.final_probs,
                "fallback_yolo_ran": fallback_ran,
            }
            pred_f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
            diag = {**row, "crop_probs": rec.crop_probs, "feature_probs": rec.feature_probs, "temporal_probs": rec.temporal_probs, "disagreement": rec.disagreement, "error_type": rec.error_type}
            diag_f.write(json.dumps(_jsonable(diag), ensure_ascii=False) + "\n")
        overlay = draw_overlay(frame, candidates, recognitions)
        side = make_side_by_side(frame, motion_map, overlay)
        cv2.imwrite(str(vis_dir / f"frame_{i:06d}.jpg"), side)
        if args.save_video:
            if writer is None:
                h, w = side.shape[:2]
                writer = cv2.VideoWriter(str(out / "visualization.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(side)
    if writer is not None:
        writer.release()
    pred_f.close()
    diag_f.close()
    if args.tracklet_classifier_weights:
        summary = apply_tracklet_filter_to_infer_outputs(
            out / "predictions.jsonl",
            out / "diagnostics.jsonl",
        args.tracklet_classifier_weights,
        threshold=args.tracklet_classifier_threshold,
        untracked_policy=args.tracklet_filter_untracked,
        promote_positive_tracklets=not args.disable_tracklet_promotion,
        promotion_score_floor=args.tracklet_promotion_score_floor,
        promotion_min_branch_drone=args.tracklet_promotion_min_branch_drone,
        promotion_max_background=args.tracklet_promotion_max_background,
        selective_promotion=args.tracklet_selective_promotion,
        selective_min_temporal_crop_delta=args.tracklet_selective_min_temporal_crop_delta,
        selective_min_temporal_background_margin=args.tracklet_selective_min_temporal_background_margin,
        selective_max_tracklet_background=args.tracklet_selective_max_tracklet_background,
        selective_max_tracklet_objectness=args.tracklet_selective_max_tracklet_objectness,
        selective_min_tracklet_rows=args.tracklet_selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=args.tracklet_selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=args.tracklet_selective_min_weak_detector_temporal_signal,
        selective_require_recovery_source=not args.tracklet_selective_allow_non_recovery_source,
        selective_max_promoted_tracklets_per_sequence=args.tracklet_selective_max_promoted_tracklets_per_sequence,
    )
        print(json.dumps(summary, indent=2))


def cmd_offline_selector_replay(args: argparse.Namespace) -> None:
    result = replay_offline_selector(
        args.predictions,
        args.out,
        top_k=args.top_k,
        max_frame_id=args.max_frame_id,
        max_jump_px=args.max_jump_px,
        max_recover_frames=args.max_recover_frames,
        min_accept_score=args.min_accept_score,
        min_motion_consistency=args.min_motion_consistency,
        min_memory_consistency=args.min_memory_consistency,
        min_topk_margin=args.min_topk_margin,
        detector_weight=args.detector_weight,
        motion_weight=args.motion_weight,
        memory_weight=args.memory_weight,
        reacquire_top_k=args.reacquire_top_k,
        reacquire_max_distance_px=args.reacquire_max_distance_px,
        reacquire_min_detector_score=args.reacquire_min_detector_score,
        reacquire_min_motion_consistency=args.reacquire_min_motion_consistency,
        reacquire_min_memory_consistency=args.reacquire_min_memory_consistency,
        reacquire_min_side_ratio=args.reacquire_min_side_ratio,
        reacquire_max_side_ratio=args.reacquire_max_side_ratio,
        reacquire_confirm_frames=args.reacquire_confirm_frames,
        reacquire_stale_after_frames=args.reacquire_stale_after_frames,
        reacquire_global_min_area=args.reacquire_global_min_area,
        reacquire_global_small_min_area=args.reacquire_global_small_min_area,
        reacquire_global_small_max_distance_px=args.reacquire_global_small_max_distance_px,
        reacquire_global_small_memory_probation_frames=args.reacquire_global_small_memory_probation_frames,
        reacquire_global_small_repeat_cooldown_frames=args.reacquire_global_small_repeat_cooldown_frames,
        reacquire_global_small_repeat_max_distance_px=args.reacquire_global_small_repeat_max_distance_px,
        reacquire_global_max_area=args.reacquire_global_max_area,
        reacquire_global_min_detector_score=args.reacquire_global_min_detector_score,
        reacquire_global_require_tracklet_confirmation=args.reacquire_global_require_tracklet_confirmation,
        reacquire_global_reject_tracklet_rejected=args.reacquire_global_reject_tracklet_rejected,
        reacquire_global_min_tracklet_score=args.reacquire_global_min_tracklet_score,
        reacquire_global_delayed_confirm_frames=args.reacquire_global_delayed_confirm_frames,
        reacquire_min_appearance_similarity=args.reacquire_min_appearance_similarity,
        reacquire_appearance_weight=args.reacquire_appearance_weight,
        reacquire_crop_score_field=args.reacquire_crop_score_field,
        reacquire_crop_weight=args.reacquire_crop_weight,
        reacquire_min_crop_drone_score=args.reacquire_min_crop_drone_score,
        reacquire_crop_weights=args.reacquire_crop_weights,
        reacquire_crop_image_size=args.reacquire_crop_image_size,
        appearance_memory_size=args.appearance_memory_size,
        video=args.video,
        save_video=args.save_video,
    )
    print(
        json.dumps(
            {
                "trajectory_csv": str(result.trajectory_csv),
                "debug_csv": str(result.debug_csv),
                "summary_json": str(result.summary_json),
                "annotated_video": str(result.annotated_video) if result.annotated_video else None,
                **result.summary,
            },
            indent=2,
        )
    )


def cmd_build_seed_admission_dataset(args: argparse.Namespace) -> None:
    result = build_seed_admission_dataset(
        args.seed_audit,
        args.annotations,
        args.out,
        dataset_source=args.dataset_source,
        run_id=args.run_id,
        iou_threshold=args.iou_threshold,
        center_distance_threshold_px=args.center_distance_threshold_px,
        interpolate_gt=not args.no_interpolate_gt,
    )
    print(json.dumps(result.summary, indent=2))


def _load_fusion_calibration(path: str | None) -> dict[str, float] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "weights" in data:
        weights = data["weights"]
    else:
        best = data.get("best") or []
        if not best:
            raise ValueError(f"No fusion weights found in calibration file: {path}")
        weights = best[0]["weights"]
    return {str(k): float(v) for k, v in weights.items()}


def _iter_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_build_crop_dataset(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = _iter_csv_rows(Path(args.annotations))
    for row in rows:
        cls = row.get("class", "unknown")
        dst_dir = out / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        frame_path = Path(row["frame_path"])
        if not frame_path.is_absolute() and args.frames:
            frame_path = Path(args.frames) / frame_path
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        bbox = tuple(float(row[k]) for k in ["x1", "y1", "x2", "y2"])
        crop = crop_with_context(frame, bbox, scale=args.scale, out_size=args.size)
        cv2.imwrite(str(dst_dir / f"{frame_path.stem}_{len(list(dst_dir.glob('*.jpg'))):06d}.jpg"), crop)


def cmd_train_recognizer(args: argparse.Namespace) -> None:
    if args.type == "crop":
        train_crop_recognizer(args.data, args.out, args.epochs, args.balance, args.target_mode, args.pretrained)
    elif args.type == "temporal":
        train_temporal_recognizer(args.data, args.out, args.epochs, args.balance, args.target_mode, args.pretrained)
    elif args.type == "feature":
        if args.pretrained:
            raise ValueError("--pretrained is currently supported for crop and temporal recognizers only")
        train_feature_recognizer(args.data, args.out, args.epochs, args.target_mode)
    else:
        raise ValueError(f"Unsupported recognizer type: {args.type}")


def cmd_build_yolo_candidate_dataset(args: argparse.Namespace) -> None:
    data_yaml = build_class_agnostic_yolo_dataset(args.annotations, args.out, args.images_root, args.val_fraction, args.seed, args.min_box_px)
    print(f"Wrote class-agnostic YOLO dataset: {data_yaml}")


def cmd_build_tiled_yolo_candidate_dataset(args: argparse.Namespace) -> None:
    data_yaml = build_tiled_class_agnostic_yolo_dataset(
        args.annotations,
        args.out,
        images_root=args.images_root,
        tile_size=args.tile_size,
        positives_per_box=args.positives_per_box,
        negatives_per_image=args.negatives_per_image,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_box_px=args.min_box_px,
        negative_pad_px=args.negative_pad_px,
        photometric_augmentations=args.photometric_augmentations,
        low_contrast_injections=args.low_contrast_injections,
        positive_repeat_patterns=tuple(args.positive_repeat_pattern or ()),
        positive_repeat_factor=args.positive_repeat_factor,
    )
    print(f"Wrote tiled class-agnostic YOLO dataset: {data_yaml}")


def cmd_train_yolo_p2(args: argparse.Namespace) -> None:
    model_yaml = args.model_yaml
    if args.write_model_yaml:
        model_yaml = str(write_yolov8_p2_model_yaml(args.write_model_yaml))
        print(f"Wrote YOLO-P2 model yaml: {model_yaml}")
        if args.no_train:
            return
    run_dir = train_yolo_p2(args.data, args.out, model_yaml, args.pretrained, args.epochs, args.imgsz, args.batch, args.device)
    print(f"YOLO-P2 candidate training run: {run_dir}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    metrics = evaluate_predictions(args.pred, args.gt, args.out)
    print(json.dumps(metrics, indent=2))


def cmd_build_tracklet_dataset(args: argparse.Namespace) -> None:
    result = build_tracklet_dataset(
        args.diagnostics,
        args.gt_csv,
        args.out,
        max_frames=args.max_frames,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_build_proposal_tracklet_dataset(args: argparse.Namespace) -> None:
    result = build_proposal_tracklet_dataset(
        args.run_roots,
        args.gt_csv,
        args.out,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        max_frames=args.max_frames,
        max_gap=args.max_gap,
        base_radius=args.base_radius,
        radius_per_side=args.radius_per_side,
        min_iou=args.min_iou,
        min_score=args.min_score,
        detector_only=args.detector_only,
        min_tracklet_rows=args.min_tracklet_rows,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
        hard_tiny_side=args.hard_tiny_side,
        hard_low_score=args.hard_low_score,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_export_yolo_oracle_tracklets(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = export_yolo_oracle_tracklets(
        args.list_files,
        args.out,
        dataset_source=args.dataset_source,
        image_size=image_size,
        max_images=args.max_images,
        skip_images=args.skip_images,
        max_labeled_images_per_seq=args.max_labeled_images_per_seq,
        max_gap=args.max_gap,
        base_radius=args.base_radius,
        radius_per_side=args.radius_per_side,
        min_iou=args.min_iou,
        min_tracklet_rows=args.min_tracklet_rows,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_export_temporal_saliency_tracklets(args: argparse.Namespace) -> None:
    result = export_temporal_saliency_tracklets(
        args.list_files,
        args.gt_csv,
        args.out,
        dataset_source=args.dataset_source,
        max_images=args.max_images,
        threshold=args.threshold,
        min_area=args.min_area,
        max_area=args.max_area,
        dilate_iters=args.dilate_iters,
        max_gap=args.max_gap,
        base_radius=args.base_radius,
        radius_per_side=args.radius_per_side,
        min_iou=args.min_iou,
        min_tracklet_rows=args.min_tracklet_rows,
        iou_threshold=args.iou_threshold,
        center_threshold=args.center_threshold,
        hard_tiny_side=args.hard_tiny_side,
        hard_low_score=args.hard_low_score,
        progress_every_sequences=args.progress_every_sequences,
    )
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_export_frame_list_from_gt_csv(args: argparse.Namespace) -> None:
    result = export_frame_list_from_gt_csv(
        args.gt_csv,
        args.frame_root,
        args.out,
        extensions=args.extensions,
        recursive=args.recursive,
        max_frames=args.max_frames,
        max_frames_per_seq=args.max_frames_per_seq,
    )
    print(json.dumps({"list_file": str(result.out_path), **result.summary}, indent=2))


def cmd_export_tracklet_jsonl_predictions(args: argparse.Namespace) -> None:
    nms_iou_threshold = None if args.nms_iou_threshold < 0 else args.nms_iou_threshold
    nms_center_threshold = None if args.nms_center_threshold < 0 else args.nms_center_threshold
    result = export_tracklet_jsonl_predictions(
        args.tracklet_jsonl,
        args.out_dir,
        dataset_name=args.dataset_name,
        score_field=args.score_field,
        min_score=args.min_score,
        class_id=args.class_id,
        formats=args.formats,
        nms_iou_threshold=nms_iou_threshold,
        nms_center_threshold=nms_center_threshold,
    )
    print(json.dumps({"summary": str(result.out_path), **result.summary}, indent=2))


def cmd_evaluate_flat_tracklet_predictions(args: argparse.Namespace) -> None:
    center_threshold = None if args.center_threshold < 0 else args.center_threshold
    result = evaluate_flat_tracklet_predictions(
        args.gt_csv,
        args.prediction_csv,
        args.out_dir,
        thresholds=args.thresholds,
        iou_threshold=args.iou_threshold,
        center_threshold=center_threshold,
        fp_limit=args.fp_limit,
        max_fppi=args.max_fppi,
        fp_limits=args.fp_limits,
        max_fppis=args.max_fppis,
    )
    print(json.dumps({"csv": str(result.csv_path), "summary": str(result.summary_path), **result.summary}, indent=2))


def _optional_float_values(values: list[str] | None) -> list[float | None] | None:
    if values is None:
        return None
    out: list[float | None] = []
    for value in values:
        text = str(value).strip().lower()
        if text in {"none", "null", "off", "-1"}:
            out.append(None)
        else:
            out.append(float(value))
    return out


def cmd_sweep_flat_tracklet_prediction_nms(args: argparse.Namespace) -> None:
    eval_center_threshold = None if args.eval_center_threshold < 0 else args.eval_center_threshold
    result = sweep_flat_tracklet_prediction_nms(
        args.tracklet_jsonl,
        args.gt_csv,
        args.out_dir,
        dataset_name=args.dataset_name,
        score_field=args.score_field,
        min_score=args.min_score,
        class_id=args.class_id,
        iou_thresholds=_optional_float_values(args.nms_iou_thresholds),
        center_thresholds=_optional_float_values(args.nms_center_thresholds),
        score_thresholds=args.score_thresholds,
        eval_iou_threshold=args.eval_iou_threshold,
        eval_center_threshold=eval_center_threshold,
        fp_limit=args.fp_limit,
        max_fppi=args.max_fppi,
        fp_limits=args.fp_limits,
        max_fppis=args.max_fppis,
    )
    print(json.dumps({"csv": str(result.csv_path), "summary": str(result.summary_path), **result.summary}, indent=2))


def cmd_compare_flat_prediction_eval_summaries(args: argparse.Namespace) -> None:
    result = compare_flat_prediction_eval_summaries(
        args.summaries,
        args.out_dir,
        method_names=args.method_names,
    )
    print(
        json.dumps(
            {"csv": str(result.csv_path), "markdown": str(result.markdown_path), "summary": str(result.summary_path), **result.summary},
            indent=2,
        )
    )


def cmd_export_yolo_labels_gt_csv(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = export_yolo_labels_to_gt_csv(
        args.list_files,
        args.out,
        image_size=image_size,
        max_images=args.max_images,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_export_yolo_predictions_route_b_run(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = export_yolo_predictions_to_route_b_run(
        args.list_files,
        args.pred_label_dir,
        args.out_run_root,
        image_size=image_size,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        source=args.source,
        max_images=args.max_images,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))


def cmd_merge_tracklet_jsonl(args: argparse.Namespace) -> None:
    result = merge_tracklet_jsonl(args.inputs, args.out, source_names=args.source_names)
    print(json.dumps({"jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_build_action_chunk_dataset(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = export_action_chunk_dataset_from_tracklets(
        args.tracklet_jsonl,
        args.out,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        normalize_by_row_image_size=args.normalize_by_row_image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
    )
    print(json.dumps({"jsonl": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_merge_action_chunk_datasets(args: argparse.Namespace) -> None:
    result = merge_action_chunk_datasets(
        args.inputs,
        args.out,
        source_names=args.source_names,
        manifest_out=args.manifest_out,
    )
    print(json.dumps({"jsonl": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_split_action_chunk_dataset(args: argparse.Namespace) -> None:
    result = split_action_chunk_dataset(
        args.jsonl,
        args.out_dir,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_field=args.group_field,
        source_field=args.source_field,
    )
    print(json.dumps({"train_jsonl": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_train_action_chunk_policy(args: argparse.Namespace) -> None:
    out = train_action_chunk_policy(
        args.jsonl,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        model_type=args.model_type,
        diffusion_steps=args.diffusion_steps,
    )
    print(f"Wrote action-chunk policy: {out}")


def cmd_eval_action_chunk_policy(args: argparse.Namespace) -> None:
    result = evaluate_action_chunk_policy(args.jsonl, args.weights, args.out)
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_export_action_prior_heatmaps(args: argparse.Namespace) -> None:
    result = export_action_prior_heatmaps_from_sample_scores(
        args.sample_scores,
        args.out_dir,
        image_size=(args.image_width, args.image_height),
        sigma_scale=args.sigma_scale,
        min_sigma=args.min_sigma,
        box_field=args.box_field,
        split_horizon=args.split_horizon,
    )
    print(json.dumps({"manifest": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_build_frame_prior_index(args: argparse.Namespace) -> None:
    result = build_frame_prior_index_from_heatmaps(args.prior_manifest, args.out_dir, merge_mode=args.merge_mode)
    print(json.dumps({"index": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_attach_frame_priors(args: argparse.Namespace) -> None:
    result = attach_frame_priors_to_tracklets(args.tracklet_jsonl, args.frame_prior_index, args.out)
    print(json.dumps({"jsonl": str(result.jsonl_path), **result.summary}, indent=2))


def cmd_fuse_action_frame_priors(args: argparse.Namespace) -> None:
    promote_threshold = None if args.promote_threshold is not None and args.promote_threshold < 0 else args.promote_threshold
    result = fuse_action_frame_prior_predictions(
        args.pred_jsonl,
        args.out,
        prior_weight=args.prior_weight,
        min_prior_score=args.min_prior_score,
        promote_threshold=promote_threshold,
        min_base_score_for_promotion=args.min_base_score_for_promotion,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_sweep_action_frame_priors(args: argparse.Namespace) -> None:
    promote_thresholds = [None if value < 0 else value for value in args.promote_thresholds]
    result = sweep_action_frame_prior_fusion(
        args.pred_jsonl,
        args.gt_csv,
        args.out_dir,
        prior_weights=args.prior_weights,
        min_prior_scores=args.min_prior_scores,
        promote_thresholds=promote_thresholds,
        min_base_scores_for_promotion=args.min_base_scores_for_promotion,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_sweep_action_frame_priors_run_root(args: argparse.Namespace) -> None:
    promote_thresholds = [None if value < 0 else value for value in args.promote_thresholds]
    result = sweep_action_frame_prior_fusion_run_root(
        args.run_roots,
        args.gt_csv,
        args.out_dir,
        profile=args.profile,
        prediction_name=args.prediction_name,
        prior_weights=args.prior_weights,
        min_prior_scores=args.min_prior_scores,
        promote_thresholds=promote_thresholds,
        min_base_scores_for_promotion=args.min_base_scores_for_promotion,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_action_policy_ablation(args: argparse.Namespace) -> None:
    result = run_action_policy_ablation(
        args.jsonl,
        args.out_dir,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_action_policy_split_selection(args: argparse.Namespace) -> None:
    result = run_action_policy_split_selection(
        args.train_jsonl,
        args.calib_jsonl,
        args.out_dir,
        test_jsonl=args.test_jsonl,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_multisource_action_policy_experiment(args: argparse.Namespace) -> None:
    result = run_multisource_action_policy_experiment(
        args.inputs,
        args.out_dir,
        source_names=args.source_names,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_field=args.group_field,
        source_field=args.source_field,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_multisource_tracklet_action_policy_experiment(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = run_multisource_tracklet_action_policy_experiment(
        args.tracklet_inputs,
        args.out_dir,
        source_names=args.source_names,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_field=args.group_field,
        source_field=args.source_field,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_multisource_tracklet_policy_benchmark(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = run_multisource_tracklet_policy_benchmark(
        args.train_tracklet_inputs,
        args.eval_tracklet_inputs,
        args.out_dir,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_field=args.group_field,
        source_field=args.source_field,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
        error_scale=args.error_scale,
        thresholds=args.thresholds,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        baseline_lower_is_better=args.baseline_lower_is_better,
        baseline_digits=args.baseline_digits,
        allow_invalid_baselines=args.allow_invalid_baselines,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_run_multisource_proposal_policy_benchmark(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = run_multisource_proposal_policy_benchmark(
        args.train_run_roots,
        args.train_gt_csvs,
        args.eval_run_roots,
        args.eval_gt_csvs,
        args.out_dir,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        max_frames=args.max_frames,
        proposal_max_gap=args.proposal_max_gap,
        proposal_base_radius=args.proposal_base_radius,
        proposal_radius_per_side=args.proposal_radius_per_side,
        proposal_min_iou=args.proposal_min_iou,
        proposal_min_score=args.proposal_min_score,
        proposal_detector_only=args.proposal_detector_only,
        proposal_min_tracklet_rows=args.proposal_min_tracklet_rows,
        proposal_iou_threshold=args.proposal_iou_threshold,
        proposal_center_threshold=args.proposal_center_threshold,
        proposal_hard_tiny_side=args.proposal_hard_tiny_side,
        proposal_hard_low_score=args.proposal_hard_low_score,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
        calib_fraction=args.calib_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_field=args.group_field,
        source_field=args.source_field,
        model_types=args.model_types,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
        error_scale=args.error_scale,
        thresholds=args.thresholds,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        baseline_lower_is_better=args.baseline_lower_is_better,
        baseline_digits=args.baseline_digits,
        allow_invalid_baselines=args.allow_invalid_baselines,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_validate_route_b_tracklet_inputs(args: argparse.Namespace) -> None:
    result = validate_route_b_tracklet_inputs(
        args.train_tracklet_inputs,
        args.eval_tracklet_inputs,
        args.out,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        past_len=args.past_len,
        future_len=args.future_len,
        min_tracklet_rows=args.min_tracklet_rows,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if args.strict and not result.summary["valid"]:
        raise SystemExit(2)


def cmd_validate_route_b_proposal_inputs(args: argparse.Namespace) -> None:
    result = validate_route_b_proposal_inputs(
        args.train_run_roots,
        args.train_gt_csvs,
        args.eval_run_roots,
        args.eval_gt_csvs,
        args.out,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        max_frames=args.max_frames,
        min_bbox_rows=args.min_bbox_rows,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if args.strict and not result.summary["valid"]:
        raise SystemExit(2)


def cmd_scan_route_b_proposal_inputs(args: argparse.Namespace) -> None:
    result = scan_route_b_proposal_inputs(
        args.scan_roots,
        args.out,
        profiles=args.profiles,
        diagnostics_names=args.diagnostics_names,
        max_depth=args.max_depth,
        max_files=args.max_files,
        max_diag_sample_files=args.max_diag_sample_files,
        max_rows_per_diag_file=args.max_rows_per_diag_file,
        max_frames=args.max_frames,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))


def cmd_write_route_b_proposal_run_manifest(args: argparse.Namespace) -> None:
    result = write_route_b_proposal_run_manifest(
        args.out_dir,
        args.train_run_roots,
        args.train_gt_csvs,
        args.eval_run_roots,
        args.eval_gt_csvs,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        run_id=args.run_id,
        benchmark_out_dir=args.benchmark_out_dir,
        runner_output_root=args.runner_output_root,
        profile=args.profile,
        diagnostics_name=args.diagnostics_name,
        max_frames=args.max_frames,
        past_len=args.past_len,
        future_len=args.future_len,
        model_types=args.model_types,
        epochs=args.epochs,
        hidden=args.hidden,
        batch_size=args.batch_size,
        thresholds=args.thresholds,
        balance_by=args.balance_by,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        baseline_lower_is_better=args.baseline_lower_is_better,
        validate_inputs=not args.skip_validation,
        preflight_min_bbox_rows=args.preflight_min_bbox_rows,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))


def cmd_score_action_chunk_tracklets(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    result = score_tracklets_with_action_policy(
        args.tracklet_jsonl,
        args.weights,
        args.out,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        normalize_by_row_image_size=args.normalize_by_row_image_size,
        error_scale=args.error_scale,
        dynamics_score_mode=args.dynamics_score_mode,
        min_tracklet_rows=args.min_tracklet_rows,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_score_constant_velocity_tracklets(args: argparse.Namespace) -> None:
    result = score_tracklets_with_constant_velocity(
        args.tracklet_jsonl,
        args.out,
        min_tracklet_rows=args.min_tracklet_rows,
        error_scale=args.error_scale,
        min_box_side=args.min_box_side,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_train_video_action_chunk_policy(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    out = train_video_action_chunk_policy(
        args.tracklet_jsonl,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        past_len=args.past_len,
        future_len=args.future_len,
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        image_size=image_size,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        allow_missing_images=args.allow_missing_images,
        verbose=not args.quiet,
    )
    print(json.dumps({"weights": str(out)}, indent=2))


def cmd_score_video_action_tracklets(args: argparse.Namespace) -> None:
    result = score_tracklets_with_video_action_policy(
        args.tracklet_jsonl,
        args.weights,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        error_scale=args.error_scale,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        allow_missing_images=args.allow_missing_images,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_train_video_action_multihead_policy(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    out = train_video_action_multihead_policy(
        args.tracklet_jsonl,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        past_len=args.past_len,
        future_len=args.future_len,
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        image_size=image_size,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        confidence_target=args.confidence_target,
        confidence_loss_weight=args.confidence_loss_weight,
        allow_missing_images=args.allow_missing_images,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        verbose=not args.quiet,
    )
    print(json.dumps({"weights": str(out)}, indent=2))


def cmd_score_video_action_multihead_tracklets(args: argparse.Namespace) -> None:
    result = score_tracklets_with_video_action_multihead_policy(
        args.tracklet_jsonl,
        args.weights,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        error_scale=args.error_scale,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        fusion_mode=args.fusion_mode,
        allow_missing_images=args.allow_missing_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        use_crops=not args.disable_crops,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_train_vatd_motion_action_policy(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    out = train_vatd_motion_action_policy(
        args.tracklet_jsonl,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        past_len=args.past_len,
        future_len=args.future_len,
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        image_size=image_size,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        action_loss_weight=args.action_loss_weight,
        allow_missing_images=args.allow_missing_images,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        motion_pos_weight=args.motion_pos_weight,
        pin_memory=not args.no_pin_memory,
        shuffle=not args.no_shuffle,
        use_crops=not args.disable_crops,
        verbose=not args.quiet,
    )
    print(json.dumps({"weights": str(out)}, indent=2))


def cmd_score_vatd_motion_action_tracklets(args: argparse.Namespace) -> None:
    result = score_tracklets_with_vatd_motion_action_policy(
        args.tracklet_jsonl,
        args.weights,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        error_scale=args.error_scale,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        fusion_mode=args.fusion_mode,
        allow_missing_images=args.allow_missing_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        use_crops=not args.disable_crops,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_train_ego_adaptive_vatd_policy(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    out = train_ego_adaptive_vatd_policy(
        args.tracklet_jsonl,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        past_len=args.past_len,
        future_len=args.future_len,
        horizons=tuple(args.horizons),
        crop_size=args.crop_size,
        crop_scale=args.crop_scale,
        image_size=image_size,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        action_loss_weight=args.action_loss_weight,
        allow_missing_images=args.allow_missing_images,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        motion_pos_weight=args.motion_pos_weight,
        pin_memory=not args.no_pin_memory,
        shuffle=not args.no_shuffle,
        use_crops=not args.disable_crops,
        verbose=not args.quiet,
    )
    print(json.dumps({"weights": str(out)}, indent=2))


def cmd_score_ego_adaptive_vatd_tracklets(args: argparse.Namespace) -> None:
    result = score_tracklets_with_ego_adaptive_vatd_policy(
        args.tracklet_jsonl,
        args.weights,
        args.out,
        frame_root=args.frame_root,
        image_name_template=args.image_name_template,
        error_scale=args.error_scale,
        min_tracklet_rows=args.min_tracklet_rows,
        max_samples=args.max_samples,
        fusion_mode=args.fusion_mode,
        allow_missing_images=args.allow_missing_images,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_cache_size=args.frame_cache_size,
        use_crops=not args.disable_crops,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_attach_vatd_scores_to_tracklets(args: argparse.Namespace) -> None:
    result = attach_vatd_scores_to_tracklets(
        args.tracklet_jsonl,
        args.vatd_scores,
        args.out,
        score_field=args.score_field,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_attach_action_dynamics_scores(args: argparse.Namespace) -> None:
    result = attach_action_dynamics_scores_to_tracklets(args.tracklet_jsonl, args.dynamics_scores, args.out)
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_attach_tracklet_confidence_fusion_scores(args: argparse.Namespace) -> None:
    result = attach_tracklet_confidence_fusion_scores(
        args.tracklet_jsonl,
        args.out,
        action_score_field=args.action_score_field,
        confidence_fields=tuple(args.confidence_fields),
        confidence_reduction=args.confidence_reduction,
        out_score_field=args.out_score_field,
        missing_action_score=args.missing_action_score,
    )
    print(json.dumps({"jsonl": str(result.out_path), **result.summary}, indent=2))


def cmd_run_action_dynamics_pipeline(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    prior_image_size = None
    if args.prior_image_width is not None or args.prior_image_height is not None:
        if args.prior_image_width is None or args.prior_image_height is None:
            raise ValueError("--prior-image-width and --prior-image-height must be provided together")
        prior_image_size = (args.prior_image_width, args.prior_image_height)
    result = run_action_dynamics_tracklet_pipeline(
        args.tracklet_jsonl,
        args.out_dir,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        normalize_by_row_image_size=args.normalize_by_row_image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        error_scale=args.error_scale,
        dynamics_score_mode=args.dynamics_score_mode,
        thresholds=args.thresholds,
        balance_by=args.balance_by,
        model_type=args.model_type,
        diffusion_steps=args.diffusion_steps,
        prior_image_size=prior_image_size,
        prior_sigma_scale=args.prior_sigma_scale,
        prior_min_sigma=args.prior_min_sigma,
        prior_split_horizon=args.prior_split_horizon,
        prior_merge_mode=args.prior_merge_mode,
    )
    print(json.dumps({"attached_tracklets": str(result.out_path), **result.summary}, indent=2))


def cmd_run_action_dynamics_ablation(args: argparse.Namespace) -> None:
    image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        image_size = (args.image_width, args.image_height)
    prior_image_size = None
    if args.prior_image_width is not None or args.prior_image_height is not None:
        if args.prior_image_width is None or args.prior_image_height is None:
            raise ValueError("--prior-image-width and --prior-image-height must be provided together")
        prior_image_size = (args.prior_image_width, args.prior_image_height)
    result = run_action_dynamics_tracklet_ablation(
        args.tracklet_jsonl,
        args.out_dir,
        model_types=args.model_types,
        past_len=args.past_len,
        future_len=args.future_len,
        image_size=image_size,
        normalize_by_row_image_size=args.normalize_by_row_image_size,
        positives_only=args.positives_only,
        min_tracklet_rows=args.min_tracklet_rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        batch_size=args.batch_size,
        error_scale=args.error_scale,
        dynamics_score_mode=args.dynamics_score_mode,
        thresholds=args.thresholds,
        balance_by=args.balance_by,
        diffusion_steps=args.diffusion_steps,
        prior_image_size=prior_image_size,
        prior_sigma_scale=args.prior_sigma_scale,
        prior_min_sigma=args.prior_min_sigma,
        prior_split_horizon=args.prior_split_horizon,
        prior_merge_mode=args.prior_merge_mode,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_collect_route_b_results(args: argparse.Namespace) -> None:
    result = collect_route_b_result_summaries(args.summaries, args.out_dir, dataset_names=args.dataset_names)
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_compare_route_b_baselines(args: argparse.Namespace) -> None:
    result = compare_route_b_results_to_baselines(
        args.route_b_csv,
        args.baseline_csv,
        args.out_dir,
        metric=args.metric,
        higher_is_better=not args.lower_is_better,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_write_route_b_baseline_template(args: argparse.Namespace) -> None:
    result = write_route_b_baseline_template(args.out, datasets=args.datasets, methods=args.methods)
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_write_route_b_official_baseline_seed(args: argparse.Namespace) -> None:
    result = write_route_b_official_baseline_seed(args.out, include_placeholders=not args.no_placeholders)
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_validate_route_b_baselines(args: argparse.Namespace) -> None:
    result = validate_route_b_baseline_csv(
        args.baseline_csv,
        args.out,
        metric=args.metric,
        require_metric_values=not args.allow_empty_metric,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if args.strict and not result.summary["valid"]:
        raise SystemExit(2)


def cmd_export_route_b_baseline_table(args: argparse.Namespace) -> None:
    result = export_route_b_baseline_markdown_table(args.comparison_summary, args.out, digits=args.digits)
    print(json.dumps({"markdown": str(result.out_path), **result.summary}, indent=2))


def cmd_build_route_b_report(args: argparse.Namespace) -> None:
    result = build_route_b_baseline_report(
        args.summaries,
        args.baseline_csv,
        args.out_dir,
        dataset_names=args.dataset_names,
        metric=args.metric,
        higher_is_better=not args.lower_is_better,
        digits=args.digits,
        strict_baselines=not args.allow_invalid_baselines,
    )
    print(json.dumps({"out": str(result.out_path), **result.summary}, indent=2))
    if result.summary.get("valid") is False and not args.allow_invalid_baselines:
        raise SystemExit(2)


def cmd_eval_action_dynamics_thresholds(args: argparse.Namespace) -> None:
    result = evaluate_action_dynamics_thresholds(args.dynamics_scores, args.out_dir, thresholds=args.thresholds)
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_export_tracklet_jsonl_classifier_dataset(args: argparse.Namespace) -> None:
    result = export_tracklet_jsonl_classifier_dataset(args.tracklet_jsonl, args.out, dataset_source=args.dataset_source)
    print(json.dumps({"csv": str(result.csv_path), "jsonl": str(result.json_path), **result.summary}, indent=2))


def cmd_merge_tracklet_classifier_datasets(args: argparse.Namespace) -> None:
    result = merge_tracklet_classifier_datasets(
        args.inputs,
        args.out,
        source_names=args.source_names,
        manifest_out=args.manifest_out,
    )
    print(json.dumps({"csv": str(result.csv_path), "manifest": str(result.manifest_path), **result.summary}, indent=2))


def cmd_validate_tracklet_classifier_mixture_inputs(args: argparse.Namespace) -> None:
    result = validate_tracklet_classifier_mixture_inputs(
        args.train_csvs,
        args.eval_csvs,
        args.out,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        min_train_rows=args.min_train_rows,
        min_eval_rows=args.min_eval_rows,
        min_train_positives=args.min_train_positives,
        min_eval_positives=args.min_eval_positives,
        require_train_negatives=not args.allow_train_without_negatives,
        require_eval_negatives=not args.allow_eval_without_negatives,
        fail_on_train_eval_sequence_overlap=not args.allow_train_eval_sequence_overlap,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_validate_tracklet_classifier_frame_benchmark_inputs(args: argparse.Namespace) -> None:
    result = validate_tracklet_classifier_frame_benchmark_inputs(
        args.run_roots,
        args.gt_csvs,
        args.weights,
        args.out,
        dataset_names=args.dataset_names,
        prediction_name=args.prediction_name,
        diagnostics_name=args.diagnostics_name,
        thresholds=args.thresholds,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        max_frames=args.max_frames,
        min_prediction_rows=args.min_prediction_rows,
        min_gt_boxes=args.min_gt_boxes,
        require_diagnostics=not args.allow_missing_diagnostics,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_build_tracklet_classifier_official_eval_bundle(args: argparse.Namespace) -> None:
    result = build_tracklet_classifier_official_eval_bundle(
        args.frame_summary,
        args.out_dir,
        preflight_json=args.preflight_json,
        baseline_comparison_json=args.baseline_comparison_json,
        copy_predictions=not args.no_copy_predictions,
        require_valid_preflight=not args.allow_missing_or_invalid_preflight,
        require_baseline_comparison=args.require_baseline_comparison,
    )
    print(json.dumps({"manifest": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_export_tracklet_classifier_official_predictions(args: argparse.Namespace) -> None:
    default_image_size = None
    if args.image_width is not None or args.image_height is not None:
        if args.image_width is None or args.image_height is None:
            raise ValueError("--image-width and --image-height must be provided together")
        default_image_size = (args.image_width, args.image_height)
    result = export_tracklet_classifier_official_predictions(
        args.bundle_manifest,
        args.out_dir,
        formats=args.formats,
        default_image_size=default_image_size,
        score_field=args.score_field,
        min_score=args.min_score,
        class_id=args.class_id,
        include_background=args.include_background,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_export_tracklet_classifier_aot_predictions(args: argparse.Namespace) -> None:
    result = export_tracklet_classifier_aot_prediction_parts(
        args.flat_csv,
        args.out_dir,
        image_name_template=args.image_name_template,
        image_name_mode=args.image_name_mode,
        frame_id_offset=args.frame_id_offset,
        part_name=args.part_name,
        min_score=args.min_score,
        score_field=args.score_field,
        class_name=args.class_name,
        group_by_image=not args.no_group_by_image,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_validate_tracklet_classifier_aot_eval_inputs(args: argparse.Namespace) -> None:
    result = validate_tracklet_classifier_aot_eval_inputs(
        args.results_folder,
        args.out,
        clip_id_to_flight_id_path=args.clip_id_to_flight_id_path,
        require_clip_pattern=not args.allow_non_clip_names,
        require_known_clip_ids=not args.allow_unknown_clip_ids,
        max_records=args.max_records,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_export_aot_prediction_parts_to_tracklets(args: argparse.Namespace) -> None:
    result = export_aot_prediction_parts_to_tracklets(
        args.results_folder,
        args.out,
        min_score=args.min_score,
        dataset_source=args.dataset_source,
        image_width=args.image_width,
        image_height=args.image_height,
        min_tracklet_rows=args.min_tracklet_rows,
        max_frame_gap=args.max_frame_gap,
        clip_id_to_flight_id_path=args.clip_id_to_flight_id_path,
        aot_groundtruth_json=args.aot_groundtruth_json,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_filter_aot_prediction_parts_by_tracklets(args: argparse.Namespace) -> None:
    result = filter_aot_prediction_parts_by_tracklets(
        args.results_folder,
        args.tracklet_jsonl,
        args.out_dir,
        part_name=args.part_name,
        score_field=args.score_field,
        min_tracklet_score=args.min_tracklet_score,
        min_tracklet_rows=args.min_tracklet_rows,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_rescore_aot_prediction_parts_by_tracklets(args: argparse.Namespace) -> None:
    result = rescore_aot_prediction_parts_by_tracklets(
        args.results_folder,
        args.tracklet_jsonl,
        args.out_dir,
        part_name=args.part_name,
        score_field=args.score_field,
        center=args.center,
        beta=args.beta,
        mode=args.mode,
        min_tracklet_rows=args.min_tracklet_rows,
        missing_score_behavior=args.missing_score_behavior,
        protect_raw_score_at=args.protect_raw_score_at,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))
    if not result.summary.get("valid", False) and not args.allow_invalid:
        raise SystemExit(2)


def cmd_train_tracklet_classifier(args: argparse.Namespace) -> None:
    out = train_tracklet_classifier(
        args.csv,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        hard_tiny_positive_augments=args.hard_tiny_positive_augments,
        balance_by=args.balance_by,
    )
    print(f"Wrote tracklet classifier: {out}")


def cmd_eval_tracklet_classifier(args: argparse.Namespace) -> None:
    metrics = evaluate_tracklet_classifier(args.csv, args.weights, args.out, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))


def cmd_eval_tracklet_classifier_thresholds(args: argparse.Namespace) -> None:
    result = evaluate_tracklet_classifier_thresholds(args.csv, args.weights, args.out_dir, thresholds=args.thresholds)
    print(json.dumps({"csv": str(result.csv_path), "json": str(result.summary_path), **result.summary}, indent=2))


def cmd_run_tracklet_classifier_mixture_benchmark(args: argparse.Namespace) -> None:
    result = run_tracklet_classifier_mixture_benchmark(
        args.train_csvs,
        args.eval_csvs,
        args.out_dir,
        train_source_names=args.train_source_names,
        eval_dataset_names=args.eval_dataset_names,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        hard_tiny_positive_augments=args.hard_tiny_positive_augments,
        balance_by=args.balance_by,
        thresholds=args.thresholds,
        preflight=not args.skip_preflight,
        strict_preflight=not args.allow_invalid_preflight,
        fail_on_train_eval_sequence_overlap=not args.allow_train_eval_sequence_overlap,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        baseline_lower_is_better=args.baseline_lower_is_better,
        baseline_digits=args.baseline_digits,
        allow_invalid_baselines=args.allow_invalid_baselines,
    )
    print(json.dumps({"json": str(result.out_path), **result.summary}, indent=2))


def cmd_run_tracklet_classifier_frame_benchmark(args: argparse.Namespace) -> None:
    result = run_tracklet_classifier_frame_benchmark(
        args.run_roots,
        args.gt_csvs,
        args.weights,
        args.out_dir,
        dataset_names=args.dataset_names,
        prediction_name=args.prediction_name,
        diagnostics_name=args.diagnostics_name,
        threshold=args.threshold,
        thresholds=args.thresholds,
        untracked_policy=args.untracked_policy,
        promote_positive_tracklets=not args.disable_tracklet_promotion,
        promotion_score_floor=args.promotion_score_floor,
        promotion_min_branch_drone=args.promotion_min_branch_drone,
        promotion_max_background=args.promotion_max_background,
        selective_promotion=args.selective_promotion,
        selective_min_temporal_crop_delta=args.selective_min_temporal_crop_delta,
        selective_min_temporal_background_margin=args.selective_min_temporal_background_margin,
        selective_max_tracklet_background=args.selective_max_tracklet_background,
        selective_max_tracklet_objectness=args.selective_max_tracklet_objectness,
        selective_min_tracklet_rows=args.selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=args.selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=args.selective_min_weak_detector_temporal_signal,
        selective_require_recovery_source=not args.selective_allow_non_recovery_source,
        selective_max_promoted_tracklets_per_sequence=args.selective_max_promoted_tracklets_per_sequence,
        iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold,
        max_frames=args.max_frames,
        baseline_csv=args.baseline_csv,
        baseline_metric=args.baseline_metric,
        baseline_lower_is_better=args.baseline_lower_is_better,
        baseline_digits=args.baseline_digits,
        allow_invalid_baselines=args.allow_invalid_baselines,
    )
    print(json.dumps({"csv": str(result.out_path), **result.summary}, indent=2))


def cmd_stage_b_oracle_benchmark(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = run_stage_b_oracle_benchmark(
        metadata,
        args.out,
        crop_weights=args.crop_weights,
        feature_weights=args.feature_weights,
        temporal_weights=args.temporal_weights,
        hard_negative_manifests=args.hard_negative_manifest,
        frame_stride=args.frame_stride,
        negative_per_positive=args.negative_per_positive,
        t=args.t,
        seed=args.seed,
        max_samples=args.max_samples,
        max_hard_negatives=args.max_hard_negatives,
    )
    print(json.dumps(summary, indent=2))


def cmd_eval_proposal_stage_b(args: argparse.Namespace) -> None:
    summary = evaluate_crop_recognizer_on_proposals(
        args.manifest,
        args.crop_weights,
        args.out,
        threshold=args.threshold,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, indent=2))


def cmd_tracker_oracle_benchmark(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = run_tracker_oracle_benchmark(
        metadata,
        args.out,
        detection_stride=args.detection_stride,
        max_frames=args.max_frames,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        tracker_r0=args.tracker_r0,
        tracker_alpha=args.tracker_alpha,
        tracker_beta=args.tracker_beta,
        tracker_reacquire=args.tracker_reacquire,
    )
    print(json.dumps(summary, indent=2))


def cmd_mode_sweep(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    counts: dict[str, int] = {}
    with Path(args.diagnostics).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            best_k = row.get("best_k")
            alignment = None
            if best_k is not None:
                alignment = row.get("per_k", {}).get(str(best_k), {}).get("alignment")
            q = float(row.get("best_quality", 0.0))
            residual = float(alignment.get("photometric_residual", 0.0)) if alignment else 0.0
            blur = float(alignment.get("blur_score", 0.0)) if alignment else 0.0
            inlier = float(alignment.get("inlier_ratio", 0.0)) if alignment else 0.0
            # Motion-debug does not log candidate motion scores, so use residual as a proxy for frame motion here.
            motion_proxy = min(1.0, residual / max(args.high_residual_threshold, 1e-6))
            mode = determine_mode(
                motion_score=motion_proxy,
                alignment_quality=q,
                track_speed=0.0,
                blur_score=blur,
                track_confidence=0.0,
                static_motion_threshold=args.static_motion_threshold,
            )
            if q < args.weak_alignment_q_threshold and mode == "normal":
                mode = "fast_target"
            counts[mode] = counts.get(mode, 0) + 1
            rows_out.append(
                {
                    "frame_id": row.get("frame_id"),
                    "mode": mode,
                    "alignment_quality": q,
                    "motion_proxy": motion_proxy,
                    "photometric_residual": residual,
                    "inlier_ratio": inlier,
                    "blur_score": blur,
                    "best_k": best_k,
                }
            )
    (out / "mode_labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows_out) + "\n", encoding="utf-8")
    summary = {"counts": counts, "num_frames": len(rows_out)}
    (out / "mode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_augment_speed(args: argparse.Namespace) -> None:
    if args.input_dir:
        outputs = make_speed_augmented_video_dir(
            args.input_dir,
            args.out,
            args.strides,
            args.pattern,
            args.max_videos,
            args.keep_fps,
            args.motion_blur,
            args.blur_direction,
        )
        for p in outputs:
            print(f"Wrote {p}")
    if args.video:
        outputs = make_speed_augmented_videos(args.video, args.out, args.strides, args.keep_fps, args.motion_blur, args.blur_direction)
        for p in outputs:
            print(f"Wrote {p}")
    if args.annotations:
        out_csv = Path(args.out) / "speed_augmented_annotations.csv"
        make_speed_augmented_frame_csv(args.annotations, out_csv, args.strides, args.frame_index_column)
        print(f"Wrote {out_csv}")
    if not args.input_dir and not args.video and not args.annotations:
        raise ValueError("Provide --input-dir, --video, and/or --annotations")


def cmd_make_static_hover(args: argparse.Namespace) -> None:
    out = make_static_hover_sample(
        args.video,
        args.out_video,
        args.x,
        args.y,
        args.radius,
        tuple(args.color),
        args.max_frames,
        args.freeze_background,
        args.jitter_px,
        args.seed,
    )
    print(f"Wrote {out}")


def cmd_make_moving_target(args: argparse.Namespace) -> None:
    out = make_moving_target_sample(
        args.video,
        args.out_video,
        args.start_x,
        args.start_y,
        args.vx,
        args.vy,
        args.radius,
        tuple(args.color),
        args.max_frames,
        args.freeze_background,
    )
    print(f"Wrote {out}")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        paths.extend(matches)
    return paths


def cmd_build_static_hover_crops(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    manifest = build_static_hover_crop_dataset(
        metadata,
        args.out,
        class_name=args.class_name,
        negative_class=args.negative_class,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.scale,
        crop_size=args.size,
        seed=args.seed,
    )
    print(f"Wrote crop dataset manifest: {manifest}")


def cmd_mine_hard_negative_crops(args: argparse.Namespace) -> None:
    videos = _expand_paths(args.video)
    excludes = _expand_paths(args.exclude_metadata) if args.exclude_metadata else []
    if not videos:
        raise FileNotFoundError(f"No videos matched: {args.video}")
    manifest = mine_motion_hard_negative_crops(
        videos,
        args.out,
        exclude_metadata_paths=excludes,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        max_crops_per_video=args.max_crops_per_video,
        min_area=args.min_area,
        max_area=args.max_area,
        min_motion_score=args.min_motion_score,
        artifact_q_threshold=args.artifact_q_threshold,
        crop_scale=args.scale,
        crop_size=args.size,
        exclude_padding_px=args.exclude_padding_px,
    )
    print(f"Wrote hard-negative crop manifest: {manifest}")


def cmd_build_static_hover_yolo_dataset(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    export_dir = Path(args.out) / "source"
    csv_path = export_static_hover_frames_csv(metadata, export_dir, args.frame_stride, args.max_frames_per_video)
    if args.tiled:
        data_yaml = build_tiled_class_agnostic_yolo_dataset(
            csv_path,
            Path(args.out) / "yolo_tiled",
            images_root=None,
            tile_size=args.tile_size,
            positives_per_box=args.positives_per_box,
            negatives_per_image=args.negatives_per_image,
            val_fraction=args.val_fraction,
            seed=args.seed,
            min_box_px=args.min_box_px,
            negative_pad_px=args.negative_pad_px,
            photometric_augmentations=args.photometric_augmentations,
            low_contrast_injections=args.low_contrast_injections,
            positive_repeat_patterns=tuple(args.positive_repeat_pattern or ()),
            positive_repeat_factor=args.positive_repeat_factor,
        )
    else:
        data_yaml = build_class_agnostic_yolo_dataset(csv_path, Path(args.out) / "yolo", images_root=None, val_fraction=args.val_fraction, seed=args.seed, min_box_px=args.min_box_px)
    print(f"Wrote synthetic frame CSV: {csv_path}")
    print(f"Wrote YOLO data.yaml: {data_yaml}")


def cmd_build_static_hover_temporal(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    manifest = build_static_hover_temporal_dataset(
        metadata,
        args.out,
        t=args.t,
        frame_stride=args.frame_stride,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.scale,
        crop_size=args.size,
        seed=args.seed,
    )
    print(f"Wrote temporal dataset manifest: {manifest}")


def cmd_build_detector_proposal_stage_b(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    summary = build_detector_proposal_stage_b_dataset(
        metadata,
        args.out,
        args.yolo_weights,
        yolo_conf=args.yolo_conf,
        tile_size=args.yolo_tile_size,
        tile_stride=args.yolo_tile_stride,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        max_proposals_per_frame=args.max_proposals_per_frame,
        max_negatives_per_frame=args.max_negatives_per_frame,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        crop_scale=args.scale,
        crop_size=args.crop_size,
        tube_t=args.t,
        tube_size=args.tube_size,
    )
    print(json.dumps(summary, indent=2))


def cmd_build_static_hover_feature_dataset(args: argparse.Namespace) -> None:
    metadata = _expand_paths(args.metadata)
    if not metadata:
        raise FileNotFoundError(f"No metadata files matched: {args.metadata}")
    csv_path = export_static_hover_feature_csv(metadata, args.out, args.frame_stride, args.negative_per_positive, args.seed)
    print(f"Wrote feature ROI CSV: {csv_path}")


def cmd_mine_hard_negative_temporal(args: argparse.Namespace) -> None:
    manifest = mine_hard_negative_temporal_dataset(
        args.manifest,
        args.out,
        t=args.t,
        crop_scale=args.scale,
        crop_size=args.size,
        max_samples_per_class=args.max_samples_per_class,
    )
    print(f"Wrote hard-negative temporal manifest: {manifest}")


def cmd_split_folder_dataset(args: argparse.Namespace) -> None:
    manifest = split_folder_dataset(args.input, args.out, args.val_fraction, args.seed)
    print(f"Wrote split manifest: {manifest}")


def cmd_split_feature_csv(args: argparse.Namespace) -> None:
    train_csv, val_csv = split_feature_csv(args.csv, args.out, args.val_fraction, args.seed)
    print(f"Wrote train CSV: {train_csv}")
    print(f"Wrote val CSV: {val_csv}")


def cmd_build_hard_negative_feature_dataset(args: argparse.Namespace) -> None:
    manifests = _expand_paths(args.manifest)
    if not manifests:
        raise FileNotFoundError(f"No manifests matched: {args.manifest}")
    csv_path = export_hard_negative_feature_csv(manifests, args.out, args.max_samples_per_class)
    print(f"Wrote hard-negative feature CSV: {csv_path}")


def _metadata_video_and_boxes(metadata_path: Path) -> tuple[Path, dict[int, tuple[float, float, float, float]]]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    video = Path(meta["output"])
    if not video.exists():
        video = metadata_path.parent / video.name
    boxes = {
        int(row["frame_id"]): tuple(float(v) for v in row["bbox_xyxy"])
        for row in meta.get("boxes", [])
    }
    return video, boxes


def cmd_stage_a_yolo_recall(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    total = 0
    hit_iou = 0
    hit_center = 0
    for metadata in _expand_paths(args.metadata):
        video, boxes = _metadata_video_and_boxes(metadata)
        frames, _ = _read_video_frames(str(video))
        for frame_id, frame in enumerate(frames):
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
            if frame_id % args.frame_stride != 0:
                continue
            gt = boxes.get(frame_id)
            if gt is None:
                continue
            total += 1
            cands = candidates_from_yolo_tiled(
                frame,
                args.yolo_weights,
                tile_size=args.yolo_tile_size,
                stride=args.yolo_tile_stride,
                conf=args.yolo_conf,
                device=args.device,
                max_det=args.max_det,
            )
            best_iou = max((bbox_iou(c.bbox_xyxy, gt) for c in cands), default=0.0)
            best_center = min((center_distance(c.bbox_xyxy, gt) for c in cands), default=float("inf"))
            ok_iou = best_iou >= args.match_iou
            ok_center = best_center <= args.match_center_px
            hit_iou += int(ok_iou)
            hit_center += int(ok_center)
            rows_out.append(
                {
                    "metadata": str(metadata),
                    "video": str(video),
                    "frame_id": frame_id,
                    "gt_bbox": list(gt),
                    "num_candidates": len(cands),
                    "best_iou": best_iou,
                    "best_center_distance": best_center,
                    "hit_iou": ok_iou,
                    "hit_center": ok_center,
                    "top_candidates": [
                        {"bbox": list(c.bbox_xyxy), "objectness": c.objectness, "source": c.source}
                        for c in sorted(cands, key=lambda x: x.objectness, reverse=True)[: args.keep_top]
                    ],
                }
            )
    summary = {
        "total_gt_frames": total,
        "recall_iou": hit_iou / max(1, total),
        "recall_center": hit_center / max(1, total),
        "match_iou": args.match_iou,
        "match_center_px": args.match_center_px,
        "yolo_weights": args.yolo_weights,
        "tile_size": args.yolo_tile_size,
        "tile_stride": args.yolo_tile_stride,
        "frame_stride": args.frame_stride,
    }
    with (out / "stage_a_yolo_recall.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _load_stage_a_real_frame(row: dict[str, str], frames_root: Path | None, cap_cache: dict[str, cv2.VideoCapture]) -> np.ndarray:
    frame_path = row.get("frame_path")
    if frame_path:
        path = Path(frame_path)
        if not path.is_absolute() and frames_root is not None:
            path = frames_root / path
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Could not read frame image: {path}")
        return frame

    video_path = row.get("video_path") or row.get("source_video")
    if not video_path:
        raise ValueError("Real Stage A recall rows need frame_path or video_path/source_video")
    frame_id = int(float(row.get("frame_id", "0")))
    cap = cap_cache.get(video_path)
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        cap_cache[video_path] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read {video_path} frame {frame_id}")
    return frame


def cmd_stage_a_real_yolo_recall(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_root) if args.frames_root else None

    with Path(args.annotations).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.class_name:
        rows = [r for r in rows if r.get("class") == args.class_name]
    if args.aug_speed:
        allowed = set(args.aug_speed)
        rows = [r for r in rows if r.get("aug_speed") in allowed]

    cap_cache: dict[str, cv2.VideoCapture] = {}
    rows_out: list[dict[str, Any]] = []
    counters: dict[str, dict[str, int]] = {}
    raw_candidate_counts: list[int] = []
    budget_candidate_counts: list[int] = []

    def bump(group: str, total_inc: int, iou_inc: int, center_inc: int) -> None:
        item = counters.setdefault(group, {"total": 0, "hit_iou": 0, "hit_center": 0})
        item["total"] += total_inc
        item["hit_iou"] += iou_inc
        item["hit_center"] += center_inc

    try:
        for idx, row in enumerate(rows):
            if args.max_frames is not None and idx >= args.max_frames:
                break
            if idx % max(1, int(args.frame_stride)) != 0:
                continue
            gt = (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))
            frame = _load_stage_a_real_frame(row, frames_root, cap_cache)
            raw_cands = candidates_from_yolo_tiled(
                frame,
                args.yolo_weights,
                tile_size=args.yolo_tile_size,
                stride=args.yolo_tile_stride,
                conf=args.yolo_conf,
                device=args.device,
                max_det=args.max_det,
            )
            cands = _budget_candidates(raw_cands, args.proposal_nms_iou, args.proposal_top_k)
            raw_candidate_counts.append(len(raw_cands))
            budget_candidate_counts.append(len(cands))
            best_iou = max((bbox_iou(c.bbox_xyxy, gt) for c in cands), default=0.0)
            best_center = min((center_distance(c.bbox_xyxy, gt) for c in cands), default=float("inf"))
            ok_iou = best_iou >= args.match_iou
            ok_center = best_center <= args.match_center_px
            iou_inc = int(ok_iou)
            center_inc = int(ok_center)
            bump("all", 1, iou_inc, center_inc)
            bump(f"tag:{row.get('tag', 'unknown')}", 1, iou_inc, center_inc)
            if row.get("aug_speed"):
                bump(f"aug_speed:{row.get('aug_speed')}", 1, iou_inc, center_inc)
            rows_out.append(
                {
                    "row_index": idx,
                    "video": row.get("video_path") or row.get("source_video"),
                    "frame_path": row.get("frame_path"),
                    "frame_id": int(float(row.get("frame_id", idx))),
                    "class": row.get("class"),
                    "tag": row.get("tag"),
                    "aug_speed": row.get("aug_speed"),
                    "gt_bbox": list(gt),
                    "num_candidates_raw": len(raw_cands),
                    "num_candidates": len(cands),
                    "best_iou": best_iou,
                    "best_center_distance": best_center,
                    "hit_iou": ok_iou,
                    "hit_center": ok_center,
                    "top_candidates": [
                        {"bbox": list(c.bbox_xyxy), "objectness": c.objectness, "source": c.source}
                        for c in sorted(cands, key=lambda x: x.objectness, reverse=True)[: args.keep_top]
                    ],
                }
            )
    finally:
        for cap in cap_cache.values():
            cap.release()

    breakdown = {
        key: {
            **value,
            "recall_iou": value["hit_iou"] / max(1, value["total"]),
            "recall_center": value["hit_center"] / max(1, value["total"]),
        }
        for key, value in sorted(counters.items())
    }
    all_counts = counters.get("all", {"total": 0, "hit_iou": 0, "hit_center": 0})
    summary = {
        "annotations": args.annotations,
        "frames_root": str(frames_root) if frames_root else None,
        "total_gt_frames": all_counts["total"],
        "recall_iou": all_counts["hit_iou"] / max(1, all_counts["total"]),
        "recall_center": all_counts["hit_center"] / max(1, all_counts["total"]),
        "match_iou": args.match_iou,
        "match_center_px": args.match_center_px,
        "yolo_weights": args.yolo_weights,
        "yolo_conf": args.yolo_conf,
        "tile_size": args.yolo_tile_size,
        "tile_stride": args.yolo_tile_stride,
        "frame_stride": args.frame_stride,
        "proposal_nms_iou": args.proposal_nms_iou,
        "proposal_top_k": args.proposal_top_k,
        "candidate_counts": {
            "raw_mean": float(np.mean(raw_candidate_counts)) if raw_candidate_counts else 0.0,
            "raw_p50": float(np.percentile(raw_candidate_counts, 50)) if raw_candidate_counts else 0.0,
            "raw_p90": float(np.percentile(raw_candidate_counts, 90)) if raw_candidate_counts else 0.0,
            "raw_max": int(max(raw_candidate_counts)) if raw_candidate_counts else 0,
            "budget_mean": float(np.mean(budget_candidate_counts)) if budget_candidate_counts else 0.0,
            "budget_p50": float(np.percentile(budget_candidate_counts, 50)) if budget_candidate_counts else 0.0,
            "budget_p90": float(np.percentile(budget_candidate_counts, 90)) if budget_candidate_counts else 0.0,
            "budget_max": int(max(budget_candidate_counts)) if budget_candidate_counts else 0,
        },
        "breakdown": breakdown,
    }
    with (out / "stage_a_real_yolo_recall.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))


def _candidate_count_stats(counts: list[int]) -> dict[str, float | int]:
    return {
        "mean": float(np.mean(counts)) if counts else 0.0,
        "p50": float(np.percentile(counts, 50)) if counts else 0.0,
        "p90": float(np.percentile(counts, 90)) if counts else 0.0,
        "p95": float(np.percentile(counts, 95)) if counts else 0.0,
        "max": int(max(counts)) if counts else 0,
    }


def _load_real_annotation_frames(annotations: str | Path, class_name: str | None = None) -> dict[str, set[int]]:
    by_video: dict[str, set[int]] = {}
    with Path(annotations).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if class_name and row.get("class") != class_name:
                continue
            video = row.get("video_path") or row.get("source_video")
            if not video:
                continue
            by_video.setdefault(video, set()).add(int(float(row.get("frame_id", "0"))))
    return by_video


def _distance_to_sorted_frames(frame_id: int, sorted_frames: list[int]) -> int:
    pos = bisect.bisect_left(sorted_frames, frame_id)
    best = 1_000_000_000
    if pos < len(sorted_frames):
        best = min(best, abs(sorted_frames[pos] - frame_id))
    if pos > 0:
        best = min(best, abs(sorted_frames[pos - 1] - frame_id))
    return best


def cmd_stage_a_real_yolo_fppi(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by_video = _load_real_annotation_frames(args.annotations, args.class_name)
    rows_out: list[dict[str, Any]] = []
    raw_counts: list[int] = []
    budget_counts: list[int] = []
    frames_with_any = 0
    sampled = 0

    for video_idx, (video, annotated_frames) in enumerate(sorted(by_video.items())):
        if args.max_videos is not None and video_idx >= args.max_videos:
            break
        sorted_ann = sorted(annotated_frames)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video}")
        try:
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for frame_id in range(0, nframes, max(1, int(args.frame_stride))):
                if args.max_frames is not None and sampled >= args.max_frames:
                    break
                if _distance_to_sorted_frames(frame_id, sorted_ann) <= args.exclude_radius:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = cap.read()
                if not ok:
                    continue
                raw_cands = candidates_from_yolo_tiled(
                    frame,
                    args.yolo_weights,
                    tile_size=args.yolo_tile_size,
                    stride=args.yolo_tile_stride,
                    conf=args.yolo_conf,
                    device=args.device,
                    max_det=args.max_det,
                )
                cands = _budget_candidates(raw_cands, args.proposal_nms_iou, args.proposal_top_k)
                raw_counts.append(len(raw_cands))
                budget_counts.append(len(cands))
                frames_with_any += int(len(cands) > 0)
                sampled += 1
                rows_out.append(
                    {
                        "video": video,
                        "frame_id": frame_id,
                        "distance_to_nearest_gt_frame": _distance_to_sorted_frames(frame_id, sorted_ann),
                        "num_candidates_raw": len(raw_cands),
                        "num_candidates": len(cands),
                        "top_candidates": [
                            {"bbox": list(c.bbox_xyxy), "objectness": c.objectness, "source": c.source}
                            for c in sorted(cands, key=lambda x: x.objectness, reverse=True)[: args.keep_top]
                        ],
                    }
                )
            if args.max_frames is not None and sampled >= args.max_frames:
                break
        finally:
            cap.release()

    summary = {
        "annotations": args.annotations,
        "sampled_frames": sampled,
        "source_videos": len(by_video),
        "max_videos": args.max_videos,
        "frame_stride": args.frame_stride,
        "exclude_radius": args.exclude_radius,
        "yolo_weights": args.yolo_weights,
        "yolo_conf": args.yolo_conf,
        "tile_size": args.yolo_tile_size,
        "tile_stride": args.yolo_tile_stride,
        "proposal_nms_iou": args.proposal_nms_iou,
        "proposal_top_k": args.proposal_top_k,
        "frames_with_any_proposal": frames_with_any,
        "frames_with_any_rate": frames_with_any / max(1, sampled),
        "fppi_budget_mean": float(np.mean(budget_counts)) if budget_counts else 0.0,
        "candidate_counts": {
            "raw": _candidate_count_stats(raw_counts),
            "budget": _candidate_count_stats(budget_counts),
        },
        "note": "Frames are unannotated ARD100 frames at least exclude_radius frames away from GT; this is a background-proxy FPPI check, not a dedicated negative dataset.",
    }
    with (out / "stage_a_real_yolo_fppi.jsonl").open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2))


def cmd_init_real_data_layout(args: argparse.Namespace) -> None:
    dirs = ensure_real_data_layout(args.root)
    print("Created/verified real data layout:")
    for d in dirs:
        print(d)


def cmd_extract_real_annotated_frames(args: argparse.Namespace) -> None:
    csv_path = extract_real_annotated_frames(
        args.annotations,
        args.frames_dir,
        args.out_csv,
        video_root=args.video_root,
        strict_labels=not args.allow_unknown_labels,
    )
    print(f"Wrote frame annotations: {csv_path}")


def cmd_prepare_real_yolo_dataset(args: argparse.Namespace) -> None:
    result = build_real_yolo_candidate_dataset(
        args.annotations,
        args.out,
        video_root=args.video_root,
        tiled=not args.full_frame,
        tile_size=args.tile_size,
        positives_per_box=args.positives_per_box,
        negatives_per_image=args.negatives_per_image,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_box_px=args.min_box_px,
        negative_pad_px=args.negative_pad_px,
        strict_labels=not args.allow_unknown_labels,
    )
    print(f"Wrote frame annotations: {result.frame_annotations_csv}")
    print(f"Wrote frames: {result.frames_dir}")
    print(f"Wrote YOLO data yaml: {result.data_yaml}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_export_anti_uav300_subset(args: argparse.Namespace) -> None:
    result = export_anti_uav300_subset_from_zip(
        args.zip,
        args.out,
        split=args.split,
        modality=args.modality,
        max_sequences=args.max_sequences,
        start_index=args.start_index,
        frame_stride=args.frame_stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    print(f"Wrote annotations: {result.annotations_csv}")
    print(f"Wrote manifest: {result.manifest_csv}")
    print(f"Extracted videos under: {result.extracted_root}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_export_ard100_annotations(args: argparse.Namespace) -> None:
    result = export_ard100_annotations(
        args.root,
        args.out,
        annotations_zip=args.annotations_zip,
        split=args.split,
        frame_stride=args.frame_stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
        tiny_side_px=args.tiny_side_px,
        default_tag=args.default_tag,
    )
    print(f"Wrote annotations: {result.annotations_csv}")
    print(f"Wrote manifest: {result.manifest_csv}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_build_real_stage_b_dataset(args: argparse.Namespace) -> None:
    result = build_real_stage_b_datasets(
        args.annotations,
        args.out,
        negative_per_positive=args.negative_per_positive,
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        tube_t=args.tube_t,
        tube_size=args.tube_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"Wrote crop dataset: {result.crop_root}")
    print(f"Wrote temporal dataset: {result.temporal_root}")
    print(f"Wrote manifest: {result.manifest_jsonl}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_build_real_detector_proposal_stage_b(args: argparse.Namespace) -> None:
    result = build_real_detector_proposal_stage_b_dataset(
        args.annotations,
        args.out,
        args.yolo_weights,
        yolo_conf=args.yolo_conf,
        fallback_yolo_weights=args.fallback_yolo_weights,
        fallback_yolo_conf=args.fallback_yolo_conf,
        tile_size=args.yolo_tile_size,
        tile_stride=args.yolo_tile_stride,
        max_samples=args.max_samples,
        max_proposals_per_frame=args.max_proposals_per_frame,
        max_fallback_proposals_per_frame=args.max_fallback_proposals_per_frame,
        proposal_nms_iou=args.proposal_nms_iou,
        max_negatives_per_frame=args.max_negatives_per_frame,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        artifact_score_threshold=args.artifact_score_threshold,
        high_score_fp_threshold=args.high_score_fp_threshold,
        non_drone_label_mode=args.non_drone_label_mode,
        hard_positive_max_size_px=args.hard_positive_max_size_px,
        hard_positive_max_score=args.hard_positive_max_score,
        hard_positive_repeat=args.hard_positive_repeat,
        crop_scale=args.crop_scale,
        crop_size=args.crop_size,
        tube_t=args.tube_t,
        tube_size=args.tube_size,
        device=args.device,
    )
    print(f"Wrote crop dataset: {result.crop_root}")
    print(f"Wrote temporal dataset: {result.temporal_root}")
    print(f"Wrote feature CSV: {result.feature_csv}")
    print(f"Wrote manifest: {result.manifest_jsonl}")
    print(f"Wrote summary: {result.summary_json}")


def cmd_calibrate_fusion(args: argparse.Namespace) -> None:
    summary = calibrate_fusion_from_diagnostics(
        args.diagnostics,
        args.gt,
        args.out,
        video_hint=args.video,
        match_iou=args.match_iou,
        match_center_px=args.match_center_px,
        threshold=args.threshold,
        frame_tolerance=args.frame_tolerance,
    )
    print(json.dumps(summary, indent=2))


def cmd_analyze_frame_failures(args: argparse.Namespace) -> None:
    summary = analyze_frame_failures(
        args.run_root,
        args.gt,
        args.out,
        profile=args.profile,
        raw_prediction_name=args.raw_prediction_name,
        filtered_prediction_name=args.filtered_prediction_name,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


def cmd_sweep_tracklet_filter(args: argparse.Namespace) -> None:
    summary = run_tracklet_filter_sweep(
        args.run_roots,
        args.gt,
        args.weights,
        args.out,
        profile=args.profile,
        classifier_thresholds=args.classifier_thresholds,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        promotion_min_branch_drone=args.promotion_min_branch_drone,
        selective_promotion=args.selective_promotion,
        selective_min_temporal_crop_deltas=args.selective_min_temporal_crop_deltas,
        selective_min_temporal_background_margins=args.selective_min_temporal_background_margins,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        selective_max_tracklet_background=args.selective_max_tracklet_background,
        selective_max_tracklet_objectness=args.selective_max_tracklet_objectness,
        selective_min_tracklet_rows=args.selective_min_tracklet_rows,
        selective_min_temporal_gain_rate=args.selective_min_temporal_gain_rate,
        selective_min_weak_detector_temporal_signal=args.selective_min_weak_detector_temporal_signal,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))


def cmd_select_tracklet_model(args: argparse.Namespace) -> None:
    summary = run_tracklet_model_selection(
        args.tracklets,
        args.run_roots,
        args.gt,
        args.out,
        profile=args.profile,
        calib_seqs=args.calib_seqs,
        calib_seq_patterns=args.calib_seq_patterns,
        calib_fraction=args.calib_fraction,
        epochs_values=args.epochs_values,
        hidden_values=args.hidden_values,
        lr_values=args.lr_values,
        hard_tiny_positive_augments_values=args.hard_tiny_positive_augments_values,
        hard_negative_augments_values=args.hard_negative_augments_values,
        classifier_thresholds=args.classifier_thresholds,
        promotion_enabled_values=[bool(v) for v in args.promotion_enabled_values] if args.promotion_enabled_values is not None else None,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        promotion_min_branch_drone=args.promotion_min_branch_drone,
        selective_promotion=args.selective_promotion,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
        max_recall_drop=args.max_recall_drop,
    )
    print(json.dumps(summary, indent=2))


def cmd_select_tracklet_sequence_model(args: argparse.Namespace) -> None:
    summary = run_tracklet_sequence_model_selection(
        args.tracklets_jsonl,
        args.run_roots,
        args.gt,
        args.out,
        profile=args.profile,
        calib_seqs=args.calib_seqs,
        calib_seq_patterns=args.calib_seq_patterns,
        calib_fraction=args.calib_fraction,
        epochs_values=args.epochs_values,
        hidden_values=args.hidden_values,
        max_len_values=args.max_len_values,
        hard_positive_augments_values=args.hard_positive_augments_values,
        hard_negative_augments_values=args.hard_negative_augments_values,
        classifier_thresholds=args.classifier_thresholds,
        promotion_enabled_values=[bool(v) for v in args.promotion_enabled_values] if args.promotion_enabled_values is not None else None,
        promotion_score_floors=args.promotion_score_floors,
        promotion_max_backgrounds=args.promotion_max_backgrounds,
        selective_promotion=args.selective_promotion,
        selective_max_promoted_tracklets_per_sequence_values=args.selective_max_promoted_tracklets_per_sequence_values,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        max_frames=args.max_frames,
        max_recall_drop=args.max_recall_drop,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qstr-dronedet", description="QSTR-DroneDet runnable research MVP")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("motion-debug")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4])
    p.set_defaults(func=cmd_motion_debug)

    p = sub.add_parser("infer")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", default=None)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=0, help="Run YOLO over tiled crops; 0 disables tiled inference")
    p.add_argument("--yolo-tile-stride", type=int, default=192)
    p.add_argument("--yolo-device", default=None, help="Optional YOLO device, e.g. 0 or cpu")
    p.add_argument("--yolo-max-det", type=int, default=300, help="Per YOLO predict max_det before QSTR proposal budgeting")
    p.add_argument("--max-yolo-candidates-per-frame", type=int, default=None, help="Keep only top-N primary YOLO candidates before merge")
    p.add_argument("--max-candidates-per-frame", type=int, default=None, help="Keep only top-N merged candidates per frame after source-priority sorting")
    p.add_argument("--fallback-yolo-weights", default=None, help="Optional fallback YOLO weights used only when primary detector has weak/no candidates")
    p.add_argument("--fallback-yolo-conf", type=float, default=0.15)
    p.add_argument("--fallback-yolo-tile-size", type=int, default=0, help="Fallback tiled inference size; 0 reuses --yolo-tile-size")
    p.add_argument("--fallback-yolo-tile-stride", type=int, default=0, help="Fallback tiled stride; 0 reuses --yolo-tile-stride")
    p.add_argument("--fallback-min-primary-candidates", type=int, default=1, help="Run fallback if primary YOLO returns fewer candidates than this")
    p.add_argument("--fallback-trigger-objectness", type=float, default=0.20, help="Run fallback if the best primary YOLO objectness is below this")
    p.add_argument("--fallback-trigger-final-score", type=float, default=0.0, help="Run fallback after primary recognition if best final drone score is below this; 0 disables post-fusion fallback")
    p.add_argument("--fallback-post-trigger-max-primary-objectness", type=float, default=1.0, help="For post-fusion fallback, require primary best objectness at or below this value; 1.0 disables this guard")
    p.add_argument("--max-fallback-yolo-candidates-per-frame", type=int, default=10, help="Keep only top-N fallback YOLO candidates before merge")
    p.add_argument("--fallback-max-box-side", type=float, default=0.0, help="Drop fallback proposals whose max box side exceeds this many pixels; 0 disables")
    p.add_argument("--fallback-min-box-side", type=float, default=0.0, help="Drop fallback proposals whose max box side is below this many pixels; 0 disables")
    p.add_argument("--disable-fallback-gate", action="store_true", help="Do not suppress fallback-source detections with weak crop/temporal or strong background/artifact evidence")
    p.add_argument("--fallback-gate-min-branch-drone", type=float, default=0.45, help="Fallback acceptance requires crop or temporal drone probability at least this high")
    p.add_argument("--fallback-gate-min-crop-temporal-mean", type=float, default=0.35, help="Fallback acceptance requires mean crop/temporal drone probability at least this high")
    p.add_argument("--fallback-gate-max-negative-evidence", type=float, default=0.55, help="Reject fallback if background/artifact evidence exceeds this")
    p.add_argument("--disable-verified-objectness", action="store_true", help="Do not raise effective objectness for tracker/fallback candidates verified by Stage B")
    p.add_argument("--verified-objectness-mode", choices=["always", "hard_recovery"], default="hard_recovery", help="hard_recovery boosts only fallback or low-objectness tracker recovery candidates; always preserves the older tracker/fallback behavior")
    p.add_argument("--verified-min-branch-drone", type=float, default=0.45)
    p.add_argument("--verified-min-crop-temporal-mean", type=float, default=0.48)
    p.add_argument("--verified-max-negative-evidence", type=float, default=0.62)
    p.add_argument("--verified-objectness-floor", type=float, default=0.55)
    p.add_argument("--enable-hard-tiny-recovery", action="store_true", help="Allow a narrow fallback/tracker recovery rule when crop+temporal support a tiny drone but fusion still predicts background")
    p.add_argument("--hard-tiny-min-crop-drone", type=float, default=0.40)
    p.add_argument("--hard-tiny-min-temporal-drone", type=float, default=0.55)
    p.add_argument("--hard-tiny-min-temporal-crop-delta", type=float, default=0.0, help="Require temporal drone probability to exceed crop drone probability by this margin")
    p.add_argument("--hard-tiny-max-bg-minus-drone", type=float, default=0.08)
    p.add_argument("--hard-tiny-min-support", type=float, default=0.15, help="Minimum tracker support unless the candidate source is fallback")
    p.add_argument("--hard-tiny-score-floor", type=float, default=0.22, help="Effective objectness floor for hard-tiny recovered candidates")
    p.add_argument("--hard-tiny-allow-tracker-only", action="store_true", help="Allow hard-tiny recovery for tracker-only candidates; off by default because stale tracks can create many false positives")
    p.add_argument("--hard-tiny-disable-track-validation", action="store_true", help="Disable track metadata validation for tracker-only hard-tiny recovery")
    p.add_argument("--hard-tiny-max-track-frames-since-detector", type=int, default=3)
    p.add_argument("--hard-tiny-min-track-detector-updates", type=int, default=1)
    p.add_argument("--hard-tiny-max-track-drift", type=float, default=48.0)
    p.add_argument("--hard-tiny-min-track-history", type=int, default=2)
    p.add_argument("--crop-weights", default=None, help="Optional CropRecognizer .pt weights")
    p.add_argument("--feature-weights", default=None, help="Optional FeatureRecognitionModel .pt weights")
    p.add_argument("--temporal-weights", default=None, help="Optional TemporalRecognizer .pt weights")
    p.add_argument("--recognition-crop-scale", type=float, default=4.0, help="Context scale for inference crop recognizer crops")
    p.add_argument("--recognition-tube-scale", type=float, default=4.0, help="Context scale for inference temporal tube crops")
    p.add_argument("--fusion-calibration", default=None, help="Optional fusion_calibration_summary.json or {'weights': ...} JSON")
    p.add_argument("--tracklet-classifier-weights", default=None, help="Optional TrackletMLP .pt checkpoint used as a post-infer filter")
    p.add_argument("--tracklet-classifier-threshold", type=float, default=0.5)
    p.add_argument("--tracklet-filter-untracked", choices=["keep", "suppress"], default="keep", help="How to handle drone predictions without a track_id when tracklet filtering is enabled")
    p.add_argument("--disable-tracklet-promotion", action="store_true", help="Only reject non-drone tracklets; do not promote positive tracklets to drone detections")
    p.add_argument("--tracklet-promotion-score-floor", type=float, default=0.22)
    p.add_argument("--tracklet-promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--tracklet-promotion-max-background", type=float, default=0.68)
    p.add_argument("--tracklet-selective-promotion", action="store_true", help="Require tracklet-level recovery evidence and a per-sequence promotion budget before promoting background rows")
    p.add_argument("--tracklet-selective-min-temporal-crop-delta", type=float, default=0.05)
    p.add_argument("--tracklet-selective-min-temporal-background-margin", type=float, default=-0.05)
    p.add_argument("--tracklet-selective-max-tracklet-background", type=float, default=0.60)
    p.add_argument("--tracklet-selective-max-tracklet-objectness", type=float, default=0.50)
    p.add_argument("--tracklet-selective-min-tracklet-rows", type=int, default=2)
    p.add_argument("--tracklet-selective-min-temporal-gain-rate", type=float, default=0.40)
    p.add_argument("--tracklet-selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    p.add_argument("--tracklet-selective-allow-non-recovery-source", action="store_true")
    p.add_argument("--tracklet-selective-max-promoted-tracklets-per-sequence", type=int, default=2, help="0 disables the per-sequence promotion budget")
    p.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--max-area", type=int, default=5000)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--disable-motion-candidates", action="store_true", help="Skip motion candidate generation; useful for oracle/seed recognition experiments")
    p.add_argument("--seed-box", nargs=4, type=float, default=None, metavar=("X1", "Y1", "X2", "Y2"), help="Experimental/oracle candidate box injected every frame")
    p.add_argument("--seed-objectness", type=float, default=0.75)
    p.add_argument("--seed-track-score", type=float, default=0.8)
    p.add_argument("--save-video", action="store_true")
    p.set_defaults(func=cmd_infer)

    p = sub.add_parser("offline-selector-replay")
    p.add_argument("--predictions", required=True, help="Candidate-level predictions.jsonl produced by infer")
    p.add_argument("--out", required=True, help="Output directory for trajectory.csv and selector_debug.csv")
    p.add_argument("--top-k", type=int, default=5, help="Number of detector-ranked candidates to replay per frame")
    p.add_argument("--max-frame-id", type=int, default=None, help="Optional last frame id to emit, including missing frames")
    p.add_argument("--max-jump-px", type=float, default=48.0, help="Guard threshold for motion/memory center jump")
    p.add_argument("--max-recover-frames", type=int, default=3, help="Frames to bridge with memory prediction before LOST")
    p.add_argument("--min-accept-score", type=float, default=0.25)
    p.add_argument("--min-motion-consistency", type=float, default=0.15)
    p.add_argument("--min-memory-consistency", type=float, default=0.15)
    p.add_argument("--min-topk-margin", type=float, default=0.0)
    p.add_argument("--detector-weight", type=float, default=0.55)
    p.add_argument("--motion-weight", type=float, default=0.25)
    p.add_argument("--memory-weight", type=float, default=0.20)
    p.add_argument("--reacquire-top-k", type=int, default=0, help="Deep candidates to scan for REACQUIRE; 0 scans all frame candidates")
    p.add_argument("--reacquire-max-distance-px", type=float, default=96.0, help="Maximum center distance from predicted memory bbox for REACQUIRE")
    p.add_argument("--reacquire-min-detector-score", type=float, default=0.10, help="Minimum detector score for a REACQUIRE candidate")
    p.add_argument("--reacquire-min-motion-consistency", type=float, default=0.30, help="Minimum predicted-motion consistency for a REACQUIRE candidate")
    p.add_argument("--reacquire-min-memory-consistency", type=float, default=0.20, help="Minimum recent-memory consistency for a REACQUIRE candidate")
    p.add_argument("--reacquire-min-side-ratio", type=float, default=0.35, help="Minimum candidate/reference side ratio for REACQUIRE")
    p.add_argument("--reacquire-max-side-ratio", type=float, default=2.50, help="Maximum candidate/reference side ratio for REACQUIRE")
    p.add_argument("--reacquire-confirm-frames", type=int, default=2, help="Consecutive candidate frames required before REACQUIRE is emitted")
    p.add_argument("--reacquire-stale-after-frames", type=int, default=-1, help="Enable global REACQUIRE after this LOST streak; -1 disables it")
    p.add_argument("--reacquire-global-min-area", type=float, default=0.0, help="Minimum bbox area for stale global REACQUIRE candidates")
    p.add_argument("--reacquire-global-small-min-area", type=float, default=0.0, help="Optional lower bbox area floor for size-adaptive stale global REACQUIRE; requires tracklet-confirmed, crop-persistent pending candidates")
    p.add_argument("--reacquire-global-small-max-distance-px", type=float, default=0.0, help="Optional max distance from predicted memory for size-adaptive small-area global REACQUIRE; 0 disables this cap")
    p.add_argument("--reacquire-global-small-memory-probation-frames", type=int, default=0, help="Suppress memory-near REACQUIRE for this many frames after a confirmed small-area global seed")
    p.add_argument("--reacquire-global-small-repeat-cooldown-frames", type=int, default=0, help="Reject repeated small-area global seeds inside this cooldown unless spatially continuous; 0 disables this guard")
    p.add_argument("--reacquire-global-small-repeat-max-distance-px", type=float, default=0.0, help="Maximum center distance from the last accepted small-area global seed during repeat cooldown; 0 rejects all repeats")
    p.add_argument("--reacquire-global-max-area", type=float, default=0.0, help="Maximum bbox area for stale global REACQUIRE candidates; 0 disables the cap")
    p.add_argument("--reacquire-global-min-detector-score", type=float, default=0.10, help="Minimum detector score for stale global REACQUIRE candidates")
    p.add_argument("--reacquire-global-require-tracklet-confirmation", action="store_true", help="Require positive tracklet-classifier evidence for stale global REACQUIRE candidates")
    p.add_argument("--reacquire-global-reject-tracklet-rejected", action="store_true", help="Reject stale global REACQUIRE candidates only when tracklet filtering explicitly marked them rejected")
    p.add_argument("--reacquire-global-min-tracklet-score", type=float, default=0.50, help="Minimum tracklet classifier probability when global tracklet confirmation is required")
    p.add_argument("--reacquire-global-delayed-confirm-frames", type=int, default=0, help="Override global REACQUIRE confirmation window; 0 uses --reacquire-confirm-frames")
    p.add_argument("--reacquire-min-appearance-similarity", type=float, default=0.0, help="Minimum HSV crop similarity to recent accepted boxes for REACQUIRE; 0 disables it")
    p.add_argument("--reacquire-appearance-weight", type=float, default=0.0, help="Soft HSV crop similarity weight added to REACQUIRE ranking; 0 disables it")
    p.add_argument("--reacquire-crop-score-field", default="", help="Optional candidate row field containing precomputed crop/re-id drone probability")
    p.add_argument("--reacquire-crop-weight", type=float, default=0.0, help="Soft crop classifier/re-id score weight added to REACQUIRE ranking; 0 disables it")
    p.add_argument("--reacquire-min-crop-drone-score", type=float, default=0.0, help="Minimum crop classifier/re-id drone probability for REACQUIRE; 0 disables it")
    p.add_argument("--reacquire-crop-weights", default=None, help="Optional CropRecognizer checkpoint used to score REACQUIRE crops from --video")
    p.add_argument("--reacquire-crop-image-size", type=int, default=128, help="CropRecognizer input size when --reacquire-crop-weights is provided")
    p.add_argument("--appearance-memory-size", type=int, default=8, help="Recent accepted candidate crops stored for appearance REACQUIRE")
    p.add_argument("--video", default=None, help="Optional source video for selector_annotated.mp4")
    p.add_argument("--save-video", action="store_true", help="Render annotated video when --video is provided")
    p.set_defaults(func=cmd_offline_selector_replay)

    p = sub.add_parser("build-seed-admission-dataset")
    p.add_argument("--seed-audit", required=True, help="reacquire_seed_audit.csv written by offline-selector-replay")
    p.add_argument("--annotations", required=True, help="Sparse/exact GT bbox CSV with frame_id,x1,y1,x2,y2 columns")
    p.add_argument("--out", required=True, help="Output CSV for seed-admission training rows")
    p.add_argument("--dataset-source", default="", help="Segment/source id stamped on every output row")
    p.add_argument("--run-id", default="", help="Selector run id stamped on every output row")
    p.add_argument("--iou-threshold", type=float, default=0.10, help="Positive label threshold for seed/GT IoU")
    p.add_argument("--center-distance-threshold-px", type=float, default=32.0, help="Positive label threshold for seed/GT center distance")
    p.add_argument("--no-interpolate-gt", action="store_true", help="Only label frames with exact annotation rows")
    p.set_defaults(func=cmd_build_seed_admission_dataset)

    p = sub.add_parser("build-crop-dataset")
    p.add_argument("--frames", default=None)
    p.add_argument("--annotations", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["csv", "coco"], default="csv")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--size", type=int, default=128)
    p.set_defaults(func=cmd_build_crop_dataset)

    p = sub.add_parser("train-recognizer")
    p.add_argument("--type", choices=["crop", "temporal", "feature"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--balance", choices=["sampler", "class_weight", "none"], default="sampler")
    p.add_argument("--target-mode", choices=["multiclass", "drone_binary"], default="multiclass", help="Use drone_binary to train drone vs non-drone while keeping the 8-class output head")
    p.add_argument("--pretrained", default=None, help="Optional recognizer checkpoint for fine-tuning")
    p.set_defaults(func=cmd_train_recognizer)

    p = sub.add_parser("build-yolo-candidate-dataset")
    p.add_argument("--annotations", required=True, help="CSV with frame_path,x1,y1,x2,y2,class; class is ignored for candidate training")
    p.add_argument("--images-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-box-px", type=float, default=1.0, help="Expand tiny labels to at least this many pixels for candidate training")
    p.set_defaults(func=cmd_build_yolo_candidate_dataset)

    p = sub.add_parser("build-tiled-yolo-candidate-dataset")
    p.add_argument("--annotations", required=True, help="CSV with frame_path,x1,y1,x2,y2,class")
    p.add_argument("--images-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--min-box-px", type=float, default=8.0, help="Expand tiny labels inside each tile")
    p.add_argument("--negative-pad-px", type=float, default=32.0, help="Reject negative tiles this close to a known box")
    p.add_argument("--photometric-augmentations", type=int, default=0, help="Extra brightness/contrast variants per positive tile")
    p.add_argument("--low-contrast-injections", type=int, default=0, help="Extra positive variants with a locally redrawn low-contrast tiny target")
    p.add_argument("--positive-repeat-pattern", action="append", default=[], help="Repeat positives when this substring appears in the source frame path; can be passed multiple times")
    p.add_argument("--positive-repeat-factor", type=int, default=1, help="Positive oversampling factor for matching repeat patterns")
    p.set_defaults(func=cmd_build_tiled_yolo_candidate_dataset)

    p = sub.add_parser("train-yolo-p2")
    p.add_argument("--data", required=True, help="YOLO data.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--model-yaml", default=None)
    p.add_argument("--write-model-yaml", default=None, help="Write default YOLO-P2 yaml here before training")
    p.add_argument("--no-train", action="store_true", help="Only write model yaml")
    p.add_argument("--pretrained", default=None, help="Optional local pretrained YOLO weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_train_yolo_p2)

    p = sub.add_parser("evaluate")
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("build-tracklet-dataset")
    p.add_argument("--diagnostics", nargs="+", required=True, help="One or more diagnostics.jsonl files from infer")
    p.add_argument("--gt-csv", required=True, help="Unified CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--center-threshold", type=float, default=24.0)
    p.set_defaults(func=cmd_build_tracklet_dataset)

    p = sub.add_parser("build-proposal-tracklet-dataset")
    p.add_argument("--run-roots", nargs="+", required=True, help="Profile benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--gt-csv", required=True, help="Unified CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--base-radius", type=float, default=18.0)
    p.add_argument("--radius-per-side", type=float, default=0.75)
    p.add_argument("--min-iou", type=float, default=0.05)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--detector-only", action="store_true")
    p.add_argument("--min-tracklet-rows", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--center-threshold", type=float, default=24.0)
    p.add_argument("--hard-tiny-side", type=float, default=24.0)
    p.add_argument("--hard-low-score", type=float, default=0.25)
    p.set_defaults(func=cmd_build_proposal_tracklet_dataset)

    p = sub.add_parser("export-yolo-oracle-tracklets")
    p.add_argument("--list-files", nargs="+", required=True, help="YOLO image list files such as train.txt/val.txt/test.txt")
    p.add_argument("--out", required=True, help="Output directory for oracle_tracklets.{csv,jsonl} and summary.json")
    p.add_argument("--dataset-source", default="yolo_oracle", help="dataset_source metadata for action-chunk multi-source training")
    p.add_argument("--image-width", type=int, default=None, help="Optional fixed image width; otherwise read each image")
    p.add_argument("--image-height", type=int, default=None, help="Optional fixed image height; otherwise read each image")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--skip-images", type=int, default=0, help="Skip this many image-list rows before exporting")
    p.add_argument("--max-labeled-images-per-seq", type=int, default=None, help="Cap labeled frames exported per parsed sequence")
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--base-radius", type=float, default=18.0)
    p.add_argument("--radius-per-side", type=float, default=0.75)
    p.add_argument("--min-iou", type=float, default=0.05)
    p.add_argument("--min-tracklet-rows", type=int, default=2)
    p.set_defaults(func=cmd_export_yolo_oracle_tracklets)

    p = sub.add_parser("export-temporal-saliency-tracklets")
    p.add_argument("--list-files", nargs="+", required=True, help="Image list files in frame order")
    p.add_argument("--gt-csv", required=True, help="Unified CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True, help="Output directory for proposal_tracklets.{csv,jsonl} and summary.json")
    p.add_argument("--dataset-source", default="vatd_temporal_saliency")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--threshold", type=float, default=24.0, help="Absdiff threshold; <=0 uses Otsu")
    p.add_argument("--min-area", type=float, default=2.0)
    p.add_argument("--max-area", type=float, default=400.0)
    p.add_argument("--dilate-iters", type=int, default=1)
    p.add_argument("--max-gap", type=int, default=3)
    p.add_argument("--base-radius", type=float, default=18.0)
    p.add_argument("--radius-per-side", type=float, default=0.75)
    p.add_argument("--min-iou", type=float, default=0.0)
    p.add_argument("--min-tracklet-rows", type=int, default=2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--center-threshold", type=float, default=24.0)
    p.add_argument("--hard-tiny-side", type=float, default=24.0)
    p.add_argument("--hard-low-score", type=float, default=0.25)
    p.add_argument("--progress-every-sequences", type=int, default=10)
    p.set_defaults(func=cmd_export_temporal_saliency_tracklets)

    p = sub.add_parser("export-frame-list-from-gt-csv")
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--frame-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg"])
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-frames-per-seq", type=int, default=None)
    p.set_defaults(func=cmd_export_frame_list_from_gt_csv)

    p = sub.add_parser("export-tracklet-jsonl-predictions")
    p.add_argument("--tracklet-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", default="vatd")
    p.add_argument("--score-field", default="vatd_score")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--formats", nargs="+", default=["flat_csv", "yolo_txt"])
    p.add_argument("--nms-iou-threshold", type=float, default=-1.0, help="Set >=0 to suppress duplicate boxes per frame by IoU")
    p.add_argument("--nms-center-threshold", type=float, default=-1.0, help="Set >=0 to suppress duplicate boxes per frame by center distance")
    p.set_defaults(func=cmd_export_tracklet_jsonl_predictions)

    p = sub.add_parser("evaluate-flat-tracklet-predictions")
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--prediction-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--center-threshold", type=float, default=-1.0, help="Set >=0 to allow center-distance matching")
    p.add_argument("--fp-limit", type=int, default=None)
    p.add_argument("--max-fppi", type=float, default=None)
    p.add_argument("--fp-limits", nargs="+", type=int, default=None, help="Optional FP budgets for recall curve reporting")
    p.add_argument("--max-fppis", nargs="+", type=float, default=None, help="Optional FPPI budgets for recall curve reporting")
    p.set_defaults(func=cmd_evaluate_flat_tracklet_predictions)

    p = sub.add_parser("sweep-flat-tracklet-prediction-nms")
    p.add_argument("--tracklet-jsonl", required=True)
    p.add_argument("--gt-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", default="vatd")
    p.add_argument("--score-field", default="vatd_score")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--nms-iou-thresholds", nargs="+", default=["none", "0.3", "0.5"])
    p.add_argument("--nms-center-thresholds", nargs="+", default=["none", "6", "12"])
    p.add_argument("--score-thresholds", nargs="+", type=float, default=None)
    p.add_argument("--eval-iou-threshold", type=float, default=0.5)
    p.add_argument("--eval-center-threshold", type=float, default=-1.0, help="Set >=0 to allow center-distance matching")
    p.add_argument("--fp-limit", type=int, default=None)
    p.add_argument("--max-fppi", type=float, default=None)
    p.add_argument("--fp-limits", nargs="+", type=int, default=None, help="Optional FP budgets for recall curve reporting")
    p.add_argument("--max-fppis", nargs="+", type=float, default=None, help="Optional FPPI budgets for recall curve reporting")
    p.set_defaults(func=cmd_sweep_flat_tracklet_prediction_nms)

    p = sub.add_parser("compare-flat-prediction-eval-summaries")
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--method-names", nargs="+", default=None)
    p.set_defaults(func=cmd_compare_flat_prediction_eval_summaries)

    p = sub.add_parser("export-yolo-labels-gt-csv")
    p.add_argument("--list-files", nargs="+", required=True, help="YOLO image list files")
    p.add_argument("--out", required=True, help="Output GT CSV path")
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.set_defaults(func=cmd_export_yolo_labels_gt_csv)

    p = sub.add_parser("export-yolo-predictions-route-b-run")
    p.add_argument("--list-files", nargs="+", required=True, help="YOLO image list files used for inference")
    p.add_argument("--pred-label-dir", required=True, help="Directory of YOLO prediction txt files from --save-txt --save-conf")
    p.add_argument("--out-run-root", required=True, help="Output Route B run root")
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--source", default="yolomg_lowconf")
    p.add_argument("--max-images", type=int, default=None)
    p.set_defaults(func=cmd_export_yolo_predictions_route_b_run)

    p = sub.add_parser("merge-tracklet-jsonl")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-names", nargs="+", default=None)
    p.set_defaults(func=cmd_merge_tracklet_jsonl)

    p = sub.add_parser("build-action-chunk-dataset")
    p.add_argument("--tracklet-jsonl", required=True, help="tracklets.jsonl or proposal_tracklets.jsonl with meta/rows entries")
    p.add_argument("--out", required=True, help="Output action_chunk_samples.jsonl")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None, help="Normalize boxes by image width when provided with --image-height")
    p.add_argument("--image-height", type=int, default=None, help="Normalize boxes by image height when provided with --image-width")
    p.add_argument("--normalize-by-row-image-size", action="store_true", help="Normalize each row by its image_width/image_height metadata")
    p.add_argument("--positives-only", action="store_true", help="Export only positive UAV tracklets")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.set_defaults(func=cmd_build_action_chunk_dataset)

    p = sub.add_parser("merge-action-chunk-datasets")
    p.add_argument("--inputs", nargs="+", required=True, help="Per-source action_chunk_samples.jsonl files")
    p.add_argument("--out", required=True, help="Merged action-chunk JSONL for multi-dataset training")
    p.add_argument("--source-names", nargs="+", default=None, help="Optional canonical source names, one per input")
    p.add_argument("--manifest-out", default=None, help="Optional merged-dataset manifest JSON path")
    p.set_defaults(func=cmd_merge_action_chunk_datasets)

    p = sub.add_parser("split-action-chunk-dataset")
    p.add_argument("--jsonl", required=True, help="Merged action_chunk_samples.jsonl")
    p.add_argument("--out-dir", required=True, help="Directory for train/calib/test action-chunk JSONL files and split manifest")
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--group-field", default="seq", help="Group field kept intact across splits, default seq")
    p.add_argument("--source-field", default="dataset_source", help="Source field used for per-dataset stratification")
    p.set_defaults(func=cmd_split_action_chunk_dataset)

    p = sub.add_parser("train-action-chunk-policy")
    p.add_argument("--jsonl", required=True, help="action_chunk_samples.jsonl from build-action-chunk-dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-type", choices=["mlp", "diffusion", "residual_mlp"], default="mlp", help="Action policy backend")
    p.add_argument("--diffusion-steps", type=int, default=16, help="Denoising steps when --model-type diffusion")
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training, e.g. dataset_source bucket label",
    )
    p.set_defaults(func=cmd_train_action_chunk_policy)

    p = sub.add_parser("eval-action-chunk-policy")
    p.add_argument("--jsonl", required=True, help="action_chunk_samples.jsonl")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True, help="Output per-sample score JSONL")
    p.set_defaults(func=cmd_eval_action_chunk_policy)

    p = sub.add_parser("export-action-prior-heatmaps")
    p.add_argument("--sample-scores", required=True, help="action_chunk_sample_scores.jsonl with learned_boxes")
    p.add_argument("--out-dir", required=True, help="Directory for heatmap .npy files and action_prior_heatmaps.jsonl")
    p.add_argument("--image-width", type=int, required=True)
    p.add_argument("--image-height", type=int, required=True)
    p.add_argument("--sigma-scale", type=float, default=1.5)
    p.add_argument("--min-sigma", type=float, default=2.0)
    p.add_argument("--box-field", default="learned_boxes")
    p.add_argument("--split-horizon", action="store_true", help="Export one heatmap per future action step instead of one combined chunk heatmap")
    p.set_defaults(func=cmd_export_action_prior_heatmaps)

    p = sub.add_parser("build-frame-prior-index")
    p.add_argument("--prior-manifest", required=True, help="action_prior_heatmaps.jsonl from export-action-prior-heatmaps")
    p.add_argument("--out-dir", required=True, help="Directory for frame_prior_index.jsonl and merged frame prior .npy files")
    p.add_argument("--merge-mode", choices=["max", "mean"], default="max")
    p.set_defaults(func=cmd_build_frame_prior_index)

    p = sub.add_parser("attach-frame-priors")
    p.add_argument("--tracklet-jsonl", required=True, help="Nested tracklet JSONL or flat proposal JSONL with seq/frame_id rows")
    p.add_argument("--frame-prior-index", required=True, help="frame_prior_index.jsonl from build-frame-prior-index")
    p.add_argument("--out", required=True, help="Output JSONL with action_frame_prior fields attached to matching rows")
    p.set_defaults(func=cmd_attach_frame_priors)

    p = sub.add_parser("fuse-action-frame-priors")
    p.add_argument("--pred-jsonl", required=True, help="Flat predictions/proposals JSONL with action_frame_prior_score fields")
    p.add_argument("--out", required=True, help="Output JSONL with action-prior-fused final_drone_score")
    p.add_argument("--prior-weight", type=float, default=0.35, help="Blend weight for action prior support")
    p.add_argument("--min-prior-score", type=float, default=0.25, help="Minimum action_frame_prior_score needed before fusion")
    p.add_argument("--promote-threshold", type=float, default=0.20, help="Promote non-drone rows once fused score reaches this value; use a negative value to disable")
    p.add_argument("--min-base-score-for-promotion", type=float, default=0.0, help="Minimum original final_drone_score required for promotion")
    p.set_defaults(func=cmd_fuse_action_frame_priors)

    p = sub.add_parser("sweep-action-frame-priors")
    p.add_argument("--pred-jsonl", required=True, help="Flat predictions/proposals JSONL with action_frame_prior_score fields")
    p.add_argument("--gt-csv", required=True, help="Ground-truth CSV with video_path,frame_id,x1,y1,x2,y2")
    p.add_argument("--out-dir", required=True, help="Directory for sweep CSV and summary JSON")
    p.add_argument("--prior-weights", nargs="+", type=float, default=[0.2, 0.35, 0.5])
    p.add_argument("--min-prior-scores", nargs="+", type=float, default=[0.2, 0.35, 0.5])
    p.add_argument("--promote-thresholds", nargs="+", type=float, default=[-1.0, 0.2, 0.3], help="Use negative value to include no-promotion configs")
    p.add_argument("--min-base-scores-for-promotion", nargs="+", type=float, default=[0.0])
    p.add_argument("--score-threshold", type=float, default=0.20)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_sweep_action_frame_priors)

    p = sub.add_parser("sweep-action-frame-priors-run-root")
    p.add_argument("--run-roots", nargs="+", required=True, help="Benchmark output roots containing <profile>/<seq>/<prediction-name>")
    p.add_argument("--gt-csv", required=True, help="Ground-truth CSV with video_path,frame_id,x1,y1,x2,y2")
    p.add_argument("--out-dir", required=True, help="Directory for dataset-level sweep CSV and summary JSON")
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--prediction-name", default="predictions.jsonl")
    p.add_argument("--prior-weights", nargs="+", type=float, default=[0.2, 0.35, 0.5])
    p.add_argument("--min-prior-scores", nargs="+", type=float, default=[0.2, 0.35, 0.5])
    p.add_argument("--promote-thresholds", nargs="+", type=float, default=[-1.0, 0.2, 0.3], help="Use negative value to include no-promotion configs")
    p.add_argument("--min-base-scores-for-promotion", nargs="+", type=float, default=[0.0])
    p.add_argument("--score-threshold", type=float, default=0.20)
    p.add_argument("--iou-threshold", type=float, default=0.30)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_sweep_action_frame_priors_run_root)

    p = sub.add_parser("run-action-policy-ablation")
    p.add_argument("--jsonl", required=True, help="action_chunk_samples.jsonl")
    p.add_argument("--out-dir", required=True, help="Directory for model weights, sample scores, and ablation tables")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training",
    )
    p.set_defaults(func=cmd_run_action_policy_ablation)

    p = sub.add_parser("run-action-policy-split-selection")
    p.add_argument("--train-jsonl", required=True, help="Train split from split-action-chunk-dataset")
    p.add_argument("--calib-jsonl", required=True, help="Calibration split used for model/backend selection")
    p.add_argument("--test-jsonl", default=None, help="Optional held-out test split evaluated after calibration selection")
    p.add_argument("--out-dir", required=True, help="Directory for model weights, calib/test scores, and selection tables")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training",
    )
    p.set_defaults(func=cmd_run_action_policy_split_selection)

    p = sub.add_parser("run-multisource-action-policy-experiment")
    p.add_argument("--inputs", nargs="+", required=True, help="Per-source action_chunk_samples.jsonl files")
    p.add_argument("--out-dir", required=True, help="Experiment directory for merge/split/selection artifacts")
    p.add_argument("--source-names", nargs="+", default=None, help="Optional canonical source names, one per input")
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--group-field", default="seq", help="Group field kept intact across splits, default seq")
    p.add_argument("--source-field", default="dataset_source", help="Source field used for per-dataset stratification")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training, e.g. dataset_source bucket label",
    )
    p.set_defaults(func=cmd_run_multisource_action_policy_experiment)

    p = sub.add_parser("run-multisource-tracklet-action-policy-experiment")
    p.add_argument("--tracklet-inputs", nargs="+", required=True, help="Per-source nested tracklet/proposal_tracklet JSONL files")
    p.add_argument("--out-dir", required=True, help="Experiment directory for action-chunk export, merge/split, and policy selection")
    p.add_argument("--source-names", nargs="+", default=None, help="Optional canonical source names, one per tracklet input")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None, help="Normalize boxes by image width when provided with --image-height")
    p.add_argument("--image-height", type=int, default=None, help="Normalize boxes by image height when provided with --image-width")
    p.add_argument("--positives-only", action="store_true", help="Export only positive UAV tracklets")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--group-field", default="seq", help="Group field kept intact across splits, default seq")
    p.add_argument("--source-field", default="dataset_source", help="Source field used for per-dataset stratification")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training, e.g. dataset_source bucket label",
    )
    p.set_defaults(func=cmd_run_multisource_tracklet_action_policy_experiment)

    p = sub.add_parser("run-multisource-tracklet-policy-benchmark")
    p.add_argument("--train-tracklet-inputs", nargs="+", required=True, help="Per-source train tracklet/proposal_tracklet JSONL files")
    p.add_argument("--eval-tracklet-inputs", nargs="+", required=True, help="Held-out eval tracklet/proposal_tracklet JSONL files")
    p.add_argument("--out-dir", required=True, help="Benchmark directory for training, per-dataset eval, and collected Route B results")
    p.add_argument("--train-source-names", nargs="+", default=None, help="Optional canonical source names, one per train input")
    p.add_argument("--eval-dataset-names", nargs="+", default=None, help="Optional dataset names, one per eval input")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None, help="Normalize boxes by image width when provided with --image-height")
    p.add_argument("--image-height", type=int, default=None, help="Normalize boxes by image height when provided with --image-width")
    p.add_argument("--positives-only", action="store_true", help="Export only positive UAV tracklets for training")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--group-field", default="seq", help="Group field kept intact across train/calib/test splits, default seq")
    p.add_argument("--source-field", default="dataset_source", help="Source field used for per-dataset stratification")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--error-scale", type=float, default=8.0)
    p.add_argument("--thresholds", nargs="*", type=float, default=None, help="Optional dynamics-score thresholds for held-out eval sweeps")
    p.add_argument("--baseline-csv", default=None, help="Optional baseline CSV with dataset,method,<metric> for automatic comparison")
    p.add_argument("--baseline-metric", default="best_f1", help="Metric column to compare against baselines")
    p.add_argument("--baseline-lower-is-better", action="store_true", help="Use when lower baseline metric values are better")
    p.add_argument("--baseline-digits", type=int, default=3, help="Digits for generated Markdown baseline table")
    p.add_argument("--allow-invalid-baselines", action="store_true", help="Continue when baseline CSV validation fails")
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced training, e.g. dataset_source bucket label",
    )
    p.set_defaults(func=cmd_run_multisource_tracklet_policy_benchmark)

    p = sub.add_parser("run-multisource-proposal-policy-benchmark")
    p.add_argument("--train-run-roots", nargs="+", required=True, help="Per-source benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--train-gt-csvs", nargs="+", required=True, help="Per-source GT CSVs, one per train run root")
    p.add_argument("--eval-run-roots", nargs="+", required=True, help="Held-out benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--eval-gt-csvs", nargs="+", required=True, help="Held-out GT CSVs, one per eval run root")
    p.add_argument("--out-dir", required=True, help="Output directory for proposal tracklets, benchmark, and reports")
    p.add_argument("--train-source-names", nargs="+", default=None)
    p.add_argument("--eval-dataset-names", nargs="+", default=None)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--proposal-max-gap", type=int, default=3)
    p.add_argument("--proposal-base-radius", type=float, default=18.0)
    p.add_argument("--proposal-radius-per-side", type=float, default=0.75)
    p.add_argument("--proposal-min-iou", type=float, default=0.05)
    p.add_argument("--proposal-min-score", type=float, default=0.0)
    p.add_argument("--proposal-detector-only", action="store_true")
    p.add_argument("--proposal-min-tracklet-rows", type=int, default=1)
    p.add_argument("--proposal-iou-threshold", type=float, default=0.3)
    p.add_argument("--proposal-center-threshold", type=float, default=24.0)
    p.add_argument("--proposal-hard-tiny-side", type=float, default=24.0)
    p.add_argument("--proposal-hard-low-score", type=float, default=0.25)
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--positives-only", action="store_true")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--calib-fraction", type=float, default=0.2)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=59)
    p.add_argument("--group-field", default="seq")
    p.add_argument("--source-field", default="dataset_source")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--error-scale", type=float, default=8.0)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--baseline-csv", default=None)
    p.add_argument("--baseline-metric", default="best_f1")
    p.add_argument("--baseline-lower-is-better", action="store_true")
    p.add_argument("--baseline-digits", type=int, default=3)
    p.add_argument("--allow-invalid-baselines", action="store_true")
    p.add_argument("--balance-by", nargs="*", default=None)
    p.set_defaults(func=cmd_run_multisource_proposal_policy_benchmark)

    p = sub.add_parser("validate-route-b-tracklet-inputs")
    p.add_argument("--train-tracklet-inputs", nargs="+", required=True, help="Per-source train tracklet/proposal_tracklet JSONL files")
    p.add_argument("--eval-tracklet-inputs", nargs="+", required=True, help="Held-out eval tracklet/proposal_tracklet JSONL files")
    p.add_argument("--out", required=True, help="Output preflight validation JSON")
    p.add_argument("--train-source-names", nargs="+", default=None, help="Optional canonical source names, one per train input")
    p.add_argument("--eval-dataset-names", nargs="+", default=None, help="Optional dataset names, one per eval input")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    p.set_defaults(func=cmd_validate_route_b_tracklet_inputs)

    p = sub.add_parser("validate-route-b-proposal-inputs")
    p.add_argument("--train-run-roots", nargs="+", required=True, help="Per-source benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--train-gt-csvs", nargs="+", required=True, help="Per-source GT CSVs, one per train run root")
    p.add_argument("--eval-run-roots", nargs="+", required=True, help="Held-out benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--eval-gt-csvs", nargs="+", required=True, help="Held-out GT CSVs, one per eval run root")
    p.add_argument("--out", required=True, help="Output preflight validation JSON")
    p.add_argument("--train-source-names", nargs="+", default=None)
    p.add_argument("--eval-dataset-names", nargs="+", default=None)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-bbox-rows", type=int, default=1, help="Minimum bbox rows required per source")
    p.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    p.set_defaults(func=cmd_validate_route_b_proposal_inputs)

    p = sub.add_parser("scan-route-b-proposal-inputs")
    p.add_argument("--scan-roots", nargs="+", required=True, help="Roots to scan for <run>/<profile>/<seq>/diagnostics*.jsonl and GT CSV files")
    p.add_argument("--out", required=True, help="Output scan JSON")
    p.add_argument("--profiles", nargs="+", default=["hard_recovery"])
    p.add_argument("--diagnostics-names", nargs="+", default=["diagnostics_raw.jsonl", "diagnostics.jsonl"])
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--max-files", type=int, default=20000)
    p.add_argument("--max-diag-sample-files", type=int, default=20)
    p.add_argument("--max-rows-per-diag-file", type=int, default=200)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_scan_route_b_proposal_inputs)

    p = sub.add_parser("write-route-b-proposal-run-manifest")
    p.add_argument("--out-dir", required=True, help="Directory for manifest plus preflight/start/monitor helper commands")
    p.add_argument("--train-run-roots", nargs="+", required=True, help="Per-source benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--train-gt-csvs", nargs="+", required=True, help="Per-source GT CSVs, one per train run root")
    p.add_argument("--eval-run-roots", nargs="+", required=True, help="Held-out benchmark roots containing <profile>/<seq>/diagnostics*.jsonl")
    p.add_argument("--eval-gt-csvs", nargs="+", required=True, help="Held-out GT CSVs, one per eval run root")
    p.add_argument("--train-source-names", nargs="+", default=None)
    p.add_argument("--eval-dataset-names", nargs="+", default=None)
    p.add_argument("--run-id", default="route_b_proposal_benchmark")
    p.add_argument("--benchmark-out-dir", default=None, help="Optional benchmark output directory; defaults to <out-dir>/run")
    p.add_argument("--runner-output-root", default=None, help="Optional detached runner metadata/log directory; defaults to <out-dir>/runner")
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--diagnostics-name", default="diagnostics_raw.jsonl")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--balance-by", nargs="*", default=["dataset_source"])
    p.add_argument("--baseline-csv", default=None)
    p.add_argument("--baseline-metric", default="best_f1")
    p.add_argument("--baseline-lower-is-better", action="store_true")
    p.add_argument("--skip-validation", action="store_true", help="Write manifest without running proposal input preflight")
    p.add_argument("--preflight-min-bbox-rows", type=int, default=1)
    p.set_defaults(func=cmd_write_route_b_proposal_run_manifest)

    p = sub.add_parser("score-action-chunk-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="tracklets.jsonl or proposal_tracklets.jsonl with meta/rows entries")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True, help="Output per-tracklet dynamics score JSONL")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--normalize-by-row-image-size", action="store_true", help="Normalize boxes by each row's image_width/image_height metadata")
    p.add_argument("--error-scale", type=float, default=8.0, help="Center-error scale for exp(-error/scale) dynamics score")
    p.add_argument("--dynamics-score-mode", choices=["learned_consistency", "cv_consistency", "improvement", "hybrid"], default="learned_consistency")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.set_defaults(func=cmd_score_action_chunk_tracklets)

    p = sub.add_parser("score-constant-velocity-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="tracklets.jsonl or proposal_tracklets.jsonl with meta/rows entries")
    p.add_argument("--out", required=True, help="Output per-tracklet constant-velocity dynamics score JSONL")
    p.add_argument("--min-tracklet-rows", type=int, default=3)
    p.add_argument("--error-scale", type=float, default=1.0, help="Scale for exp(-normalized_center_error/scale)")
    p.add_argument("--min-box-side", type=float, default=1.0)
    p.set_defaults(func=cmd_score_constant_velocity_tracklets)

    p = sub.add_parser("train-video-action-chunk-policy")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL with frame ids and bboxes")
    p.add_argument("--out", required=True, help="Output Video-Action policy .pt checkpoint")
    p.add_argument("--frame-root", default=None, help="Root containing video frames, e.g. AOT part0/frames")
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--past-len", type=int, default=4)
    p.add_argument("--future-len", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train_video_action_chunk_policy)

    p = sub.add_parser("score-video-action-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL with frame ids and bboxes")
    p.add_argument("--weights", required=True, help="Video-Action policy .pt checkpoint")
    p.add_argument("--out", required=True, help="Output per-tracklet Video-Action dynamics scores")
    p.add_argument("--frame-root", default=None)
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--error-scale", type=float, default=0.02)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.set_defaults(func=cmd_score_video_action_tracklets)

    p = sub.add_parser("train-video-action-multihead-policy")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL with frame ids and bboxes")
    p.add_argument("--out", required=True, help="Output multihead Video-Action policy .pt checkpoint")
    p.add_argument("--frame-root", default=None, help="Root containing video frames, e.g. AOT part0/frames")
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--past-len", type=int, default=4)
    p.add_argument("--future-len", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--confidence-target", choices=["mean", "max"], default="max")
    p.add_argument("--confidence-loss-weight", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train_video_action_multihead_policy)

    p = sub.add_parser("score-video-action-multihead-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL with frame ids and bboxes")
    p.add_argument("--weights", required=True, help="Multihead Video-Action policy .pt checkpoint")
    p.add_argument("--out", required=True, help="Output per-tracklet multihead Video-Action scores")
    p.add_argument("--frame-root", default=None)
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--error-scale", type=float, default=0.02)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fusion-mode", choices=["predicted_confidence", "dynamics_times_predicted_confidence"], default="dynamics_times_predicted_confidence")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.set_defaults(func=cmd_score_video_action_multihead_tracklets)

    p = sub.add_parser("train-vatd-motion-action-policy")
    p.add_argument("--tracklet-jsonl", required=True, help="Tracklet JSONL with labels, frame ids, and bboxes")
    p.add_argument("--out", required=True, help="Output VATD motion-action .pt checkpoint")
    p.add_argument("--frame-root", default=None, help="Root containing video frames")
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--past-len", type=int, default=4)
    p.add_argument("--future-len", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--action-loss-weight", type=float, default=0.25)
    p.add_argument("--motion-pos-weight", default="auto", help="Positive class weight for motion-action BCE; use 'auto' or a positive number")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory for large Windows/CUDA batches")
    p.add_argument("--no-shuffle", action="store_true", help="Preserve dataset order so per-worker frame cache can improve video crop IO")
    p.add_argument("--disable-crops", action="store_true", help="Train from bbox/score/visible state only without image crop decoding")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train_vatd_motion_action_policy)

    p = sub.add_parser("score-vatd-motion-action-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="Tracklet JSONL with labels, frame ids, and bboxes")
    p.add_argument("--weights", required=True, help="VATD motion-action .pt checkpoint")
    p.add_argument("--out", required=True, help="Output per-tracklet VATD scores")
    p.add_argument("--frame-root", default=None)
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--error-scale", type=float, default=0.02)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fusion-mode", choices=["motion_action", "motion_times_action_consistency"], default="motion_action")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--disable-crops", action="store_true", help="Use zero crop tensors and score from bbox/state/motion features only")
    p.set_defaults(func=cmd_score_vatd_motion_action_tracklets)

    p = sub.add_parser("train-ego-adaptive-vatd-policy")
    p.add_argument("--tracklet-jsonl", required=True, help="Tracklet JSONL with labels, frame ids, bboxes, and optional camera motion fields")
    p.add_argument("--out", required=True, help="Output ego-adaptive VATD .pt checkpoint")
    p.add_argument("--frame-root", default=None, help="Root containing video frames")
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--past-len", type=int, default=7)
    p.add_argument("--future-len", type=int, default=2)
    p.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 7], help="Soft-routed action chunk horizons, each <= past-len")
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--action-loss-weight", type=float, default=0.25)
    p.add_argument("--motion-pos-weight", default="auto", help="Positive class weight for motion-action BCE; use 'auto' or a positive number")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--no-pin-memory", action="store_true", help="Disable DataLoader pinned memory for large Windows/CUDA batches")
    p.add_argument("--no-shuffle", action="store_true", help="Preserve dataset order so per-worker frame cache can improve video crop IO")
    p.add_argument("--disable-crops", action="store_true", help="Train from bbox/score/visible state only without image crop decoding")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train_ego_adaptive_vatd_policy)

    p = sub.add_parser("score-ego-adaptive-vatd-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="Tracklet JSONL with labels, frame ids, bboxes, and optional camera motion fields")
    p.add_argument("--weights", required=True, help="Ego-adaptive VATD .pt checkpoint")
    p.add_argument("--out", required=True, help="Output per-tracklet ego-adaptive VATD scores")
    p.add_argument("--frame-root", default=None)
    p.add_argument("--image-name-template", default="{seq}_{frame_id_05d}.png")
    p.add_argument("--error-scale", type=float, default=0.02)
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--fusion-mode", choices=["motion_action", "motion_times_action_consistency"], default="motion_action")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--frame-cache-size", type=int, default=8, help="Per-worker in-memory decoded-frame LRU size; uses RAM, not disk")
    p.add_argument("--allow-missing-images", action="store_true", help="Use black frames for missing images; for smoke tests only")
    p.add_argument("--disable-crops", action="store_true", help="Use zero crop tensors and score from bbox/state/motion features only")
    p.set_defaults(func=cmd_score_ego_adaptive_vatd_tracklets)

    p = sub.add_parser("attach-vatd-scores-to-tracklets")
    p.add_argument("--tracklet-jsonl", required=True, help="Original nested tracklets JSONL")
    p.add_argument("--vatd-scores", required=True, help="Output JSONL from score-vatd-motion-action-tracklets")
    p.add_argument("--out", required=True, help="Output nested tracklets JSONL with VATD fields copied into meta and rows")
    p.add_argument("--score-field", default="vatd_score")
    p.set_defaults(func=cmd_attach_vatd_scores_to_tracklets)

    p = sub.add_parser("attach-action-dynamics-scores")
    p.add_argument("--tracklet-jsonl", required=True, help="Original tracklets.jsonl or proposal_tracklets.jsonl")
    p.add_argument("--dynamics-scores", required=True, help="Output JSONL from score-action-chunk-tracklets")
    p.add_argument("--out", required=True, help="Output tracklets JSONL with dynamics fields attached")
    p.set_defaults(func=cmd_attach_action_dynamics_scores)

    p = sub.add_parser("attach-tracklet-confidence-fusion-scores")
    p.add_argument("--tracklet-jsonl", required=True, help="Tracklet JSONL containing a tracklet-level action score")
    p.add_argument("--out", required=True, help="Output tracklets JSONL with fused score fields attached")
    p.add_argument("--action-score-field", default="action_dynamics_score")
    p.add_argument("--confidence-fields", nargs="+", default=["final_drone_score", "objectness", "score"])
    p.add_argument("--confidence-reduction", choices=["mean", "max"], default="mean")
    p.add_argument("--out-score-field", default="video_action_conf_score")
    p.add_argument("--missing-action-score", type=float, default=None)
    p.set_defaults(func=cmd_attach_tracklet_confidence_fusion_scores)

    p = sub.add_parser("run-action-dynamics-pipeline")
    p.add_argument("--tracklet-jsonl", required=True, help="Original tracklets.jsonl or proposal_tracklets.jsonl")
    p.add_argument("--out-dir", required=True, help="Directory for action samples, policy weights, scores, and attached tracklets")
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--normalize-by-row-image-size", action="store_true", help="Normalize boxes by each row's image_width/image_height metadata")
    p.add_argument("--positives-only", action="store_true")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model-type", choices=["mlp", "diffusion", "residual_mlp"], default="mlp", help="Action policy backend")
    p.add_argument("--diffusion-steps", type=int, default=16, help="Denoising steps when --model-type diffusion")
    p.add_argument("--error-scale", type=float, default=8.0)
    p.add_argument("--dynamics-score-mode", choices=["learned_consistency", "cv_consistency", "improvement", "hybrid"], default="learned_consistency")
    p.add_argument("--thresholds", nargs="*", type=float, default=None, help="Optional dynamics-score thresholds for the built-in sweep")
    p.add_argument("--prior-image-width", type=int, default=None, help="Export action-prior heatmaps at this width")
    p.add_argument("--prior-image-height", type=int, default=None, help="Export action-prior heatmaps at this height")
    p.add_argument("--prior-sigma-scale", type=float, default=1.5)
    p.add_argument("--prior-min-sigma", type=float, default=2.0)
    p.add_argument("--prior-split-horizon", action="store_true", help="Export one prior heatmap per future action step")
    p.add_argument("--prior-merge-mode", choices=["max", "mean"], default="max", help="Frame-level merge mode for split-horizon priors")
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced action-policy training",
    )
    p.set_defaults(func=cmd_run_action_dynamics_pipeline)

    p = sub.add_parser("run-action-dynamics-ablation")
    p.add_argument("--tracklet-jsonl", required=True, help="Original tracklets.jsonl or proposal_tracklets.jsonl")
    p.add_argument("--out-dir", required=True, help="Directory for per-backend B-route runs and ablation tables")
    p.add_argument("--model-types", nargs="+", choices=["mlp", "diffusion", "residual_mlp"], default=["mlp", "residual_mlp", "diffusion"])
    p.add_argument("--past-len", type=int, default=8)
    p.add_argument("--future-len", type=int, default=8)
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--normalize-by-row-image-size", action="store_true", help="Normalize boxes by each row's image_width/image_height metadata")
    p.add_argument("--positives-only", action="store_true")
    p.add_argument("--min-tracklet-rows", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--diffusion-steps", type=int, default=16)
    p.add_argument("--error-scale", type=float, default=8.0)
    p.add_argument("--dynamics-score-mode", choices=["learned_consistency", "cv_consistency", "improvement", "hybrid"], default="learned_consistency")
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--prior-image-width", type=int, default=None, help="Export action-prior heatmaps at this width")
    p.add_argument("--prior-image-height", type=int, default=None, help="Export action-prior heatmaps at this height")
    p.add_argument("--prior-sigma-scale", type=float, default=1.5)
    p.add_argument("--prior-min-sigma", type=float, default=2.0)
    p.add_argument("--prior-split-horizon", action="store_true", help="Export one prior heatmap per future action step")
    p.add_argument("--prior-merge-mode", choices=["max", "mean"], default="max", help="Frame-level merge mode for split-horizon priors")
    p.add_argument(
        "--balance-by",
        nargs="*",
        default=None,
        help="Optional sample metadata fields for inverse-frequency balanced action-policy training",
    )
    p.set_defaults(func=cmd_run_action_dynamics_ablation)

    p = sub.add_parser("collect-route-b-results")
    p.add_argument("--summaries", nargs="+", required=True, help="Route B pipeline or tracklet ablation summary JSON files")
    p.add_argument("--out-dir", required=True, help="Directory for route_b_results_table.csv and route_b_results_summary.json")
    p.add_argument("--dataset-names", nargs="+", default=None, help="Optional dataset names, one per summary JSON")
    p.set_defaults(func=cmd_collect_route_b_results)

    p = sub.add_parser("compare-route-b-baselines")
    p.add_argument("--route-b-csv", required=True, help="route_b_results_table.csv from collect-route-b-results")
    p.add_argument("--baseline-csv", required=True, help="CSV with at least dataset,method,<metric> columns")
    p.add_argument("--out-dir", required=True, help="Directory for baseline comparison CSV/JSON outputs")
    p.add_argument("--metric", default="best_f1", help="Metric column to compare, default: best_f1")
    p.add_argument("--lower-is-better", action="store_true", help="Use when lower metric values are better")
    p.set_defaults(func=cmd_compare_route_b_baselines)

    p = sub.add_parser("write-route-b-baseline-template")
    p.add_argument("--out", required=True, help="Output baseline CSV template")
    p.add_argument("--datasets", nargs="+", default=None, help="Dataset names to include")
    p.add_argument("--methods", nargs="+", default=None, help="Baseline method names to include")
    p.set_defaults(func=cmd_write_route_b_baseline_template)

    p = sub.add_parser("write-route-b-official-baseline-seed")
    p.add_argument("--out", required=True, help="Output provenance-heavy baseline seed CSV")
    p.add_argument("--no-placeholders", action="store_true", help="Only write source-backed rows with filled values")
    p.set_defaults(func=cmd_write_route_b_official_baseline_seed)

    p = sub.add_parser("validate-route-b-baselines")
    p.add_argument("--baseline-csv", required=True, help="Baseline CSV to validate")
    p.add_argument("--out", required=True, help="Output validation JSON")
    p.add_argument("--metric", default="best_f1", help="Metric column to validate, default: best_f1")
    p.add_argument("--allow-empty-metric", action="store_true", help="Warn instead of error when metric cells are empty")
    p.add_argument("--strict", action="store_true", help="Exit non-zero when validation fails")
    p.set_defaults(func=cmd_validate_route_b_baselines)

    p = sub.add_parser("export-route-b-baseline-table")
    p.add_argument("--comparison-summary", required=True, help="route_b_baseline_comparison_summary.json")
    p.add_argument("--out", required=True, help="Output Markdown table path")
    p.add_argument("--digits", type=int, default=3)
    p.set_defaults(func=cmd_export_route_b_baseline_table)

    p = sub.add_parser("build-route-b-report")
    p.add_argument("--summaries", nargs="+", required=True, help="Route B pipeline, ablation, or action-prior sweep summary JSON files")
    p.add_argument("--baseline-csv", required=True, help="Validated baseline CSV with dataset,method,<metric>")
    p.add_argument("--out-dir", required=True, help="Directory for collected results, comparison outputs, and Markdown report")
    p.add_argument("--dataset-names", nargs="+", default=None, help="Optional dataset names, one per summary JSON")
    p.add_argument("--metric", default="best_f1")
    p.add_argument("--lower-is-better", action="store_true")
    p.add_argument("--digits", type=int, default=3)
    p.add_argument("--allow-invalid-baselines", action="store_true", help="Continue even when baseline validation fails")
    p.set_defaults(func=cmd_build_route_b_report)

    p = sub.add_parser("eval-action-dynamics-thresholds")
    p.add_argument("--dynamics-scores", required=True, help="Output JSONL from score-action-chunk-tracklets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.set_defaults(func=cmd_eval_action_dynamics_thresholds)

    p = sub.add_parser("export-tracklet-jsonl-classifier-dataset")
    p.add_argument("--tracklet-jsonl", required=True, help="Nested tracklets/proposal_tracklets JSONL, optionally with action dynamics attached")
    p.add_argument("--out", required=True, help="Output directory for classifier tracklets.csv/jsonl")
    p.add_argument("--dataset-source", default=None, help="Optional dataset/source tag, e.g. nps, aot, ard100, transvisdrone")
    p.set_defaults(func=cmd_export_tracklet_jsonl_classifier_dataset)

    p = sub.add_parser("merge-tracklet-classifier-datasets")
    p.add_argument("--inputs", nargs="+", required=True, help="Per-source classifier tracklets.csv files")
    p.add_argument("--out", required=True, help="Merged classifier CSV path")
    p.add_argument("--source-names", nargs="+", default=None, help="Optional dataset_source override, one per input")
    p.add_argument("--manifest-out", default=None, help="Optional output manifest JSON path")
    p.set_defaults(func=cmd_merge_tracklet_classifier_datasets)

    p = sub.add_parser("validate-tracklet-classifier-mixture-inputs")
    p.add_argument("--train-csvs", nargs="+", required=True, help="Per-source train classifier CSVs")
    p.add_argument("--eval-csvs", nargs="+", required=True, help="Held-out eval classifier CSVs")
    p.add_argument("--out", required=True, help="Output preflight JSON path")
    p.add_argument("--train-source-names", nargs="+", default=None)
    p.add_argument("--eval-dataset-names", nargs="+", default=None)
    p.add_argument("--min-train-rows", type=int, default=1)
    p.add_argument("--min-eval-rows", type=int, default=1)
    p.add_argument("--min-train-positives", type=int, default=1)
    p.add_argument("--min-eval-positives", type=int, default=1)
    p.add_argument("--allow-train-without-negatives", action="store_true")
    p.add_argument("--allow-eval-without-negatives", action="store_true")
    p.add_argument("--allow-train-eval-sequence-overlap", action="store_true")
    p.add_argument("--allow-invalid", action="store_true", help="Write report and exit 0 even if preflight fails")
    p.set_defaults(func=cmd_validate_tracklet_classifier_mixture_inputs)

    p = sub.add_parser("train-tracklet-classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hard-tiny-positive-augments", type=int, default=0, help="Synthetic hard-tiny positive variants per positive tracklet")
    p.add_argument("--balance-by", nargs="*", default=None, help="Metadata fields for inverse-frequency sample balancing, e.g. dataset_source bucket label")
    p.set_defaults(func=cmd_train_tracklet_classifier)

    p = sub.add_parser("eval-tracklet-classifier")
    p.add_argument("--csv", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=cmd_eval_tracklet_classifier)

    p = sub.add_parser("eval-tracklet-classifier-thresholds")
    p.add_argument("--csv", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.set_defaults(func=cmd_eval_tracklet_classifier_thresholds)

    p = sub.add_parser("run-tracklet-classifier-mixture-benchmark")
    p.add_argument("--train-csvs", nargs="+", required=True, help="Per-source train classifier CSVs")
    p.add_argument("--eval-csvs", nargs="+", required=True, help="Held-out eval classifier CSVs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-source-names", nargs="+", default=None)
    p.add_argument("--eval-dataset-names", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hard-tiny-positive-augments", type=int, default=0)
    p.add_argument("--balance-by", nargs="*", default=None)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--allow-invalid-preflight", action="store_true")
    p.add_argument("--allow-train-eval-sequence-overlap", action="store_true")
    p.add_argument("--baseline-csv", default=None, help="Optional baseline CSV with dataset,method,<metric> columns")
    p.add_argument("--baseline-metric", default="tracklet_best_f1", help="Metric column to compare, default is tracklet_best_f1")
    p.add_argument("--baseline-lower-is-better", action="store_true")
    p.add_argument("--baseline-digits", type=int, default=3)
    p.add_argument("--allow-invalid-baselines", action="store_true")
    p.set_defaults(func=cmd_run_tracklet_classifier_mixture_benchmark)

    p = sub.add_parser("run-tracklet-classifier-frame-benchmark")
    p.add_argument("--run-roots", nargs="+", required=True, help="Inference output roots containing predictions.jsonl/diagnostics.jsonl, or roots with per-sequence subdirs")
    p.add_argument("--gt-csvs", nargs="+", required=True, help="GT CSVs aligned one-to-one with run roots")
    p.add_argument("--weights", required=True, help="Joint tracklet classifier checkpoint")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-names", nargs="+", default=None)
    p.add_argument("--prediction-name", default="predictions.jsonl")
    p.add_argument("--diagnostics-name", default="diagnostics.jsonl")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--untracked-policy", choices=["keep", "suppress"], default="keep")
    p.add_argument("--disable-tracklet-promotion", action="store_true")
    p.add_argument("--promotion-score-floor", type=float, default=0.22)
    p.add_argument("--promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--promotion-max-background", type=float, default=0.68)
    p.add_argument("--selective-promotion", action="store_true")
    p.add_argument("--selective-min-temporal-crop-delta", type=float, default=0.05)
    p.add_argument("--selective-min-temporal-background-margin", type=float, default=-0.05)
    p.add_argument("--selective-max-tracklet-background", type=float, default=0.60)
    p.add_argument("--selective-max-tracklet-objectness", type=float, default=0.50)
    p.add_argument("--selective-min-tracklet-rows", type=int, default=2)
    p.add_argument("--selective-min-temporal-gain-rate", type=float, default=0.40)
    p.add_argument("--selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    p.add_argument("--selective-allow-non-recovery-source", action="store_true")
    p.add_argument("--selective-max-promoted-tracklets-per-sequence", type=int, default=2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--score-threshold", type=float, default=0.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--baseline-csv", default=None, help="Optional frame-level baseline CSV with dataset,method,<metric>")
    p.add_argument("--baseline-metric", default="frame_best_f1")
    p.add_argument("--baseline-lower-is-better", action="store_true")
    p.add_argument("--baseline-digits", type=int, default=3)
    p.add_argument("--allow-invalid-baselines", action="store_true")
    p.set_defaults(func=cmd_run_tracklet_classifier_frame_benchmark)

    p = sub.add_parser("validate-tracklet-classifier-frame-benchmark-inputs")
    p.add_argument("--run-roots", nargs="+", required=True, help="Inference output roots containing predictions.jsonl/diagnostics.jsonl, or roots with per-sequence subdirs")
    p.add_argument("--gt-csvs", nargs="+", required=True, help="GT CSVs aligned one-to-one with run roots")
    p.add_argument("--weights", required=True, help="Joint tracklet classifier checkpoint")
    p.add_argument("--out", required=True, help="Output preflight JSON path")
    p.add_argument("--dataset-names", nargs="+", default=None)
    p.add_argument("--prediction-name", default="predictions.jsonl")
    p.add_argument("--diagnostics-name", default="diagnostics.jsonl")
    p.add_argument("--thresholds", nargs="*", type=float, default=None)
    p.add_argument("--baseline-csv", default=None, help="Optional frame-level baseline CSV with dataset,method,<metric>")
    p.add_argument("--baseline-metric", default="frame_best_f1")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--min-prediction-rows", type=int, default=1)
    p.add_argument("--min-gt-boxes", type=int, default=1)
    p.add_argument("--allow-missing-diagnostics", action="store_true")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_validate_tracklet_classifier_frame_benchmark_inputs)

    p = sub.add_parser("build-tracklet-classifier-official-eval-bundle")
    p.add_argument("--frame-summary", required=True, help="tracklet_classifier_frame_benchmark_summary.json")
    p.add_argument("--out-dir", required=True, help="Output bundle directory")
    p.add_argument("--preflight-json", default=None, help="Optional frame benchmark preflight JSON")
    p.add_argument("--baseline-comparison-json", default=None, help="Optional baseline_report/comparison/route_b_baseline_comparison_summary.json")
    p.add_argument("--no-copy-predictions", action="store_true", help="Reference best filtered predictions in place instead of copying to the bundle")
    p.add_argument("--allow-missing-or-invalid-preflight", action="store_true")
    p.add_argument("--require-baseline-comparison", action="store_true")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_build_tracklet_classifier_official_eval_bundle)

    p = sub.add_parser("export-tracklet-classifier-official-predictions")
    p.add_argument("--bundle-manifest", required=True, help="official_eval_bundle_manifest.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--formats", nargs="+", default=["flat_csv", "yolo_txt"], choices=["flat_csv", "yolo_txt"])
    p.add_argument("--image-width", type=int, default=None, help="Fallback image width for YOLO txt normalization")
    p.add_argument("--image-height", type=int, default=None, help="Fallback image height for YOLO txt normalization")
    p.add_argument("--score-field", default="final_drone_score")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--class-id", type=int, default=0)
    p.add_argument("--include-background", action="store_true")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_export_tracklet_classifier_official_predictions)

    p = sub.add_parser("export-tracklet-classifier-aot-predictions")
    p.add_argument("--flat-csv", required=True, help="flat_xyxy_predictions.csv from export-tracklet-classifier-official-predictions")
    p.add_argument("--out-dir", required=True, help="Output directory containing aotpredictions/")
    p.add_argument("--image-name-template", default="{image_stem}.png", help="Python format string using image_stem,dataset,seq,frame_id,frame_id_05d,frame_id_06d")
    p.add_argument("--image-name-mode", default="template", choices=["template", "aot_clip_frame"], help="Use template formatting, or force Clip_<clip_id>_<frame:05d>.png for AOT official eval")
    p.add_argument("--frame-id-offset", type=int, default=0, help="Offset applied in aot_clip_frame mode before zero-padding frame_id")
    p.add_argument("--part-name", default="predictions_split_0.pkl")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--score-field", default="score")
    p.add_argument("--class-name", default="airborne")
    p.add_argument("--no-group-by-image", action="store_true")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_export_tracklet_classifier_aot_predictions)

    p = sub.add_parser("validate-tracklet-classifier-aot-eval-inputs")
    p.add_argument("--results-folder", required=True, help="Folder containing AOT predictions_split_*.pkl files")
    p.add_argument("--out", required=True)
    p.add_argument("--clip-id-to-flight-id-path", default=None, help="Optional aot_clip_id_to_flight_id.pkl")
    p.add_argument("--allow-non-clip-names", action="store_true")
    p.add_argument("--allow-unknown-clip-ids", action="store_true")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_validate_tracklet_classifier_aot_eval_inputs)

    p = sub.add_parser("export-aot-prediction-parts-to-tracklets")
    p.add_argument("--results-folder", required=True, help="Folder containing AOT predictions_split_*.pkl files")
    p.add_argument("--out", required=True, help="Output Route B tracklet JSONL")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--dataset-source", default="aot")
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=1)
    p.add_argument("--max-frame-gap", type=int, default=None, help="Split a raw track_id into new segments when consecutive frames are farther apart")
    p.add_argument("--clip-id-to-flight-id-path", default=None, help="Optional AOT clip-id to flight-id pickle; enables flight_id fields")
    p.add_argument("--aot-groundtruth-json", default=None, help="Optional AOT groundtruth.json; enables per-row image_path fields")
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_export_aot_prediction_parts_to_tracklets)

    p = sub.add_parser("filter-aot-prediction-parts-by-tracklets")
    p.add_argument("--results-folder", required=True, help="Source folder containing AOT predictions_split_*.pkl files")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL allowlist, optionally with action scores")
    p.add_argument("--out-dir", required=True, help="Output directory containing aotpredictions/")
    p.add_argument("--part-name", default="predictions_split_0.pkl")
    p.add_argument("--score-field", default=None, help="Optional tracklet score field to threshold, for example action_dynamics_score")
    p.add_argument("--min-tracklet-score", type=float, default=None)
    p.add_argument("--min-tracklet-rows", type=int, default=1)
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_filter_aot_prediction_parts_by_tracklets)

    p = sub.add_parser("rescore-aot-prediction-parts-by-tracklets")
    p.add_argument("--results-folder", required=True, help="Source folder containing AOT predictions_split_*.pkl files")
    p.add_argument("--tracklet-jsonl", required=True, help="Route B tracklet JSONL with video/action tracklet scores")
    p.add_argument("--out-dir", required=True, help="Output directory containing rescored aotpredictions/")
    p.add_argument("--part-name", default="predictions_split_0.pkl")
    p.add_argument("--score-field", default="video_action_model_fusion_score")
    p.add_argument("--center", type=float, default=0.486)
    p.add_argument("--beta", type=float, default=0.4)
    p.add_argument("--mode", choices=["additive", "suppress-only", "boost-only"], default="suppress-only")
    p.add_argument("--min-tracklet-rows", type=int, default=1)
    p.add_argument("--missing-score-behavior", choices=["keep", "error"], default="keep")
    p.add_argument("--protect-raw-score-at", type=float, default=None)
    p.add_argument("--clip-min", type=float, default=0.0)
    p.add_argument("--clip-max", type=float, default=1.0)
    p.add_argument("--allow-invalid", action="store_true")
    p.set_defaults(func=cmd_rescore_aot_prediction_parts_by_tracklets)

    p = sub.add_parser("stage-b-oracle-benchmark")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--crop-weights", default=None)
    p.add_argument("--feature-weights", default=None)
    p.add_argument("--temporal-weights", default=None)
    p.add_argument("--hard-negative-manifest", nargs="*", default=None, help="Optional hard_negative_manifest.jsonl files to include as oracle negatives")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-hard-negatives", type=int, default=None)
    p.set_defaults(func=cmd_stage_b_oracle_benchmark)

    p = sub.add_parser("eval-proposal-stage-b")
    p.add_argument("--manifest", required=True, help="proposal_manifest.jsonl from build-real-detector-proposal-stage-b")
    p.add_argument("--crop-weights", required=True, help="CropRecognizer .pt checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_eval_proposal_stage_b)

    p = sub.add_parser("tracker-oracle-benchmark")
    p.add_argument("--metadata", nargs="+", required=True, help="JSON metadata files or glob patterns with oracle boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--detection-stride", type=int, default=5, help="Only feed oracle detections every N frames")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=16.0)
    p.add_argument("--tracker-r0", type=float, default=24.0)
    p.add_argument("--tracker-alpha", type=float, default=1.5)
    p.add_argument("--tracker-beta", type=float, default=20.0)
    p.add_argument("--tracker-reacquire", type=float, default=18.0)
    p.set_defaults(func=cmd_tracker_oracle_benchmark)

    p = sub.add_parser("mode-sweep")
    p.add_argument("--diagnostics", required=True, help="motion-debug diagnostics.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--weak-alignment-q-threshold", type=float, default=0.66)
    p.add_argument("--high-residual-threshold", type=float, default=0.0072)
    p.add_argument("--static-motion-threshold", type=float, default=0.08)
    p.set_defaults(func=cmd_mode_sweep)

    p = sub.add_parser("augment-speed")
    p.add_argument("--input-dir", default=None, help="Directory of videos to batch augment")
    p.add_argument("--pattern", default="*.mp4", help="Glob pattern for --input-dir")
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--video", default=None, help="Input video to temporally subsample")
    p.add_argument("--annotations", default=None, help="Optional annotation CSV to subsample by frame_index")
    p.add_argument("--out", required=True)
    p.add_argument("--strides", nargs="+", type=int, default=[2, 4], help="Temporal stride values; speed factor when --keep-fps is set")
    p.add_argument("--keep-fps", action="store_true", default=True, help="Keep original FPS so apparent motion is faster")
    p.add_argument("--motion-blur", type=int, default=0, help="Optional odd/even kernel size; 0 disables blur")
    p.add_argument("--blur-direction", choices=["horizontal", "vertical", "diagonal"], default="horizontal")
    p.add_argument("--frame-index-column", default="frame_index")
    p.set_defaults(func=cmd_augment_speed)

    p = sub.add_parser("make-static-hover")
    p.add_argument("--video", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--x", type=int, default=None)
    p.add_argument("--y", type=int, default=None)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--color", nargs=3, type=int, default=[15, 15, 15], help="BGR dot color")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--freeze-background", action="store_true", help="Repeat the first frame before drawing the fixed target")
    p.add_argument("--jitter-px", type=int, default=0, help="Per-frame integer jitter range around the target center")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_make_static_hover)

    p = sub.add_parser("make-moving-target")
    p.add_argument("--video", required=True)
    p.add_argument("--out-video", required=True)
    p.add_argument("--start-x", type=int, default=None)
    p.add_argument("--start-y", type=int, default=None)
    p.add_argument("--vx", type=float, default=16.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--color", nargs=3, type=int, default=[0, 0, 0], help="BGR dot color")
    p.add_argument("--max-frames", type=int, default=80)
    p.add_argument("--freeze-background", action="store_true", default=True)
    p.set_defaults(func=cmd_make_moving_target)

    p = sub.add_parser("build-static-hover-crops")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--class-name", default="drone")
    p.add_argument("--negative-class", default="background")
    p.add_argument("--negative-per-positive", type=int, default=3)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seed", type=int, default=17)
    p.set_defaults(func=cmd_build_static_hover_crops)

    p = sub.add_parser("mine-hard-negative-crops")
    p.add_argument("--video", nargs="+", required=True, help="Videos or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-metadata", nargs="*", default=None, help="Static-hover JSON metadata boxes to avoid")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-crops-per-video", type=int, default=300)
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--max-area", type=int, default=5000)
    p.add_argument("--min-motion-score", type=float, default=0.08)
    p.add_argument("--artifact-q-threshold", type=float, default=0.66)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--exclude-padding-px", type=float, default=24.0)
    p.set_defaults(func=cmd_mine_hard_negative_crops)

    p = sub.add_parser("build-static-hover-yolo-dataset")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-frames-per-video", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-box-px", type=float, default=1.0, help="Expand tiny labels to at least this many pixels for candidate training")
    p.add_argument("--tiled", action="store_true", help="Build tiled/high-res YOLO dataset instead of full-frame dataset")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--negative-pad-px", type=float, default=32.0)
    p.add_argument("--photometric-augmentations", type=int, default=0, help="Extra brightness/contrast variants per positive tile")
    p.add_argument("--low-contrast-injections", type=int, default=0, help="Extra positive variants with a locally redrawn low-contrast tiny target")
    p.add_argument("--positive-repeat-pattern", action="append", default=[], help="Repeat positives when this substring appears in the source frame path; can be passed multiple times")
    p.add_argument("--positive-repeat-factor", type=int, default=1, help="Positive oversampling factor for matching repeat patterns")
    p.set_defaults(func=cmd_build_static_hover_yolo_dataset)

    p = sub.add_parser("build-detector-proposal-stage-b")
    p.add_argument("--metadata", nargs="+", required=True, help="Synthetic metadata JSON files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--max-frames-per-video", type=int, default=80)
    p.add_argument("--max-proposals-per-frame", type=int, default=8)
    p.add_argument("--max-negatives-per-frame", type=int, default=4)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.set_defaults(func=cmd_build_detector_proposal_stage_b)

    p = sub.add_parser("build-static-hover-temporal")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=29)
    p.set_defaults(func=cmd_build_static_hover_temporal)

    p = sub.add_parser("build-static-hover-feature-dataset")
    p.add_argument("--metadata", nargs="+", required=True, help="Static-hover JSON metadata files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--seed", type=int, default=37)
    p.set_defaults(func=cmd_build_static_hover_feature_dataset)

    p = sub.add_parser("mine-hard-negative-temporal")
    p.add_argument("--manifest", required=True, help="hard_negative_manifest.jsonl from mine-hard-negative-crops")
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--scale", type=float, default=8.0)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--max-samples-per-class", type=int, default=80)
    p.set_defaults(func=cmd_mine_hard_negative_temporal)

    p = sub.add_parser("split-folder-dataset")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=53)
    p.set_defaults(func=cmd_split_folder_dataset)

    p = sub.add_parser("split-feature-csv")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=59)
    p.set_defaults(func=cmd_split_feature_csv)

    p = sub.add_parser("build-hard-negative-feature-dataset")
    p.add_argument("--manifest", nargs="+", required=True, help="Hard-negative manifest files or glob patterns")
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples-per-class", type=int, default=120)
    p.set_defaults(func=cmd_build_hard_negative_feature_dataset)

    p = sub.add_parser("stage-a-yolo-recall")
    p.add_argument("--metadata", nargs="+", required=True, help="Synthetic metadata JSON files with output video and boxes")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--keep-top", type=int, default=5)
    p.set_defaults(func=cmd_stage_a_yolo_recall)

    p = sub.add_parser("stage-a-real-yolo-recall")
    p.add_argument("--annotations", required=True, help="QSTR real CSV or extracted frame_annotations.csv")
    p.add_argument("--frames-root", default=None, help="Root for relative frame_path entries")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=192)
    p.add_argument("--device", default=None)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--match-iou", type=float, default=0.3)
    p.add_argument("--match-center-px", type=float, default=16.0)
    p.add_argument("--keep-top", type=int, default=5)
    p.add_argument("--class-name", default="drone")
    p.add_argument("--aug-speed", nargs="*", default=None, help="Optional aug_speed values to include, e.g. speedx2 speedx4")
    p.add_argument("--proposal-nms-iou", type=float, default=None, help="Optional global candidate NMS before recall/budget metrics")
    p.add_argument("--proposal-top-k", type=int, default=None, help="Optional per-frame top-k proposal budget after NMS")
    p.set_defaults(func=cmd_stage_a_real_yolo_recall)

    p = sub.add_parser("stage-a-real-yolo-fppi")
    p.add_argument("--annotations", required=True, help="QSTR real CSV used to exclude GT frames")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=192)
    p.add_argument("--device", default=None)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-videos", type=int, default=None)
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--exclude-radius", type=int, default=20, help="Skip frames within this many frames of any GT annotation")
    p.add_argument("--keep-top", type=int, default=5)
    p.add_argument("--class-name", default="drone")
    p.add_argument("--proposal-nms-iou", type=float, default=None)
    p.add_argument("--proposal-top-k", type=int, default=None)
    p.set_defaults(func=cmd_stage_a_real_yolo_fppi)

    p = sub.add_parser("init-real-data-layout")
    p.add_argument("--root", default="data/real")
    p.set_defaults(func=cmd_init_real_data_layout)

    p = sub.add_parser("extract-real-annotated-frames")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--frames-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--video-root", default=None)
    p.add_argument("--allow-unknown-labels", action="store_true")
    p.set_defaults(func=cmd_extract_real_annotated_frames)

    p = sub.add_parser("prepare-real-yolo-dataset")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--video-root", default=None)
    p.add_argument("--full-frame", action="store_true", help="Build full-frame YOLO dataset instead of tiled dataset")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--positives-per-box", type=int, default=2)
    p.add_argument("--negatives-per-image", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--min-box-px", type=float, default=8.0)
    p.add_argument("--negative-pad-px", type=float, default=32.0)
    p.add_argument("--allow-unknown-labels", action="store_true")
    p.set_defaults(func=cmd_prepare_real_yolo_dataset)

    p = sub.add_parser("export-anti-uav300-subset")
    p.add_argument("--zip", required=True, help="Path to Anti-UAV300.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--modality", choices=["visible", "infrared"], default="visible")
    p.add_argument("--max-sequences", type=int, default=5)
    p.add_argument("--start-index", type=int, default=0, help="Skip this many sorted sequences before exporting")
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--max-frames-per-sequence", type=int, default=80)
    p.set_defaults(func=cmd_export_anti_uav300_subset)

    p = sub.add_parser("export-ard100-annotations")
    p.add_argument("--root", required=True, help="ARD100 root containing train_videos, test_videos, and annotations.zip")
    p.add_argument("--out", required=True)
    p.add_argument("--annotations-zip", default=None, help="Optional explicit annotations.zip path")
    p.add_argument("--split", choices=["all", "train", "test"], default="all")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--max-frames-per-sequence", type=int, default=None)
    p.add_argument("--tiny-side-px", type=float, default=24.0)
    p.add_argument("--default-tag", choices=sorted(["static_hovering", "fast_target", "bad_alignment", "tiny", "hard_negative"]), default="fast_target")
    p.set_defaults(func=cmd_export_ard100_annotations)

    p = sub.add_parser("build-real-stage-b-dataset")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--negative-per-positive", type=int, default=1)
    p.add_argument("--crop-scale", type=float, default=4.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--tube-t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_build_real_stage_b_dataset)

    p = sub.add_parser("build-real-detector-proposal-stage-b")
    p.add_argument("--annotations", required=True, help="CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--yolo-conf", type=float, default=0.05)
    p.add_argument("--fallback-yolo-weights", default=None)
    p.add_argument("--fallback-yolo-conf", type=float, default=0.15)
    p.add_argument("--yolo-tile-size", type=int, default=256)
    p.add_argument("--yolo-tile-stride", type=int, default=128)
    p.add_argument("--device", default="0")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-proposals-per-frame", type=int, default=8)
    p.add_argument("--max-fallback-proposals-per-frame", type=int, default=4)
    p.add_argument("--proposal-nms-iou", type=float, default=0.35)
    p.add_argument("--max-negatives-per-frame", type=int, default=4)
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--artifact-score-threshold", type=float, default=0.25, help="Nonmatched fallback proposals above this score become alignment_artifact samples")
    p.add_argument("--high-score-fp-threshold", type=float, default=0.5, help="Nonmatched detector proposals above this score become high_score_detector_fp diagnostic samples")
    p.add_argument("--non-drone-label-mode", choices=["multiclass_artifact", "binary_buckets"], default="multiclass_artifact", help="binary_buckets trains all non-drone proposals as background while preserving diagnostic_bucket metadata")
    p.add_argument("--hard-positive-max-size-px", type=float, default=24.0, help="Matched positives with proposal max side at or below this are hard_tiny_positive candidates")
    p.add_argument("--hard-positive-max-score", type=float, default=0.5, help="Matched tiny positives at or below this score are repeated as hard positives")
    p.add_argument("--hard-positive-repeat", type=int, default=1, help="Repeat each hard tiny positive this many times in Stage B datasets")
    p.add_argument("--crop-scale", type=float, default=2.0)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--tube-t", type=int, default=5)
    p.add_argument("--tube-size", type=int, default=96)
    p.set_defaults(func=cmd_build_real_detector_proposal_stage_b)

    p = sub.add_parser("calibrate-fusion")
    p.add_argument("--diagnostics", required=True)
    p.add_argument("--gt", required=True, help="QSTR real CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--video", default=None, help="Optional video path when diagnostics come from one video")
    p.add_argument("--match-iou", type=float, default=0.1)
    p.add_argument("--match-center-px", type=float, default=24.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--frame-tolerance", type=int, default=0, help="Allow nearest GT annotation within this many frames")
    p.set_defaults(func=cmd_calibrate_fusion)

    p = sub.add_parser("analyze-frame-failures")
    p.add_argument("--run-root", required=True, help="Profile benchmark output root containing hard_recovery/<seq>/predictions*.jsonl")
    p.add_argument("--gt", required=True, help="QSTR real CSV with video_path,frame_id,x1,y1,x2,y2,class,tag")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--raw-prediction-name", default="predictions_raw.jsonl")
    p.add_argument("--filtered-prediction-name", default="predictions.jsonl")
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_analyze_frame_failures)

    p = sub.add_parser("sweep-tracklet-filter")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots; do not pass frozen test roots for calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--weights", required=True, help="Tracklet classifier checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85, 0.95])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.20, 0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60, 0.68])
    p.add_argument("--promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--selective-promotion", action="store_true")
    p.add_argument("--selective-min-temporal-crop-deltas", nargs="+", type=float, default=[0.05])
    p.add_argument("--selective-min-temporal-background-margins", nargs="+", type=float, default=[-0.05])
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[2])
    p.add_argument("--selective-max-tracklet-background", type=float, default=0.60)
    p.add_argument("--selective-max-tracklet-objectness", type=float, default=0.50)
    p.add_argument("--selective-min-tracklet-rows", type=int, default=2)
    p.add_argument("--selective-min-temporal-gain-rate", type=float, default=0.40)
    p.add_argument("--selective-min-weak-detector-temporal-signal", type=float, default=0.05)
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_sweep_tracklet_filter)

    p = sub.add_parser("select-tracklet-model")
    p.add_argument("--tracklets", required=True, help="Tracklet CSV built from train/adapt diagnostics")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots used for downstream calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--calib-seqs", nargs="+", default=None, help="Exact sequence names reserved for calibration")
    p.add_argument("--calib-seq-patterns", nargs="+", default=None, help="fnmatch or substring patterns reserved for calibration")
    p.add_argument("--calib-fraction", type=float, default=0.4, help="Fallback deterministic sequence fraction when no calibration names/patterns are given")
    p.add_argument("--epochs-values", nargs="+", type=int, default=[30, 60])
    p.add_argument("--hidden-values", nargs="+", type=int, default=[32])
    p.add_argument("--lr-values", nargs="+", type=float, default=[1e-3])
    p.add_argument("--hard-tiny-positive-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--hard-negative-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85])
    p.add_argument("--promotion-enabled-values", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60])
    p.add_argument("--promotion-min-branch-drone", type=float, default=0.40)
    p.add_argument("--selective-promotion", action="store_true", help="Apply selective promotion budget while evaluating promoted candidates")
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[1, 2])
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-recall-drop", type=float, default=0.02, help="Selection prefers configs within this recall drop from raw before minimizing FP")
    p.set_defaults(func=cmd_select_tracklet_model)

    p = sub.add_parser("select-tracklet-sequence-model")
    p.add_argument("--tracklets-jsonl", required=True, help="Tracklet JSONL built from train/adapt diagnostics")
    p.add_argument("--run-roots", nargs="+", required=True, help="Train/adapt profile output roots used for downstream calibration")
    p.add_argument("--gt", required=True, help="QSTR real CSV covering the run roots")
    p.add_argument("--out", required=True)
    p.add_argument("--profile", default="hard_recovery")
    p.add_argument("--calib-seqs", nargs="+", default=None)
    p.add_argument("--calib-seq-patterns", nargs="+", default=None)
    p.add_argument("--calib-fraction", type=float, default=0.4)
    p.add_argument("--epochs-values", nargs="+", type=int, default=[30])
    p.add_argument("--hidden-values", nargs="+", type=int, default=[32])
    p.add_argument("--max-len-values", nargs="+", type=int, default=[12, 24])
    p.add_argument("--hard-positive-augments-values", nargs="+", type=int, default=[0], help="Oversample/degrade hard positive tracklets during sequence training")
    p.add_argument("--hard-negative-augments-values", nargs="+", type=int, default=[0, 2])
    p.add_argument("--classifier-thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.85])
    p.add_argument("--promotion-enabled-values", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--promotion-score-floors", nargs="+", type=float, default=[0.22, 0.30])
    p.add_argument("--promotion-max-backgrounds", nargs="+", type=float, default=[0.55, 0.60])
    p.add_argument("--selective-promotion", action="store_true")
    p.add_argument("--selective-max-promoted-tracklets-per-sequence-values", nargs="+", type=int, default=[1, 2])
    p.add_argument("--score-threshold", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.3)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-recall-drop", type=float, default=0.02)
    p.set_defaults(func=cmd_select_tracklet_sequence_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


