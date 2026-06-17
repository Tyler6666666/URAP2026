from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BBox = tuple[float, float, float, float]


@dataclass
class OfflineSelectorResult:
    trajectory_csv: Path
    debug_csv: Path
    summary_json: Path
    annotated_video: Path | None
    summary: dict[str, Any]


def replay_offline_selector(
    predictions_jsonl: str | Path,
    out_dir: str | Path,
    *,
    top_k: int = 5,
    max_frame_id: int | None = None,
    max_jump_px: float = 48.0,
    max_recover_frames: int = 3,
    min_accept_score: float = 0.25,
    min_motion_consistency: float = 0.15,
    min_memory_consistency: float = 0.15,
    min_topk_margin: float = 0.0,
    detector_weight: float = 0.55,
    motion_weight: float = 0.25,
    memory_weight: float = 0.20,
    reacquire_top_k: int = 0,
    reacquire_max_distance_px: float = 96.0,
    reacquire_min_detector_score: float = 0.10,
    reacquire_min_motion_consistency: float = 0.30,
    reacquire_min_memory_consistency: float = 0.20,
    reacquire_min_side_ratio: float = 0.35,
    reacquire_max_side_ratio: float = 2.50,
    reacquire_confirm_frames: int = 2,
    reacquire_stale_after_frames: int = -1,
    reacquire_global_min_area: float = 0.0,
    reacquire_global_small_min_area: float = 0.0,
    reacquire_global_small_max_distance_px: float = 0.0,
    reacquire_global_small_memory_probation_frames: int = 0,
    reacquire_global_max_area: float = 0.0,
    reacquire_global_min_detector_score: float = 0.10,
    reacquire_global_require_tracklet_confirmation: bool = False,
    reacquire_global_reject_tracklet_rejected: bool = False,
    reacquire_global_min_tracklet_score: float = 0.50,
    reacquire_global_delayed_confirm_frames: int = 0,
    reacquire_min_appearance_similarity: float = 0.0,
    reacquire_appearance_weight: float = 0.0,
    reacquire_crop_score_field: str = "",
    reacquire_crop_weight: float = 0.0,
    reacquire_min_crop_drone_score: float = 0.0,
    reacquire_crop_weights: str | Path | None = None,
    reacquire_crop_image_size: int = 128,
    appearance_memory_size: int = 8,
    video: str | Path | None = None,
    save_video: bool = False,
) -> OfflineSelectorResult:
    """Replay candidate-level QSTR predictions into one online trajectory.

    The replay is intentionally file based: it consumes existing infer outputs
    and does not rerun detectors or recognizers.
    """
    pred_path = Path(predictions_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows_by_frame = _load_rows_by_frame(pred_path)
    if max_frame_id is None:
        max_frame_id = max(rows_by_frame.keys(), default=-1)

    memory: list[tuple[int, BBox]] = []
    lost_streak = 0
    accepted_candidate_frames = 0
    memory_recover_frames = 0
    reacquired_frames = 0
    reacquire_pending_frames = 0
    reacquire_confirmed_frames = 0
    reacquire_memory_frames = 0
    reacquire_global_frames = 0
    reacquire_global_small_area_frames = 0
    reacquire_memory_probation_rejected_frames = 0
    memory_recover_probation_rejected_frames = 0
    appearance_rejected_frames = 0
    appearance_soft_scored_frames = 0
    crop_rejected_frames = 0
    crop_soft_scored_frames = 0
    lost_frames = 0
    fallback_needed_frames = 0
    trajectory_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    pending_reacquire: dict[str, Any] | None = None
    small_global_memory_probation_until_frame = -1
    small_global_memory_recover_grace_until_frame = -1
    confirm_frames = max(1, int(reacquire_confirm_frames))
    global_confirm_frames = (
        max(confirm_frames, int(reacquire_global_delayed_confirm_frames))
        if reacquire_global_delayed_confirm_frames > 0
        else confirm_frames
    )
    appearance_memory: list[list[float]] = []
    appearance_stats = {"rejected": 0}
    crop_stats = {"rejected": 0}
    global_tracklet_stats = {"rejected": 0}
    crop_score_field = str(reacquire_crop_score_field or "")
    crop_weights_path = Path(reacquire_crop_weights) if reacquire_crop_weights else None
    if (reacquire_crop_weight > 0 or reacquire_min_crop_drone_score > 0) and not crop_score_field and crop_weights_path is None:
        raise ValueError("crop scoring requires --reacquire-crop-score-field or --reacquire-crop-weights")
    if crop_weights_path is not None and video is None:
        raise ValueError("--video is required when crop recognizer weights are enabled")
    crop_scorer = (
        _CropRecognizerScorer(crop_weights_path, image_size=reacquire_crop_image_size)
        if crop_weights_path is not None
        else None
    )
    frame_reader = None
    appearance_enabled = reacquire_min_appearance_similarity > 0 or reacquire_appearance_weight > 0
    if appearance_enabled:
        if video is None:
            raise ValueError("--video is required when appearance similarity is enabled")
    if appearance_enabled or crop_scorer is not None:
        frame_reader = _SequentialVideoFrameReader(video)  # type: ignore[arg-type]

    for frame_id in range(max_frame_id + 1):
        frame_image = frame_reader.read(frame_id) if frame_reader is not None else None
        frame_rows = rows_by_frame.get(frame_id, [])
        candidates = _top_k_candidates(frame_rows, top_k)
        predicted_bbox = _predict_bbox(memory, frame_id)
        memory_bbox = _memory_bbox(memory)
        scored = [
            _score_candidate(
                row,
                predicted_bbox=predicted_bbox,
                memory_bbox=memory_bbox,
                detector_weight=detector_weight,
                motion_weight=motion_weight,
                memory_weight=memory_weight,
            )
            for row in candidates
        ]
        scored.sort(key=lambda item: item["selector_score"], reverse=True)

        decisions = [
            _candidate_decision(
                item,
                scored[idx + 1] if idx + 1 < len(scored) else None,
                has_memory=bool(memory),
                min_accept_score=min_accept_score,
                min_motion_consistency=min_motion_consistency,
                min_memory_consistency=min_memory_consistency,
                min_topk_margin=min_topk_margin,
                max_jump_px=max_jump_px,
            )
            for idx, item in enumerate(scored)
        ]
        selected_idx = next((idx for idx, decision in enumerate(decisions) if decision["accepted"]), None)
        selected_item = scored[selected_idx] if selected_idx is not None else None
        selected_decision = decisions[selected_idx] if selected_idx is not None else None
        best = selected_item if selected_item is not None else (scored[0] if scored else None)
        decision = selected_decision or (decisions[0] if decisions else {"accepted": False, "guard_reason": "no_candidate"})

        for rank, item in enumerate(scored, start=1):
            item_decision = decisions[rank - 1]
            debug_rows.append(
                _debug_row(
                    frame_id,
                    rank,
                    item,
                    selected=selected_item is item,
                    guard_reason=item_decision["guard_reason"],
                )
            )

        if pending_reacquire is not None and selected_item is not None and decision["accepted"]:
            confirming_item = dict(selected_item)
            confirming_item.setdefault("reacquire_mode", pending_reacquire["mode"])
            confirming_item.setdefault("reacquire_reason", pending_reacquire["reason"])
            confirming_item.setdefault(
                "global_small_area_candidate",
                bool(pending_reacquire.get("global_small_area_candidate", False)),
            )
            confirmation_features_ok = _apply_appearance_score(
                confirming_item,
                frame_image=frame_image,
                appearance_memory=appearance_memory,
                min_similarity=reacquire_min_appearance_similarity,
                appearance_weight=reacquire_appearance_weight,
                appearance_stats=appearance_stats,
            )
            if confirmation_features_ok:
                confirmation_features_ok = _apply_crop_score(
                    confirming_item,
                    frame_image=frame_image,
                    crop_score_field=crop_score_field,
                    crop_scorer=crop_scorer,
                    min_crop_drone_score=reacquire_min_crop_drone_score,
                    crop_weight=reacquire_crop_weight,
                    crop_stats=crop_stats,
                )
            if (
                confirmation_features_ok
                and pending_reacquire["mode"] == "global"
                and reacquire_global_require_tracklet_confirmation
            ):
                confirmation_features_ok = _apply_global_tracklet_confirmation(
                    confirming_item,
                    min_tracklet_score=reacquire_global_min_tracklet_score,
                    global_tracklet_stats=global_tracklet_stats,
                )
            if (
                confirmation_features_ok
                and pending_reacquire["mode"] == "global"
                and not reacquire_global_require_tracklet_confirmation
                and reacquire_global_reject_tracklet_rejected
            ):
                confirmation_features_ok = _apply_global_tracklet_rejection_veto(
                    confirming_item,
                    global_tracklet_stats=global_tracklet_stats,
                )
            if not confirmation_features_ok:
                pending_reacquire = None
                decision = {"accepted": False, "guard_reason": "reacquire_confirmation_feature_rejected"}
                selected_item = None
            else:
                selected_item = confirming_item
                next_pending = _advance_pending_reacquire(pending_reacquire, confirming_item, frame_id)
                _remember_global_tracklet_confirmation(
                    next_pending,
                    confirming_item,
                    min_tracklet_score=reacquire_global_min_tracklet_score,
                )
                required_confirm_frames = _required_reacquire_confirm_frames(
                    next_pending,
                    confirm_frames=confirm_frames,
                    global_confirm_frames=global_confirm_frames,
                )
                if next_pending["count"] >= required_confirm_frames:
                    bbox = selected_item["bbox"]
                    memory.append((frame_id, bbox))
                    memory = memory[-8:]
                    _remember_appearance(appearance_memory, frame_image, bbox, appearance_memory_size)
                    lost_streak = 0
                    accepted_candidate_frames += 1
                    reacquired_frames += 1
                    reacquire_confirmed_frames += 1
                    if next_pending["mode"] == "global":
                        reacquire_global_frames += 1
                        if bool(next_pending.get("global_small_area_candidate", False)):
                            reacquire_global_small_area_frames += 1
                            small_global_memory_probation_until_frame = _extend_memory_probation(
                                small_global_memory_probation_until_frame,
                                frame_id,
                                reacquire_global_small_memory_probation_frames,
                            )
                            small_global_memory_recover_grace_until_frame = max(
                                small_global_memory_recover_grace_until_frame,
                                frame_id + max(0, int(max_recover_frames)),
                            )
                    else:
                        reacquire_memory_frames += 1
                    confirmed_reason = f"confirmed_{next_pending['reason']}"
                    pending_reacquire = None
                    trajectory_rows.append(
                        _trajectory_row(
                            frame_id,
                            selected=True,
                            state="REACQUIRE",
                            selected_source=str(selected_item["row"].get("source", "")),
                            guard_reason="reacquire_confirmed",
                            bbox=bbox,
                            score_row=selected_item,
                            reacquire_reason=confirmed_reason,
                            reacquire_distance_px=selected_item["jump_px"],
                        )
                    )
                    debug_rows.append(
                        _debug_row(
                            frame_id,
                            0,
                            selected_item,
                            selected=True,
                            guard_reason="reacquire_confirmed",
                            reacquire_reason=confirmed_reason,
                            reacquire_distance_px=selected_item["jump_px"],
                        )
                    )
                    continue
                pending_reacquire = next_pending
                reacquire_pending_frames += 1
                decision = {"accepted": False, "guard_reason": "reacquire_pending_confirmation"}
                selected_item = None

        if selected_item is not None and decision["accepted"]:
            pending_reacquire = None
            bbox = best["bbox"]
            memory.append((frame_id, bbox))
            memory = memory[-8:]
            _remember_appearance(appearance_memory, frame_image, bbox, appearance_memory_size)
            lost_streak = 0
            accepted_candidate_frames += 1
            trajectory_rows.append(
                _trajectory_row(
                    frame_id,
                    selected=True,
                    state="TRACK",
                    selected_source=str(best["row"].get("source", "")),
                    guard_reason="accepted",
                    bbox=bbox,
                    score_row=best,
                )
            )
            continue

        reacquired = _find_reacquire_candidate(
            frame_rows,
            predicted_bbox=predicted_bbox,
            memory_bbox=memory_bbox,
            detector_weight=detector_weight,
            motion_weight=motion_weight,
            memory_weight=memory_weight,
            reacquire_top_k=reacquire_top_k,
            reacquire_max_distance_px=reacquire_max_distance_px,
            reacquire_min_detector_score=reacquire_min_detector_score,
            reacquire_min_motion_consistency=reacquire_min_motion_consistency,
            reacquire_min_memory_consistency=reacquire_min_memory_consistency,
            reacquire_min_side_ratio=reacquire_min_side_ratio,
            reacquire_max_side_ratio=reacquire_max_side_ratio,
            frame_image=frame_image,
            appearance_memory=appearance_memory,
            reacquire_min_appearance_similarity=reacquire_min_appearance_similarity,
            reacquire_appearance_weight=reacquire_appearance_weight,
            appearance_stats=appearance_stats,
            crop_score_field=crop_score_field,
            crop_scorer=crop_scorer,
            reacquire_crop_weight=reacquire_crop_weight,
            reacquire_min_crop_drone_score=reacquire_min_crop_drone_score,
            crop_stats=crop_stats,
        ) if frame_id > small_global_memory_probation_until_frame else None
        if frame_id <= small_global_memory_probation_until_frame and predicted_bbox is not None and frame_rows:
            reacquire_memory_probation_rejected_frames += 1
        if reacquired is None and reacquire_stale_after_frames >= 0 and lost_streak >= reacquire_stale_after_frames:
            reacquired = _find_global_reacquire_candidate(
                frame_rows,
                predicted_bbox=predicted_bbox,
                memory_bbox=memory_bbox,
                detector_weight=detector_weight,
                motion_weight=motion_weight,
                memory_weight=memory_weight,
                reacquire_min_detector_score=reacquire_global_min_detector_score,
                reacquire_global_min_area=reacquire_global_min_area,
                reacquire_global_small_min_area=reacquire_global_small_min_area,
                reacquire_global_small_max_distance_px=reacquire_global_small_max_distance_px,
                reacquire_global_max_area=reacquire_global_max_area,
                reacquire_global_require_tracklet_confirmation=reacquire_global_require_tracklet_confirmation,
                reacquire_global_reject_tracklet_rejected=reacquire_global_reject_tracklet_rejected,
                reacquire_global_min_tracklet_score=reacquire_global_min_tracklet_score,
                global_tracklet_stats=global_tracklet_stats,
                frame_image=frame_image,
                appearance_memory=appearance_memory,
                reacquire_min_appearance_similarity=reacquire_min_appearance_similarity,
                reacquire_appearance_weight=reacquire_appearance_weight,
                appearance_stats=appearance_stats,
                crop_score_field=crop_score_field,
                crop_scorer=crop_scorer,
                reacquire_crop_weight=reacquire_crop_weight,
                reacquire_min_crop_drone_score=reacquire_min_crop_drone_score,
                crop_stats=crop_stats,
            )
        if reacquired is not None:
            pending_reacquire = _advance_pending_reacquire(pending_reacquire, reacquired, frame_id)
            _remember_global_tracklet_confirmation(
                pending_reacquire,
                reacquired,
                min_tracklet_score=reacquire_global_min_tracklet_score,
            )
            reacquire_reason = str(pending_reacquire["reason"])
            required_confirm_frames = _required_reacquire_confirm_frames(
                pending_reacquire,
                confirm_frames=confirm_frames,
                global_confirm_frames=global_confirm_frames,
            )
            if pending_reacquire["count"] < required_confirm_frames:
                reacquire_pending_frames += 1
                decision = {"accepted": False, "guard_reason": "reacquire_pending_confirmation"}
                debug_rows.append(
                    _debug_row(
                        frame_id,
                        0,
                        reacquired,
                        selected=False,
                        guard_reason="reacquire_pending_confirmation",
                        reacquire_reason=f"pending_{reacquire_reason}",
                        reacquire_distance_px=reacquired["reacquire_distance_px"],
                    )
                )
            else:
                reacquire_mode = str(pending_reacquire["mode"])
                reacquire_small_area = bool(pending_reacquire.get("global_small_area_candidate", False))
                pending_reacquire = None
                reacquire_confirmed_frames += 1
                bbox = reacquired["bbox"]
                memory.append((frame_id, bbox))
                memory = memory[-8:]
                _remember_appearance(appearance_memory, frame_image, bbox, appearance_memory_size)
                lost_streak = 0
                accepted_candidate_frames += 1
                reacquired_frames += 1
                if reacquire_mode == "global":
                    reacquire_global_frames += 1
                    if reacquire_small_area:
                        reacquire_global_small_area_frames += 1
                        small_global_memory_probation_until_frame = _extend_memory_probation(
                            small_global_memory_probation_until_frame,
                            frame_id,
                            reacquire_global_small_memory_probation_frames,
                        )
                        small_global_memory_recover_grace_until_frame = max(
                            small_global_memory_recover_grace_until_frame,
                            frame_id + max(0, int(max_recover_frames)),
                        )
                else:
                    reacquire_memory_frames += 1
                confirmed_reason = (
                    f"confirmed_{reacquire_reason}" if confirm_frames > 1 else reacquire_reason
                )
                trajectory_rows.append(
                    _trajectory_row(
                        frame_id,
                        selected=True,
                        state="REACQUIRE",
                        selected_source=str(reacquired["row"].get("source", "")),
                        guard_reason="reacquire_confirmed" if confirm_frames > 1 else "reacquire_memory_proximity",
                        bbox=bbox,
                        score_row=reacquired,
                        reacquire_reason=confirmed_reason,
                        reacquire_distance_px=reacquired["reacquire_distance_px"],
                    )
                )
                debug_rows.append(
                    _debug_row(
                        frame_id,
                        0,
                        reacquired,
                        selected=True,
                        guard_reason="reacquire_confirmed" if confirm_frames > 1 else "reacquire_memory_proximity",
                        reacquire_reason=confirmed_reason,
                        reacquire_distance_px=reacquired["reacquire_distance_px"],
                    )
                )
                continue
        else:
            pending_reacquire = None

        fallback_needed = bool(memory)
        if fallback_needed:
            fallback_needed_frames += 1
        recover_bbox = predicted_bbox or memory_bbox
        recover_probation_active = (
            frame_id <= small_global_memory_probation_until_frame
            and frame_id > small_global_memory_recover_grace_until_frame
        )
        if (
            recover_bbox is not None
            and lost_streak < max_recover_frames
            and not recover_probation_active
        ):
            lost_streak += 1
            memory_recover_frames += 1
            memory.append((frame_id, recover_bbox))
            memory = memory[-8:]
            trajectory_rows.append(
                _trajectory_row(
                    frame_id,
                    selected=True,
                    state="RECOVER",
                    selected_source="memory_recover",
                    guard_reason=decision["guard_reason"] or "no_candidate",
                    bbox=recover_bbox,
                    score_row=best,
                )
            )
        else:
            if (
                recover_bbox is not None
                and lost_streak < max_recover_frames
                and recover_probation_active
            ):
                memory_recover_probation_rejected_frames += 1
            lost_streak += 1
            lost_frames += 1
            trajectory_rows.append(
                _trajectory_row(
                    frame_id,
                    selected=False,
                    state="LOST",
                    selected_source="none",
                    guard_reason=decision["guard_reason"] or "no_candidate",
                    bbox=None,
                    score_row=best,
                )
            )

    if frame_reader is not None:
        frame_reader.close()
    appearance_rejected_frames = int(appearance_stats["rejected"])
    appearance_soft_scored_frames = int(appearance_stats.get("soft_scored", 0))
    crop_rejected_frames = int(crop_stats["rejected"])
    crop_soft_scored_frames = int(crop_stats.get("soft_scored", 0))
    global_tracklet_rejected_frames = int(global_tracklet_stats["rejected"])

    trajectory_csv = out / "trajectory.csv"
    debug_csv = out / "selector_debug.csv"
    summary_json = out / "selector_summary.json"
    _write_csv(trajectory_csv, trajectory_rows, TRAJECTORY_FIELDS)
    _write_csv(debug_csv, debug_rows, DEBUG_FIELDS)
    summary = {
        "predictions_jsonl": str(pred_path),
        "total_frames": len(trajectory_rows),
        "candidate_frames": sum(1 for frame_id in range(max_frame_id + 1) if rows_by_frame.get(frame_id)),
        "accepted_candidate_frames": accepted_candidate_frames,
        "reacquired_frames": reacquired_frames,
        "reacquire_pending_frames": reacquire_pending_frames,
        "reacquire_confirmed_frames": reacquire_confirmed_frames,
        "reacquire_memory_frames": reacquire_memory_frames,
        "reacquire_global_frames": reacquire_global_frames,
        "reacquire_global_small_area_frames": reacquire_global_small_area_frames,
        "reacquire_global_small_min_area": reacquire_global_small_min_area,
        "reacquire_global_small_max_distance_px": reacquire_global_small_max_distance_px,
        "reacquire_global_small_memory_probation_frames": int(reacquire_global_small_memory_probation_frames),
        "reacquire_memory_probation_rejected_frames": reacquire_memory_probation_rejected_frames,
        "memory_recover_probation_rejected_frames": memory_recover_probation_rejected_frames,
        "appearance_rejected_frames": appearance_rejected_frames,
        "appearance_soft_scored_frames": appearance_soft_scored_frames,
        "appearance_memory_frames": len(appearance_memory),
        "crop_rejected_frames": crop_rejected_frames,
        "crop_soft_scored_frames": crop_soft_scored_frames,
        "crop_score_field": crop_score_field,
        "crop_weights": str(crop_weights_path) if crop_weights_path is not None else "",
        "global_tracklet_rejected_frames": global_tracklet_rejected_frames,
        "reacquire_global_delayed_confirm_frames": global_confirm_frames if global_confirm_frames > confirm_frames else 0,
        "memory_recover_frames": memory_recover_frames,
        "lost_frames": lost_frames,
        "fallback_needed_frames": fallback_needed_frames,
        "trajectory_csv": str(trajectory_csv),
        "debug_csv": str(debug_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    annotated_video = _render_video(video, trajectory_rows, out) if save_video and video else None
    if annotated_video is not None:
        summary["annotated_video"] = str(annotated_video)
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return OfflineSelectorResult(trajectory_csv, debug_csv, summary_json, annotated_video, summary)


TRAJECTORY_FIELDS = [
    "frame_id",
    "selected",
    "state",
    "selected_source",
    "guard_reason",
    "x1",
    "y1",
    "x2",
    "y2",
    "selector_score",
    "detector_score",
    "motion_consistency",
    "memory_consistency",
    "objectness",
    "final_drone_score",
    "reacquire_reason",
    "reacquire_distance_px",
    "appearance_similarity",
    "crop_drone_score",
]


DEBUG_FIELDS = [
    "frame_id",
    "rank",
    "selected",
    "guard_reason",
    "source",
    "x1",
    "y1",
    "x2",
    "y2",
    "selector_score",
    "detector_score",
    "motion_consistency",
    "memory_consistency",
    "jump_px",
    "objectness",
    "final_drone_score",
    "reacquire_reason",
    "reacquire_distance_px",
    "appearance_similarity",
    "crop_drone_score",
]


def _load_rows_by_frame(path: Path) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, list[dict[str, Any]]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_id = int(row.get("frame_id", 0))
            if _bbox_from_row(row) is None:
                continue
            rows.setdefault(frame_id, []).append(row)
    return rows


def _top_k_candidates(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    candidates = sorted(rows, key=_detector_score, reverse=True)
    if top_k > 0:
        return candidates[:top_k]
    return candidates


def _score_candidate(
    row: dict[str, Any],
    *,
    predicted_bbox: BBox | None,
    memory_bbox: BBox | None,
    detector_weight: float,
    motion_weight: float,
    memory_weight: float,
) -> dict[str, Any]:
    bbox = _bbox_from_row(row)
    if bbox is None:
        raise ValueError("candidate row is missing bbox")
    detector_score = _detector_score(row)
    if predicted_bbox is None:
        motion_consistency = 1.0
        jump_px = 0.0
    else:
        jump_px = _center_distance(bbox, predicted_bbox)
        motion_consistency = _bbox_consistency(bbox, predicted_bbox)
    memory_consistency = 1.0 if memory_bbox is None else _bbox_consistency(bbox, memory_bbox)
    total_weight = max(1e-9, detector_weight + motion_weight + memory_weight)
    selector_score = (
        detector_weight * detector_score
        + motion_weight * motion_consistency
        + memory_weight * memory_consistency
    ) / total_weight
    return {
        "row": row,
        "bbox": bbox,
        "selector_score": selector_score,
        "detector_score": detector_score,
        "motion_consistency": motion_consistency,
        "memory_consistency": memory_consistency,
        "jump_px": jump_px,
        "appearance_similarity": "",
        "appearance_rank_score": detector_score,
        "crop_drone_score": "",
    }


def _candidate_decision(
    best: dict[str, Any] | None,
    second: dict[str, Any] | None,
    *,
    has_memory: bool,
    min_accept_score: float,
    min_motion_consistency: float,
    min_memory_consistency: float,
    min_topk_margin: float,
    max_jump_px: float,
) -> dict[str, Any]:
    if best is None:
        return {"accepted": False, "guard_reason": "no_candidate"}
    if best["selector_score"] < min_accept_score:
        return {"accepted": False, "guard_reason": "low_selector_score"}
    if second is not None and min_topk_margin > 0:
        if best["selector_score"] - second["selector_score"] < min_topk_margin:
            return {"accepted": False, "guard_reason": "topk_ambiguous"}
    if has_memory and (
        best["jump_px"] > max_jump_px
        or best["motion_consistency"] < min_motion_consistency
        or best["memory_consistency"] < min_memory_consistency
    ):
        return {"accepted": False, "guard_reason": "jump_or_memory_inconsistent"}
    return {"accepted": True, "guard_reason": "accepted"}


def _find_reacquire_candidate(
    rows: list[dict[str, Any]],
    *,
    predicted_bbox: BBox | None,
    memory_bbox: BBox | None,
    detector_weight: float,
    motion_weight: float,
    memory_weight: float,
    reacquire_top_k: int,
    reacquire_max_distance_px: float,
    reacquire_min_detector_score: float,
    reacquire_min_motion_consistency: float,
    reacquire_min_memory_consistency: float,
    reacquire_min_side_ratio: float,
    reacquire_max_side_ratio: float,
    frame_image: Any,
    appearance_memory: list[list[float]],
    reacquire_min_appearance_similarity: float,
    reacquire_appearance_weight: float,
    appearance_stats: dict[str, int],
    crop_score_field: str,
    crop_scorer: Any,
    reacquire_crop_weight: float,
    reacquire_min_crop_drone_score: float,
    crop_stats: dict[str, int],
) -> dict[str, Any] | None:
    if predicted_bbox is None:
        return None
    search_rows = sorted(rows, key=_detector_score, reverse=True)
    if reacquire_top_k > 0:
        search_rows = search_rows[:reacquire_top_k]
    candidates: list[dict[str, Any]] = []
    for row in search_rows:
        if _detector_score(row) < reacquire_min_detector_score:
            continue
        item = _score_candidate(
            row,
            predicted_bbox=predicted_bbox,
            memory_bbox=memory_bbox,
            detector_weight=detector_weight,
            motion_weight=motion_weight,
            memory_weight=memory_weight,
        )
        distance = _center_distance(item["bbox"], predicted_bbox)
        if distance > reacquire_max_distance_px:
            continue
        if item["motion_consistency"] < reacquire_min_motion_consistency:
            continue
        if memory_bbox is not None and item["memory_consistency"] < reacquire_min_memory_consistency:
            continue
        if not _bbox_size_consistent(
            item["bbox"],
            predicted_bbox,
            min_side_ratio=reacquire_min_side_ratio,
            max_side_ratio=reacquire_max_side_ratio,
        ):
            continue
        if memory_bbox is not None and not _bbox_size_consistent(
            item["bbox"],
            memory_bbox,
            min_side_ratio=reacquire_min_side_ratio,
            max_side_ratio=reacquire_max_side_ratio,
        ):
            continue
        if not _apply_appearance_score(
            item,
            frame_image=frame_image,
            appearance_memory=appearance_memory,
            min_similarity=reacquire_min_appearance_similarity,
            appearance_weight=reacquire_appearance_weight,
            appearance_stats=appearance_stats,
        ):
            continue
        if not _apply_crop_score(
            item,
            frame_image=frame_image,
            crop_score_field=crop_score_field,
            crop_scorer=crop_scorer,
            min_crop_drone_score=reacquire_min_crop_drone_score,
            crop_weight=reacquire_crop_weight,
            crop_stats=crop_stats,
        ):
            continue
        item["reacquire_distance_px"] = distance
        item["reacquire_mode"] = "memory"
        item["reacquire_reason"] = "deep_candidate_near_prediction"
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item["appearance_rank_score"], item["reacquire_distance_px"]))
    return candidates[0]


def _find_global_reacquire_candidate(
    rows: list[dict[str, Any]],
    *,
    predicted_bbox: BBox | None,
    memory_bbox: BBox | None,
    detector_weight: float,
    motion_weight: float,
    memory_weight: float,
    reacquire_min_detector_score: float,
    reacquire_global_min_area: float,
    reacquire_global_small_min_area: float,
    reacquire_global_small_max_distance_px: float,
    reacquire_global_max_area: float,
    reacquire_global_require_tracklet_confirmation: bool,
    reacquire_global_reject_tracklet_rejected: bool,
    reacquire_global_min_tracklet_score: float,
    global_tracklet_stats: dict[str, int],
    frame_image: Any,
    appearance_memory: list[list[float]],
    reacquire_min_appearance_similarity: float,
    reacquire_appearance_weight: float,
    appearance_stats: dict[str, int],
    crop_score_field: str,
    crop_scorer: Any,
    reacquire_crop_weight: float,
    reacquire_min_crop_drone_score: float,
    crop_stats: dict[str, int],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in sorted(rows, key=_detector_score, reverse=True):
        if _detector_score(row) < reacquire_min_detector_score:
            continue
        item = _score_candidate(
            row,
            predicted_bbox=predicted_bbox,
            memory_bbox=memory_bbox,
            detector_weight=detector_weight,
            motion_weight=motion_weight,
            memory_weight=memory_weight,
        )
        area = _box_area(item["bbox"])
        global_small_area_candidate = False
        if area < reacquire_global_min_area:
            if reacquire_global_small_min_area <= 0 or area < reacquire_global_small_min_area:
                continue
            if not _global_tracklet_confirmation_ok(row, min_tracklet_score=reacquire_global_min_tracklet_score):
                continue
            if _sequence_gate_rejected(row):
                continue
            if reacquire_min_crop_drone_score <= 0:
                continue
            if (
                reacquire_global_small_max_distance_px > 0
                and predicted_bbox is not None
                and _center_distance(item["bbox"], predicted_bbox) > reacquire_global_small_max_distance_px
            ):
                continue
            global_small_area_candidate = True
        if reacquire_global_max_area > 0 and area > reacquire_global_max_area:
            continue
        if reacquire_global_require_tracklet_confirmation and not _apply_global_tracklet_confirmation(
            item,
            min_tracklet_score=reacquire_global_min_tracklet_score,
            global_tracklet_stats=global_tracklet_stats,
        ):
            continue
        if (
            not reacquire_global_require_tracklet_confirmation
            and reacquire_global_reject_tracklet_rejected
            and not _apply_global_tracklet_rejection_veto(
                item,
                global_tracklet_stats=global_tracklet_stats,
            )
        ):
            continue
        if not _apply_appearance_score(
            item,
            frame_image=frame_image,
            appearance_memory=appearance_memory,
            min_similarity=reacquire_min_appearance_similarity,
            appearance_weight=reacquire_appearance_weight,
            appearance_stats=appearance_stats,
        ):
            continue
        if not _apply_crop_score(
            item,
            frame_image=frame_image,
            crop_score_field=crop_score_field,
            crop_scorer=crop_scorer,
            min_crop_drone_score=reacquire_min_crop_drone_score,
            crop_weight=reacquire_crop_weight,
            crop_stats=crop_stats,
        ):
            continue
        item["reacquire_distance_px"] = (
            _center_distance(item["bbox"], predicted_bbox) if predicted_bbox is not None else 0.0
        )
        item["reacquire_mode"] = "global"
        item["global_small_area_candidate"] = global_small_area_candidate
        item["reacquire_reason"] = (
            "global_small_area_candidate_after_stale_lost"
            if global_small_area_candidate
            else "global_candidate_after_stale_lost"
        )
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item["appearance_rank_score"], item["reacquire_distance_px"]))
    return candidates[0]


def _apply_global_tracklet_rejection_veto(
    item: dict[str, Any],
    *,
    global_tracklet_stats: dict[str, int],
) -> bool:
    if not _global_tracklet_rejected(item["row"]):
        return True
    global_tracklet_stats["rejected"] = int(global_tracklet_stats.get("rejected", 0)) + 1
    return False


def _apply_global_tracklet_confirmation(
    item: dict[str, Any],
    *,
    min_tracklet_score: float,
    global_tracklet_stats: dict[str, int],
) -> bool:
    if _global_tracklet_confirmation_ok(item["row"], min_tracklet_score=min_tracklet_score):
        return True
    global_tracklet_stats["rejected"] = int(global_tracklet_stats.get("rejected", 0)) + 1
    return False


def _global_tracklet_confirmation_ok(row: dict[str, Any], *, min_tracklet_score: float) -> bool:
    is_drone = _to_optional_bool(row.get("tracklet_is_drone"))
    if is_drone is True:
        return True
    score = _parse_probability_value(row.get("tracklet_classifier_prob"))
    if score is not None and score >= min_tracklet_score:
        return True
    diagnostic_cause = str(row.get("diagnostic_cause") or "").lower()
    return "tracklet_confirmed" in diagnostic_cause and "tracklet_rejected" not in diagnostic_cause


def _global_tracklet_rejected(row: dict[str, Any]) -> bool:
    diagnostic_cause = str(row.get("diagnostic_cause") or "").lower()
    if "tracklet_rejected" in diagnostic_cause:
        return True
    filter_applied = _to_optional_bool(row.get("tracklet_filter_applied"))
    is_drone = _to_optional_bool(row.get("tracklet_is_drone"))
    return filter_applied is True and is_drone is False


def _sequence_gate_rejected(row: dict[str, Any]) -> bool:
    diagnostic_cause = str(row.get("diagnostic_cause") or "").lower()
    if "sequence_gate_rejected" in diagnostic_cause:
        return True
    confirmed = _to_optional_bool(row.get("sequence_gate_confirmed"))
    if confirmed is False:
        reason = str(row.get("sequence_gate_reason") or "").lower()
        return reason not in {"", "unlinked"}
    return False


def _apply_appearance_score(
    item: dict[str, Any],
    *,
    frame_image: Any,
    appearance_memory: list[list[float]],
    min_similarity: float,
    appearance_weight: float,
    appearance_stats: dict[str, int],
) -> bool:
    item["appearance_rank_score"] = item["detector_score"]
    if min_similarity <= 0 and appearance_weight <= 0:
        return True
    similarity = _appearance_similarity(frame_image, item["bbox"], appearance_memory)
    item["appearance_similarity"] = "" if similarity is None else similarity
    if similarity is None or similarity < min_similarity:
        appearance_stats["rejected"] = int(appearance_stats.get("rejected", 0)) + 1
        return False
    if appearance_weight > 0:
        appearance_stats["soft_scored"] = int(appearance_stats.get("soft_scored", 0)) + 1
        item["appearance_rank_score"] = item["detector_score"] + appearance_weight * (similarity or 0.0)
    return True


def _apply_crop_score(
    item: dict[str, Any],
    *,
    frame_image: Any,
    crop_score_field: str,
    crop_scorer: Any,
    min_crop_drone_score: float,
    crop_weight: float,
    crop_stats: dict[str, int],
) -> bool:
    if not crop_score_field and crop_scorer is None and min_crop_drone_score <= 0 and crop_weight <= 0:
        return True
    score = _crop_score_from_row(item["row"], crop_score_field)
    if score is None and crop_scorer is not None:
        score = crop_scorer.score(frame_image, item["bbox"])
    if score is None:
        if min_crop_drone_score > 0:
            crop_stats["rejected"] = int(crop_stats.get("rejected", 0)) + 1
            return False
        return True
    item["crop_drone_score"] = score
    if min_crop_drone_score > 0 and score < min_crop_drone_score:
        crop_stats["rejected"] = int(crop_stats.get("rejected", 0)) + 1
        return False
    if crop_weight > 0:
        crop_stats["soft_scored"] = int(crop_stats.get("soft_scored", 0)) + 1
        item["appearance_rank_score"] = item.get("appearance_rank_score", item["detector_score"]) + crop_weight * score
    return True


def _crop_score_from_row(row: dict[str, Any], field: str) -> float | None:
    if not field or field not in row:
        return None
    return _parse_probability_value(row.get(field))


def _parse_probability_value(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return _optional_probability_float(text)
    if isinstance(value, dict):
        for key in ("drone", "crop_drone_score", "drone_score", "p_drone", "prob_drone"):
            if key in value:
                return _parse_probability_value(value[key])
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _parse_probability_value(value[0])
    return _optional_probability_float(value)


def _optional_probability_float(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return min(1.0, max(0.0, score))


def _remember_appearance(
    appearance_memory: list[list[float]],
    frame_image: Any,
    bbox: BBox,
    max_features: int,
) -> None:
    feature = _appearance_feature(frame_image, bbox)
    if feature is None:
        return
    appearance_memory.append(feature)
    del appearance_memory[: max(0, len(appearance_memory) - max(1, max_features))]


def _appearance_similarity(
    frame_image: Any,
    bbox: BBox,
    appearance_memory: list[list[float]],
) -> float | None:
    feature = _appearance_feature(frame_image, bbox)
    if feature is None or not appearance_memory:
        return None
    return max(_cosine_similarity(feature, memory_feature) for memory_feature in appearance_memory)


def _appearance_feature(frame_image: Any, bbox: BBox) -> list[float] | None:
    if frame_image is None:
        return None
    import cv2
    import numpy as np

    height, width = frame_image.shape[:2]
    x1 = max(0, min(width, int(math.floor(bbox[0]))))
    y1 = max(0, min(height, int(math.floor(bbox[1]))))
    x2 = max(0, min(width, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height, int(math.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (16, 16), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    flat = hist.reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm <= 1e-9:
        return None
    return (flat / norm).tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _advance_pending_reacquire(
    pending: dict[str, Any] | None,
    item: dict[str, Any],
    frame_id: int,
) -> dict[str, Any]:
    item_mode = str(item.get("reacquire_mode", pending["mode"] if pending is not None else "memory"))
    item_reason = str(
        item.get("reacquire_reason", pending["reason"] if pending is not None else "deep_candidate_near_prediction")
    )
    if (
        pending is not None
        and pending["mode"] == item_mode
        and int(pending["last_frame_id"]) == frame_id - 1
        and _pending_reacquire_matches(pending["bbox"], item["bbox"])
    ):
        count = int(pending["count"]) + 1
        first_frame_id = int(pending["first_frame_id"])
        global_small_area_candidate = bool(pending.get("global_small_area_candidate", False)) and bool(
            item.get("global_small_area_candidate", False)
        )
    else:
        count = 1
        first_frame_id = frame_id
        global_small_area_candidate = bool(item.get("global_small_area_candidate", False))
    return {
        "bbox": item["bbox"],
        "count": count,
        "first_frame_id": first_frame_id,
        "last_frame_id": frame_id,
        "mode": item_mode,
        "reason": item_reason,
        "global_small_area_candidate": global_small_area_candidate,
    }


def _required_reacquire_confirm_frames(
    pending: dict[str, Any],
    *,
    confirm_frames: int,
    global_confirm_frames: int,
) -> int:
    if str(pending.get("mode", "")) == "global":
        if bool(pending.get("global_tracklet_confirmed_seen", False)):
            return confirm_frames
        return global_confirm_frames
    return confirm_frames


def _remember_global_tracklet_confirmation(
    pending: dict[str, Any],
    item: dict[str, Any],
    *,
    min_tracklet_score: float,
) -> None:
    if str(pending.get("mode", "")) != "global":
        return
    if _global_tracklet_confirmation_ok(item["row"], min_tracklet_score=min_tracklet_score):
        pending["global_tracklet_confirmed_seen"] = True


def _extend_memory_probation(current_until: int, frame_id: int, frames: int) -> int:
    if frames <= 0:
        return current_until
    return max(current_until, frame_id + int(frames))


def _pending_reacquire_matches(a: BBox, b: BBox) -> bool:
    max_step = max(24.0, 3.0 * _box_side(a), 3.0 * _box_side(b))
    return _center_distance(a, b) <= max_step or _bbox_consistency(a, b) >= 0.10


def _trajectory_row(
    frame_id: int,
    *,
    selected: bool,
    state: str,
    selected_source: str,
    guard_reason: str,
    bbox: BBox | None,
    score_row: dict[str, Any] | None,
    reacquire_reason: str = "",
    reacquire_distance_px: float | str = "",
) -> dict[str, Any]:
    row = score_row["row"] if score_row else {}
    x1, y1, x2, y2 = bbox if bbox is not None else ("", "", "", "")
    return {
        "frame_id": frame_id,
        "selected": 1 if selected else 0,
        "state": state,
        "selected_source": selected_source,
        "guard_reason": guard_reason,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "selector_score": "" if score_row is None else score_row["selector_score"],
        "detector_score": "" if score_row is None else score_row["detector_score"],
        "motion_consistency": "" if score_row is None else score_row["motion_consistency"],
        "memory_consistency": "" if score_row is None else score_row["memory_consistency"],
        "objectness": row.get("objectness", ""),
        "final_drone_score": row.get("final_drone_score", ""),
        "reacquire_reason": reacquire_reason,
        "reacquire_distance_px": reacquire_distance_px,
        "appearance_similarity": "" if score_row is None else score_row.get("appearance_similarity", ""),
        "crop_drone_score": "" if score_row is None else score_row.get("crop_drone_score", ""),
    }


def _debug_row(
    frame_id: int,
    rank: int,
    item: dict[str, Any],
    *,
    selected: bool,
    guard_reason: str,
    reacquire_reason: str = "",
    reacquire_distance_px: float | str = "",
) -> dict[str, Any]:
    row = item["row"]
    x1, y1, x2, y2 = item["bbox"]
    return {
        "frame_id": frame_id,
        "rank": rank,
        "selected": 1 if selected else 0,
        "guard_reason": guard_reason,
        "source": row.get("source", ""),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "selector_score": item["selector_score"],
        "detector_score": item["detector_score"],
        "motion_consistency": item["motion_consistency"],
        "memory_consistency": item["memory_consistency"],
        "jump_px": item["jump_px"],
        "objectness": row.get("objectness", ""),
        "final_drone_score": row.get("final_drone_score", ""),
        "reacquire_reason": reacquire_reason,
        "reacquire_distance_px": reacquire_distance_px,
        "appearance_similarity": item.get("appearance_similarity", ""),
        "crop_drone_score": item.get("crop_drone_score", ""),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _detector_score(row: dict[str, Any]) -> float:
    final_score = _to_float(row.get("final_drone_score"), 0.0)
    objectness = _to_float(row.get("objectness"), 0.0)
    if final_score > 0:
        return 0.65 * final_score + 0.35 * objectness
    return objectness


def _bbox_from_row(row: dict[str, Any]) -> BBox | None:
    raw = row.get("bbox", row.get("bbox_xyxy"))
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    vals = list(raw)
    if len(vals) != 4:
        return None
    return tuple(float(v) for v in vals)  # type: ignore[return-value]


def _predict_bbox(memory: list[tuple[int, BBox]], frame_id: int) -> BBox | None:
    if not memory:
        return None
    last_frame, last = memory[-1]
    if len(memory) < 2:
        return last
    prev_frame, prev = memory[-2]
    dt = max(1, last_frame - prev_frame)
    horizon = max(1, frame_id - last_frame)
    lc = _center(last)
    pc = _center(prev)
    vx = (lc[0] - pc[0]) / dt
    vy = (lc[1] - pc[1]) / dt
    dx = vx * horizon
    dy = vy * horizon
    return (last[0] + dx, last[1] + dy, last[2] + dx, last[3] + dy)


def _memory_bbox(memory: list[tuple[int, BBox]]) -> BBox | None:
    if not memory:
        return None
    recent = [bbox for _, bbox in memory[-5:]]
    return tuple(_median([bbox[i] for bbox in recent]) for i in range(4))  # type: ignore[return-value]


def _median(values: list[float]) -> float:
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _bbox_consistency(a: BBox, b: BBox) -> float:
    dist = _center_distance(a, b)
    side = max(_box_side(a), _box_side(b), 1.0)
    center_score = math.exp(-dist / max(12.0, 2.5 * side))
    size_score = math.exp(-abs(math.log(_box_side(a) / max(_box_side(b), 1e-6))))
    return max(_iou(a, b), center_score * size_score)


def _bbox_size_consistent(
    candidate: BBox,
    reference: BBox,
    *,
    min_side_ratio: float,
    max_side_ratio: float,
) -> bool:
    ref_side = _box_side(reference)
    side_ratio = _box_side(candidate) / max(ref_side, 1e-6)
    area_ratio = _box_area(candidate) / max(_box_area(reference), 1e-6)
    min_area_ratio = min_side_ratio * min_side_ratio
    max_area_ratio = max_side_ratio * max_side_ratio
    return min_side_ratio <= side_ratio <= max_side_ratio and min_area_ratio <= area_ratio <= max_area_ratio


def _center_distance(a: BBox, b: BBox) -> float:
    ac = _center(a)
    bc = _center(b)
    return float(math.hypot(ac[0] - bc[0], ac[1] - bc[1]))


def _center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def _box_side(bbox: BBox) -> float:
    return max(1.0, float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))


def _box_area(bbox: BBox) -> float:
    return max(1.0, float(bbox[2] - bbox[0])) * max(1.0, float(bbox[3] - bbox[1]))


def _iou(a: BBox, b: BBox) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        if float(value) == 0.0:
            return False
        if float(value) == 1.0:
            return True
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
    return None


class _CropRecognizerScorer:
    def __init__(self, weights: str | Path, *, image_size: int) -> None:
        import torch

        from qstr_dronedet.recognition.crop_recognizer import CropRecognizer
        from qstr_dronedet.types import CLASSES

        self.torch = torch
        self.image_size = int(image_size)
        self.drone_index = CLASSES.index("drone")
        self.background_index = CLASSES.index("background")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(weights), map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        self.target_mode = str(ckpt.get("target_mode", "multiclass")) if isinstance(ckpt, dict) else "multiclass"
        self.model = CropRecognizer()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def score(self, frame_image: Any, bbox: BBox) -> float | None:
        crop = _crop_frame(frame_image, bbox)
        if crop is None:
            return None
        import cv2
        import numpy as np

        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            if logits.shape[1] == 2:
                prob = self.torch.softmax(logits, dim=1)[0, 0]
            elif self.target_mode == "drone_binary" and logits.shape[1] > self.background_index:
                prob = self.torch.softmax(logits[:, [self.drone_index, self.background_index]], dim=1)[0, 0]
            else:
                prob = self.torch.softmax(logits, dim=1)[0, self.drone_index]
        return float(prob.item())


def _crop_frame(frame_image: Any, bbox: BBox) -> Any:
    if frame_image is None:
        return None
    height, width = frame_image.shape[:2]
    x1 = max(0, min(width, int(math.floor(bbox[0]))))
    y1 = max(0, min(height, int(math.floor(bbox[1]))))
    x2 = max(0, min(width, int(math.ceil(bbox[2]))))
    y2 = max(0, min(height, int(math.ceil(bbox[3]))))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


class _SequentialVideoFrameReader:
    def __init__(self, video: str | Path) -> None:
        import cv2

        self.video_path = Path(video)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open video: {self.video_path}")
        self.next_frame_id = 0

    def read(self, frame_id: int) -> Any:
        frame = None
        while self.next_frame_id <= frame_id:
            ok, current = self.cap.read()
            if not ok:
                return None
            if self.next_frame_id == frame_id:
                frame = current
            self.next_frame_id += 1
        return frame

    def close(self) -> None:
        self.cap.release()


def _render_video(video: str | Path | None, trajectory_rows: list[dict[str, Any]], out_dir: Path) -> Path | None:
    if video is None:
        return None
    import cv2

    video_path = Path(video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / "selector_annotated.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    by_frame = {int(row["frame_id"]): row for row in trajectory_rows}
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = by_frame.get(frame_id)
        if row and str(row.get("selected")) == "1" and row.get("x1") != "":
            x1, y1, x2, y2 = [int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2")]
            color = (0, 255, 0) if row.get("state") == "TRACK" else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{row.get('state')} {row.get('selected_source')} {row.get('guard_reason')}"
            cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        writer.write(frame)
        frame_id += 1
    cap.release()
    writer.release()
    return out_path
