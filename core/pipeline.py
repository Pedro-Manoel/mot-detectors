from __future__ import annotations

import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import Config, RunSpec
from . import detect, diagnostics, evaluate, metrics, track
from .runner import write_csv_row_atomic, run_dir_for, is_done, portable_path


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _warmup(model, frames, cfg: Config, use_nms: bool) -> None:
    n = 3 if cfg.device != "cpu" else 1
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "end2end"):
        inner.end2end = (not use_nms)
    for i in range(min(n, len(frames))):
        wf = cv2.imread(str(frames[i]))
        if wf is not None:
            detect.predict_frame(model, wf, cfg.conf, use_nms, cfg.half, cfg.imgsz)
    if detect._HAS_CUDA:
        import torch
        torch.cuda.synchronize()


def _combined_pass(model, frames, cfg: Config, use_nms: bool, tracker_name, frame_rate,
                   reid_weights):
    # The paper's protocol: one loop reads, detects and tracks each frame, once per tracker, so
    # the work a tracker does between frames is part of what the next detection runs after.
    tracker = track.load_tracker(tracker_name, reid_weights or None, cfg.device, frame_rate)
    boxes, imread, det, pre, inf, post, track_ms, rows = [], [], [], [], [], [], [], []
    fwd = fwt = 0
    for f, img_path in enumerate(frames, start=1):
        t0 = time.perf_counter()
        frame = cv2.imread(str(img_path))
        read_ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            boxes.append(np.empty((0, 6), dtype=np.float32))
            continue
        detect.cuda_sync()
        td = time.perf_counter()
        results = detect.predict_frame(model, frame, cfg.conf, use_nms, cfg.half, cfg.imgsz)
        detect.cuda_sync()
        det.append((time.perf_counter() - td) * 1000.0)
        imread.append(read_ms)
        sp = detect.extract_ultralytics_speed(results)
        if sp["ultra_pre_ms"] is not None:
            pre.append(float(sp["ultra_pre_ms"]))
        if sp["ultra_inf_ms"] is not None:
            inf.append(float(sp["ultra_inf_ms"]))
        if sp["ultra_post_ms"] is not None:
            post.append(float(sp["ultra_post_ms"]))
        d = detect.parse_ultralytics_results(results, cfg.conf)
        boxes.append(d)
        if len(d) > 0:
            fwd += 1
        detect.cuda_sync()
        tt = time.perf_counter()
        out = tracker.update(d, frame)
        detect.cuda_sync()
        track_ms.append((time.perf_counter() - tt) * 1000.0)
        r = track.parse_track_outputs(f, out)
        if r:
            fwt += 1
        rows.extend(r)
    return boxes, imread, det, pre, inf, post, fwd, track_ms, rows, fwt


def _evaluate(dataset, sequence, tracker, mot_rows, cfg: Config, mot_dir: Path):
    prepared_root = Path(cfg.prepared_root).resolve()
    trackeval_root = Path(cfg.trackeval_root).resolve()
    te_python = evaluate.resolve_python(cfg.trackeval_python)

    mot_dir = Path(mot_dir).resolve()
    mot_dir.mkdir(parents=True, exist_ok=True)
    mot_path = mot_dir / "mot_results.txt"
    track.write_mot_txt(mot_rows, mot_path)
    # TrackEval's folder tree goes to a short temporary path: inside a deep run folder its file
    # paths can exceed the 260-character limit of Windows.
    ws = Path(tempfile.mkdtemp(prefix="trackeval_"))
    try:
        workspace = evaluate.stage_trackeval_workspace(
            prepared_root, dataset, sequence, tracker, mot_path, ws)
        raw = evaluate.call_trackeval(
            trackeval_root, te_python, dataset, tracker, workspace, cfg.do_preproc)
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    quality = {k: raw.get(k) for k in evaluate.TRACKEVAL_WANTED_METRICS}
    if not cfg.keep_tracks and mot_path.exists():
        mot_path.unlink()
    return quality, mot_path


def _build_row(spec: RunSpec, cfg: Config, density, frame_rate, num_frames,
               frames_with_dets, frames_with_tracks, num_track_rows,
               head_active, head_end2end, timing: dict, quality: dict,
               mot_path: Path, run_dir: Path) -> dict:
    return {
        "run_id": spec.run_id, "timestamp": _now_ts(),
        "dataset": spec.dataset, "sequence": spec.sequence, "density": density,
        "model": spec.model, "model_family": spec.family,
        "model_scale": spec.scale, "mode": spec.mode,
        "tracker": spec.tracker, "device": cfg.device,
        "conf": cfg.conf, "half": cfg.half, "imgsz": cfg.imgsz,
        "reid_weights": spec.reid_weights or "",
        "with_reid": bool(spec.reid_weights and spec.tracker == "botsort"),
        "frame_rate": frame_rate,
        "num_frames": num_frames,
        "num_frames_with_detections": frames_with_dets,
        "num_frames_with_tracks": frames_with_tracks,
        "num_track_rows": num_track_rows,
        "avg_det_ms": timing["avg_det_ms"], "std_det_ms": timing["std_det_ms"], "fps_det": timing["fps_det"],
        "avg_track_ms": timing["avg_track_ms"], "std_track_ms": timing["std_track_ms"], "fps_track": timing["fps_track"],
        "avg_total_ms": timing["avg_total_ms"], "std_total_ms": timing["std_total_ms"], "fps_total": timing["fps_total"],
        "avg_preprocess_ms": timing["avg_preprocess_ms"], "std_preprocess_ms": timing["std_preprocess_ms"], "fps_preprocess": timing["fps_preprocess"],
        "avg_inference_ms": timing["avg_inference_ms"], "std_inference_ms": timing["std_inference_ms"], "fps_inference": timing["fps_inference"],
        "avg_postprocess_ms": timing["avg_postprocess_ms"], "std_postprocess_ms": timing["std_postprocess_ms"], "fps_postprocess": timing["fps_postprocess"],
        "head_active_final": head_active,
        "head_end2end_final": head_end2end,
        "frame1_det_ms": timing["frame1_det_ms"],
        "frame1_total_ms": timing["frame1_total_ms"],
        "frame1_postprocess_ms": timing["frame1_postprocess_ms"],
        **quality,
        "mot_results_file": portable_path(mot_path), "run_dir": portable_path(run_dir),
    }


def _detect_pass(model, frames, cfg: Config, use_nms: bool):
    boxes, imread, det, pre, inf, post = [], [], [], [], [], []
    fwd = 0
    for img_path in frames:
        t0 = time.perf_counter()
        frame = cv2.imread(str(img_path))
        read_ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            boxes.append(np.empty((0, 6), dtype=np.float32))
            continue
        detect.cuda_sync()
        td = time.perf_counter()
        results = detect.predict_frame(model, frame, cfg.conf, use_nms, cfg.half, cfg.imgsz)
        detect.cuda_sync()
        det.append((time.perf_counter() - td) * 1000.0)
        imread.append(read_ms)
        sp = detect.extract_ultralytics_speed(results)
        if sp["ultra_pre_ms"] is not None:
            pre.append(float(sp["ultra_pre_ms"]))
        if sp["ultra_inf_ms"] is not None:
            inf.append(float(sp["ultra_inf_ms"]))
        if sp["ultra_post_ms"] is not None:
            post.append(float(sp["ultra_post_ms"]))
        d = detect.parse_ultralytics_results(results, cfg.conf)
        boxes.append(d)
        if len(d) > 0:
            fwd += 1
    return boxes, imread, det, pre, inf, post, fwd


def _track_pass(tracker_name, boxes, frames, cfg: Config, frame_rate, reid_weights):
    tracker = track.load_tracker(tracker_name, reid_weights or None, cfg.device, frame_rate)
    track_ms, rows, fwt = [], [], 0
    for f, img_path in enumerate(frames, start=1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        d = boxes[f - 1]
        detect.cuda_sync()
        t0 = time.perf_counter()
        out = tracker.update(d, frame)
        detect.cuda_sync()
        track_ms.append((time.perf_counter() - t0) * 1000.0)
        r = track.parse_track_outputs(f, out)
        if r:
            fwt += 1
        rows.extend(r)
    return track_ms, rows, fwt


def process_group(key, specs, model, frames, meta, cfg: Config,
                  results_root: Path, slug: str, force: bool) -> list[Path]:
    dataset, sequence, family, scale, mode = key
    use_nms = (mode == "nms")
    reps = cfg.runs
    frame_rate = meta.get("frame_rate", 25)
    density = meta.get("density_avg") or meta.get("density_avg_computed") or 0.0

    by_tracker: dict = {}
    for s in specs:
        by_tracker.setdefault(s.tracker, []).append(s)
    spec_by = {(s.tracker, s.run_id): s for s in specs}
    active = [t for t, ss in by_tracker.items()
              if force or any(not is_done(run_dir_for(results_root, s, slug)) for s in ss)]
    if not active:
        return []
    pending_run_ids = sorted({
        run_id for t in active for run_id in range(1, reps + 1)
        if (t, run_id) in spec_by
        and (force or not is_done(run_dir_for(results_root, spec_by[(t, run_id)], slug)))})
    if not pending_run_ids:
        return []

    _warmup(model, frames, cfg, use_nms)
    head_active = head_end2end = None
    written: list[Path] = []
    for run_id in pending_run_ids:
        todo = [spec_by[(t, run_id)] for t in active if (t, run_id) in spec_by
                and (force or not is_done(run_dir_for(results_root, spec_by[(t, run_id)], slug)))]
        if cfg.shared_detection:
            boxes, imr, dt, pr, nf, po, fwd = _detect_pass(model, frames, cfg, use_nms)
        for i, s in enumerate(todo):
            if cfg.shared_detection:
                track_ms, rows, fwt = _track_pass(
                    s.tracker, boxes, frames, cfg, frame_rate, s.reid_weights)
            else:
                boxes, imr, dt, pr, nf, po, fwd, track_ms, rows, fwt = _combined_pass(
                    model, frames, cfg, use_nms, s.tracker, frame_rate, s.reid_weights)
            if head_active is None:
                head_active = detect.describe_active_head(model)
                head_end2end = detect.get_active_end2end_flag(model)
            if cfg.save_detections and run_id == pending_run_ids[0] and i == 0:
                diagnostics.save_detections(
                    results_root / dataset / sequence / "_detections"
                    / f"{family}_{scale}_{mode}.npz", boxes)
            run_dir = run_dir_for(results_root, s, slug)
            quality, mot_path = _evaluate(dataset, sequence, s.tracker, rows, cfg, run_dir)
            timing = metrics.run_timing_fields(imr, dt, track_ms, pr, nf, po)
            row = _build_row(s, cfg, density, frame_rate, len(frames),
                             fwd, fwt, len(rows), head_active, head_end2end,
                             timing, quality, mot_path, run_dir)
            written.append(write_csv_row_atomic(run_dir / "metrics.csv", row))
    return written
