from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import cv2

from . import detect, evaluate, metrics, track
from .config import Config, RunSpec
from .runner import is_done, portable_path, run_dir_for, timestamp, write_csv_row_atomic


def _warmup(model, frames, cfg: Config, use_nms: bool) -> None:
    for img_path in frames[:3 if cfg.device != "cpu" else 1]:
        image = cv2.imread(str(img_path))
        if image is not None:
            detect.predict_frame(model, image, cfg.conf, use_nms, cfg.half, cfg.imgsz)
    detect.cuda_sync()


def _detect_and_track(model, frames, cfg: Config, use_nms: bool, tracker_name, frame_rate,
                      reid_weights):
    tracker = track.load_tracker(tracker_name, reid_weights or None, cfg.device, frame_rate)
    times = {k: [] for k in ("imread", "det", "track", "pre", "inf", "post")}
    rows, frames_with_dets, frames_with_tracks = [], 0, 0
    for f, img_path in enumerate(frames, start=1):
        t0 = time.perf_counter()
        frame = cv2.imread(str(img_path))
        read_ms = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            continue
        detect.cuda_sync()
        td = time.perf_counter()
        results = detect.predict_frame(model, frame, cfg.conf, use_nms, cfg.half, cfg.imgsz)
        detect.cuda_sync()
        times["det"].append((time.perf_counter() - td) * 1000.0)
        times["imread"].append(read_ms)
        for key, ms in detect.extract_ultralytics_speed(results).items():
            if ms is not None:
                times[key].append(float(ms))
        boxes = detect.parse_ultralytics_results(results, cfg.conf)
        frames_with_dets += len(boxes) > 0
        detect.cuda_sync()
        tt = time.perf_counter()
        out = tracker.update(boxes, frame)
        detect.cuda_sync()
        times["track"].append((time.perf_counter() - tt) * 1000.0)
        new_rows = track.parse_track_outputs(f, out)
        frames_with_tracks += bool(new_rows)
        rows += new_rows
    return rows, times, frames_with_dets, frames_with_tracks


def _evaluate(dataset, sequence, tracker, mot_rows, cfg: Config, run_dir: Path):
    mot_path = Path(run_dir).resolve() / "mot_results.txt"
    track.write_mot_txt(mot_rows, mot_path)
    ws =Path(tempfile.mkdtemp(prefix="trackeval_"))
    try:
        workspace = evaluate.stage_trackeval_workspace(
            Path(cfg.prepared_root).resolve(), dataset, sequence, tracker, mot_path, ws)
        raw = evaluate.call_trackeval(
            Path(cfg.trackeval_root).resolve(), evaluate.resolve_python(cfg.trackeval_python),
            dataset, tracker, workspace, cfg.do_preproc)
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    if not cfg.keep_tracks:
        mot_path.unlink()
    return {k: raw.get(k) for k in evaluate.TRACKEVAL_WANTED_METRICS}, mot_path


def _build_row(spec: RunSpec, cfg: Config, density, frame_rate, num_frames,
               frames_with_dets, frames_with_tracks, num_track_rows,
               head_active, head_end2end, timing: dict, quality: dict,
               mot_path: Path, run_dir: Path) -> dict:
    return {
        "run_id": spec.run_id, "timestamp": timestamp(),
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


def process_group(key, specs, model, frames, meta, cfg: Config,
                  results_root: Path, slug: str, force: bool) -> list[Path]:
    dataset, sequence, _, _, mode = key
    pending = [s for s in specs if force or not is_done(run_dir_for(results_root, s, slug))]
    if not pending:
        return []
    use_nms = mode == "nms"
    frame_rate = meta.get("frame_rate", 25)
    density = meta.get("density_avg") or meta.get("density_avg_computed") or 0.0

    _warmup(model, frames, cfg, use_nms)
    head_active = detect.describe_active_head(model)
    head_end2end = detect.get_active_end2end_flag(model)
    written: list[Path] = []
    for s in sorted(pending, key=lambda s: s.run_id):
        rows, times, frames_with_dets, frames_with_tracks = _detect_and_track(
            model, frames, cfg, use_nms, s.tracker, frame_rate, s.reid_weights)
        run_dir = run_dir_for(results_root, s, slug)
        quality, mot_path = _evaluate(dataset, sequence, s.tracker, rows, cfg, run_dir)
        row = _build_row(s, cfg, density, frame_rate, len(frames), frames_with_dets,
                         frames_with_tracks, len(rows), head_active, head_end2end,
                         metrics.run_timing_fields(**times), quality, mot_path, run_dir)
        written.append(write_csv_row_atomic(run_dir / "metrics.csv", row))
    return written
