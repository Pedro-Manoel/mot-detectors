from core import pipeline
from core.config import Config, DetectorSpec, expand_grid
from core.runner import run_dir_for, write_csv_row_atomic


def _cfg(tmp_path) -> Config:
    return Config(
        name="t", device="cpu", runs=2, conf=0.25, imgsz=640, half=False,
        trackeval_root="TrackEval", trackeval_python="python", do_preproc=False,
        keep_tracks=False, reid_weights="osnet.pt",
        detectors=[DetectorSpec("yolo11", ["n"], ["nms"])], trackers=["bytetrack", "botsort"],
        sequences={"MOT17": ["MOT17-05"]}, output_root=str(tmp_path),
        dataset_root="datasets", prepared_root="prepared",
    )


def _passes(tmp_path, monkeypatch, done=()):
    calls = []
    monkeypatch.setattr(pipeline, "_warmup", lambda *a: None)
    monkeypatch.setattr(pipeline, "_detect_and_track",
                        lambda model, frames, cfg, use_nms, tracker, *a:
                        calls.append(tracker) or ([], {}, 0, 0))
    monkeypatch.setattr(pipeline, "_evaluate", lambda *a: ({}, tmp_path / "mot_results.txt"))
    monkeypatch.setattr(pipeline.metrics, "run_timing_fields", lambda **times: {})
    monkeypatch.setattr(pipeline, "_build_row", lambda *a: {"HOTA": 1.0})
    monkeypatch.setattr(pipeline.detect, "describe_active_head", lambda model: "head")
    monkeypatch.setattr(pipeline.detect, "get_active_end2end_flag", lambda model: False)
    cfg = _cfg(tmp_path)
    for s in expand_grid(cfg):
        if (s.tracker, s.run_id) in done:
            write_csv_row_atomic(run_dir_for(tmp_path / "t", s, "cpu") / "metrics.csv", {"HOTA": 1.0})
    written = pipeline.process_group(("MOT17", "MOT17-05", "yolo11", "n", "nms"), expand_grid(cfg),
                                     None, [], {}, cfg, tmp_path / "t", "cpu", False)
    return calls, len(written)


def test_each_repetition_runs_the_loop_once_per_tracker(tmp_path, monkeypatch):
    calls, n = _passes(tmp_path, monkeypatch)
    assert calls == ["bytetrack", "botsort"] * 2
    assert n == 4


def test_resume_runs_only_the_missing_repetitions(tmp_path, monkeypatch):
    calls, n = _passes(tmp_path, monkeypatch, done={("bytetrack", 1)})
    assert calls == ["botsort", "bytetrack", "botsort"]
    assert n == 3
