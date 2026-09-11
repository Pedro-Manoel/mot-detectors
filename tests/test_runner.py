from pathlib import Path

from core.config import Config, DetectorSpec, expand_grid
from core.runner import (
    run_dir_name, results_root_for, run_dir_for, is_done, partition,
    append_record, read_ledger, atomic_write_text, write_csv_row_atomic, run_batch,
    group_specs, ModelCache,
)
from core.config import load_config, expand_grid as _expand

PAPER = Path(__file__).resolve().parent.parent / "experiments" / "paper_main.yml"


def tiny_cfg(tmp_path: Path, runs: int = 2) -> Config:
    return Config(
        name="t", device="0", runs=runs, conf=0.25, imgsz=640, half=False,
        trackeval_root="TrackEval", trackeval_python="py", do_preproc=False,
        keep_tracks=True, reid_weights="osnet.pt",
        detectors=[DetectorSpec("yolo11", ["n"], ["nms"])],
        trackers=["bytetrack"],
        sequences={"MOT17": ["MOT17-02"]},
        output_root=str(tmp_path), dataset_root="datasets", prepared_root="prepared",
    )


def _make_done(results_root: Path, spec, dev_slug="0") -> Path:
    rd = run_dir_for(results_root, spec, dev_slug)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "metrics.csv").write_text("HOTA\n1.0\n", encoding="utf-8")
    return rd


def _fake_model(weight, mode, cfg):
    return ("model", weight, mode)


def _fake_frames(cfg, dataset, sequence, max_frames):
    return ([], {"frame_rate": 14})


def _writer_process(key, specs, model, frames, meta, cfg, root, slug, force):
    written = []
    for s in specs:
        rd = run_dir_for(root, s, slug)
        if not force and is_done(rd):
            continue
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "metrics.csv").write_text("ok", encoding="utf-8")
        written.append(rd / "metrics.csv")
    return written


def test_run_dir_name_matches_legacy(tmp_path):
    cfg = tiny_cfg(tmp_path)
    spec = expand_grid(cfg)[0]
    assert run_dir_name(spec, "0") == "yolo11_n_nms_bytetrack_0_run01"


def test_results_root_for_canonical_and_quicktest(tmp_path):
    cfg = tiny_cfg(tmp_path)
    assert results_root_for(cfg, quicktest=False) == tmp_path / "t"
    assert results_root_for(cfg, quicktest=True) == tmp_path / "t_quicktest"


def test_is_done_reflects_metrics_csv(tmp_path):
    cfg = tiny_cfg(tmp_path)
    root = results_root_for(cfg)
    spec = expand_grid(cfg)[0]
    rd = run_dir_for(root, spec, "0")
    assert is_done(rd) is False
    rd.mkdir(parents=True)
    (rd / "metrics.csv").write_text("x", encoding="utf-8")
    assert is_done(rd) is True


def test_partition_splits_done_and_pending(tmp_path):
    cfg = tiny_cfg(tmp_path, runs=2)
    root = results_root_for(cfg)
    grid = expand_grid(cfg)
    _make_done(root, grid[0])
    done, pending = partition(grid, root, "0")
    assert [s.run_id for s in done] == [1]
    assert [s.run_id for s in pending] == [2]


def test_ledger_append_and_read_roundtrip(tmp_path):
    ledger = tmp_path / "_runs.jsonl"
    append_record(ledger, {"key": "a", "status": "done"})
    append_record(ledger, {"key": "b", "status": "failed", "error": "boom"})
    recs = read_ledger(ledger)
    assert [r["key"] for r in recs] == ["a", "b"]
    assert recs[1]["status"] == "failed"


def test_atomic_write_leaves_no_tmp_and_writes_content(tmp_path):
    target = tmp_path / "metrics.csv"
    atomic_write_text(target, "header\nrow\n")
    assert target.read_text(encoding="utf-8") == "header\nrow\n"
    assert not (tmp_path / "metrics.csv.tmp").exists()


def test_write_csv_row_atomic_roundtrip(tmp_path):
    import csv
    p = tmp_path / "run" / "metrics.csv"
    write_csv_row_atomic(p, {"HOTA": 1.5, "mode": "nms"})
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    assert rows == [{"HOTA": "1.5", "mode": "nms"}]
    assert not (p.parent / "metrics.csv.tmp").exists()


def test_run_batch_skips_done_groups_and_runs_pending(tmp_path):
    cfg = tiny_cfg(tmp_path, runs=2)
    root = results_root_for(cfg)
    _make_done(root, expand_grid(cfg)[0])
    calls = []

    def proc(key, specs, *a, **k):
        calls.append(key)
        return _writer_process(key, specs, *a, **k)

    summary = run_batch(cfg, process_fn=proc, load_model_fn=_fake_model,
                        load_frames_fn=_fake_frames, aggregate_fn=lambda r: None)
    assert len(calls) == 1
    assert summary["completed"] == 1
    assert summary["done_before"] == 1


def test_run_batch_skips_fully_done_group(tmp_path):
    cfg = tiny_cfg(tmp_path, runs=2)
    root = results_root_for(cfg)
    for s in expand_grid(cfg):
        _make_done(root, s)
    calls = []
    run_batch(cfg, process_fn=lambda key, specs, *a, **k: calls.append(key) or [],
              load_model_fn=_fake_model, load_frames_fn=_fake_frames, aggregate_fn=lambda r: None)
    assert calls == []


def test_run_batch_isolates_group_failures(tmp_path):
    cfg = tiny_cfg(tmp_path, runs=2)
    cfg.sequences = {"MOT17": ["MOT17-02", "MOT17-04"]}
    root = results_root_for(cfg)

    def proc(key, specs, model, frames, meta, cfg_, r, slug, force):
        if key[1] == "MOT17-02":
            raise RuntimeError("cuda hiccup")
        return _writer_process(key, specs, model, frames, meta, cfg_, r, slug, force)

    summary = run_batch(cfg, process_fn=proc, load_model_fn=_fake_model,
                        load_frames_fn=_fake_frames, aggregate_fn=lambda r: None)
    assert summary["failed"] == 1
    assert summary["completed"] == 2
    statuses = {r["key"]: r["status"] for r in read_ledger(root / "_runs.jsonl")}
    assert statuses.get("MOT17/MOT17-02/yolo11/n/nms") == "failed"
    assert statuses.get("MOT17/MOT17-04/yolo11/n/nms") == "done"


def test_group_specs_groups_by_detector_sequence(tmp_path):
    cfg = load_config(PAPER)
    grid = _expand(cfg)
    groups = group_specs(grid)
    assert len(groups) == 153
    for key, specs in groups:
        ds, seq, fam, sc, mode = key
        for s in specs:
            assert (s.dataset, s.sequence, s.family, s.scale, s.mode) == key
        assert len(specs) == len(cfg.trackers) * cfg.runs
    assert sum(len(s) for _, s in groups) == len(grid)


def test_log_to_writes_timestamped_log_and_captures_output(tmp_path):
    from core.runner import log_to
    with log_to(tmp_path / "logs", "run") as path:
        print("audit-marker-xyz")
    assert path.exists()
    assert path.parent == tmp_path / "logs"
    assert path.name.startswith("run_") and path.name.endswith(".log")
    assert "audit-marker-xyz" in path.read_text(encoding="utf-8")


def test_tee_write_survives_closed_file(tmp_path):
    import io
    from core.runner import _Tee
    sink = io.StringIO()
    fh = open(tmp_path / "x.log", "w", encoding="utf-8")
    tee = _Tee(sink, fh)
    tee.write("a")
    fh.close()
    tee.write("b")
    tee.flush()
    assert sink.getvalue() == "ab"


def test_log_to_captures_exception_traceback(tmp_path):
    import pytest
    from core.runner import log_to
    holder = {}
    with pytest.raises(ValueError):
        with log_to(tmp_path / "logs", "run") as p:
            holder["p"] = p
            raise ValueError("boom-marker-42")
    content = holder["p"].read_text(encoding="utf-8")
    assert "boom-marker-42" in content
    assert "Traceback" in content


def test_group_specs_orders_same_detector_contiguously(tmp_path):
    cfg = load_config(PAPER)
    groups = group_specs(_expand(cfg))
    triples = [(k[2], k[3], k[4]) for k, _ in groups]
    seen, prev = set(), None
    for t in triples:
        if t != prev:
            assert t not in seen, f"{t} reappears non-contiguously"
            seen.add(t)
            prev = t


def test_model_cache_loads_once_per_key_size1():
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return f"model-{calls['n']}"

    cache = ModelCache()
    assert cache.get("a", loader) == "model-1"
    assert cache.get("a", loader) == "model-1"
    assert calls["n"] == 1
    assert cache.get("b", loader) == "model-2"
    assert calls["n"] == 2
    cache.get("a", loader)
    assert calls["n"] == 3


def test_run_batch_force_reprocesses_all(tmp_path):
    cfg = tiny_cfg(tmp_path, runs=2)
    root = results_root_for(cfg)
    _make_done(root, expand_grid(cfg)[0])
    written = []

    def proc(key, specs, model, frames, meta, cfg_, r, slug, force):
        w = _writer_process(key, specs, model, frames, meta, cfg_, r, slug, force)
        written.extend(w)
        return w

    run_batch(cfg, force=True, process_fn=proc, load_model_fn=_fake_model,
              load_frames_fn=_fake_frames, aggregate_fn=lambda r: None)
    assert len(written) == 2


def test_portable_path_is_relative_to_the_repository(tmp_path):
    from core.runner import PROJECT_ROOT, portable_path
    rel = "results/t/MOT17/MOT17-02/yolo11_n_nms_bytetrack_0_run01/mot_results.txt"
    assert portable_path(PROJECT_ROOT / rel) == rel
    assert portable_path(tmp_path) == str(tmp_path.resolve())      # outside the repository


def test_status_forgets_a_failure_once_the_group_completes(tmp_path):
    from core.runner import status
    cfg = tiny_cfg(tmp_path)
    ledger = results_root_for(cfg) / "_runs.jsonl"
    append_record(ledger, {"key": "MOT17/MOT17-02/yolo11/n/nms", "status": "failed", "error": "boom"})
    assert len(status(cfg)["failed_records"]) == 1
    append_record(ledger, {"key": "MOT17/MOT17-02/yolo11/n/nms", "status": "done", "runs": 2})
    assert status(cfg)["failed_records"] == []
