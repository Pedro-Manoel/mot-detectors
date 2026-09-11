from __future__ import annotations

import csv
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import Config, RunSpec, expand_grid, weight_filename

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _Tee:
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        if not self._fh.closed:
            self._fh.write(data)
            self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        if not self._fh.closed:
            self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextmanager
def log_to(log_dir, command: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"{command}_{ts}.log"
    fh = open(path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(old_out, fh), _Tee(old_err, fh)
    try:
        print(f"# {command}  started {datetime.now().isoformat(timespec='seconds')}")
        yield path
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        print(f"# {command}  finished {datetime.now().isoformat(timespec='seconds')}")
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()


def dev_slug(device: str) -> str:
    return str(device).replace(":", "").replace("cuda", "gpu")


def run_dir_name(spec: RunSpec, slug: str) -> str:
    return f"{spec.family}_{spec.scale}_{spec.mode}_{spec.tracker}_{slug}_run{spec.run_id:02d}"


def results_root_for(cfg: Config, quicktest: bool = False) -> Path:
    base = Path(cfg.output_root) / cfg.name
    return base.parent / (base.name + "_quicktest") if quicktest else base


def run_dir_for(results_root: Path, spec: RunSpec, slug: str) -> Path:
    return Path(results_root) / spec.dataset / spec.sequence / run_dir_name(spec, slug)


def portable_path(path) -> str:
    # Relative to the repository root with forward slashes, so metrics.csv carries no
    # machine-specific prefix; a results folder outside the repository keeps its full path.
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def is_done(run_dir: Path) -> bool:
    return (Path(run_dir) / "metrics.csv").exists()


def partition(grid, results_root: Path, slug: str):
    done, pending = [], []
    for spec in grid:
        (done if is_done(run_dir_for(results_root, spec, slug)) else pending).append(spec)
    return done, pending


def group_specs(grid):
    groups: dict = {}
    for s in grid:
        key = (s.dataset, s.sequence, s.family, s.scale, s.mode)
        groups.setdefault(key, []).append(s)
    return sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][4], kv[0][0], kv[0][1]))


class ModelCache:
    def __init__(self):
        self._key = None
        self._model = None

    def get(self, key, loader):
        if key != self._key:
            self._model = None          # drop the cached model before loading the next one
            self._model = loader()
            self._key = key
        return self._model


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv_row_atomic(path: Path, row: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    os.replace(tmp, path)
    return path


def append_record(ledger_path: Path, record: dict) -> None:
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_ledger(ledger_path: Path) -> list[dict]:
    p = Path(ledger_path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_load_model(weight, mode, cfg: Config):
    from . import detect
    detect.setup_gpu(cfg.device, cfg.half)
    detect.patch_numpy_compat()
    return detect.load_detector(weight, cfg.device, cfg.half, mode)


def _default_load_frames(cfg: Config, dataset, sequence, max_frames):
    from . import data
    frames, meta = data.load_sequence(Path(cfg.prepared_root).resolve(), dataset, sequence)
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    return frames, meta


def run_batch(cfg: Config, force: bool = False, max_frames: int = 0, *,
              process_fn=None, load_model_fn=None, load_frames_fn=None,
              aggregate_fn=None, aggregate_every: int = 10) -> dict:
    quicktest = bool(max_frames and max_frames > 0)
    if process_fn is None:
        from . import pipeline
        process_fn = pipeline.process_group
    if load_model_fn is None:
        load_model_fn = _default_load_model
    if load_frames_fn is None:
        load_frames_fn = _default_load_frames
    if aggregate_fn is None:
        from . import report
        aggregate_fn = report.aggregate

    slug = dev_slug(cfg.device)
    results_root = results_root_for(cfg, quicktest)
    results_root.mkdir(parents=True, exist_ok=True)
    grid = expand_grid(cfg)
    groups = group_specs(grid)
    done_before = sum(1 for s in grid if is_done(run_dir_for(results_root, s, slug)))
    active = [(k, ss) for k, ss in groups
              if force or any(not is_done(run_dir_for(results_root, s, slug)) for s in ss)]
    ledger = results_root / "_runs.jsonl"

    label = "quicktest" if quicktest else "run"
    print(f"[{label}] groups {len(groups)} | active {len(active)} | "
          f"runs done {done_before}/{len(grid)}")

    cache = ModelCache()
    completed = failed = 0
    interrupted = False
    try:
        for gi, (key, specs) in enumerate(active, 1):
            dataset, sequence, family, scale, mode = key
            gkey = f"{dataset}/{sequence}/{family}/{scale}/{mode}"
            print(f"[group {gi}/{len(active)}] {gkey}")
            try:
                weight = weight_filename(family, scale)
                model = None            # release the previous group's model before a new one loads
                model = cache.get((weight, mode), lambda: load_model_fn(weight, mode, cfg))
                frames, meta = load_frames_fn(cfg, dataset, sequence, max_frames)
                written = process_fn(key, specs, model, frames, meta, cfg,
                                     results_root, slug, force)
            except KeyboardInterrupt:
                interrupted = True
                break
            except Exception as e:
                failed += 1
                if not quicktest:
                    append_record(ledger, {"key": gkey, "status": "failed",
                                           "error": repr(e), "ts": _now()})
                print(f"  FAILED: {e!r} (continuing)")
                continue
            completed += len(written)
            if not quicktest:
                append_record(ledger, {"key": gkey, "status": "done",
                                       "runs": len(written), "ts": _now()})
            if (not quicktest) and aggregate_every and gi % aggregate_every == 0:
                try:
                    aggregate_fn(results_root)
                except Exception:
                    pass
    finally:
        if (not quicktest) and completed:
            try:
                aggregate_fn(results_root)
            except Exception:
                pass

    if interrupted:
        print("\nInterrupted. Resume with the same command (done runs are skipped).")

    return {"total": len(grid), "done_before": done_before,
            "completed": completed, "failed": failed,
            "interrupted": interrupted, "results_root": str(results_root)}


def status(cfg: Config) -> dict:
    slug = dev_slug(cfg.device)
    results_root = results_root_for(cfg)
    grid = expand_grid(cfg)
    done, pending = partition(grid, results_root, slug)
    # A group that failed and later completed is no longer a failure: keep its latest record.
    latest = {r.get("key"): r for r in read_ledger(results_root / "_runs.jsonl")}
    failed = [r for r in latest.values() if r.get("status") == "failed"]
    return {"total": len(grid), "done": len(done), "pending": len(pending),
            "failed_records": failed, "results_root": str(results_root)}
