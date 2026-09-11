import hashlib
import json
from pathlib import Path

import pandas as pd

from core import evaluate, reproduce
from core.config import load_config, weight_filename


def test_resolve_python_accepts_the_other_os_layout(tmp_path):
    posix = tmp_path / "a" / "bin" / "python"
    posix.parent.mkdir(parents=True)
    posix.write_text("")
    assert evaluate.resolve_python(str(tmp_path / "a" / "Scripts" / "python.exe")) == str(posix.absolute())

    win = tmp_path / "b" / "Scripts" / "python.exe"
    win.parent.mkdir(parents=True)
    win.write_text("")
    assert evaluate.resolve_python(str(tmp_path / "b" / "bin" / "python")) == str(win.absolute())


def test_resolve_python_keeps_a_bare_command():
    assert evaluate.resolve_python("python") == "python"


def test_text_hash_ignores_line_endings(tmp_path):
    crlf, lf = tmp_path / "crlf.txt", tmp_path / "lf.txt"
    crlf.write_bytes(b"1,2\r\n3,4\r\n")
    lf.write_bytes(b"1,2\n3,4\n")
    assert reproduce.file_sha256(crlf, text=True) == reproduce.file_sha256(lf, text=True)
    assert reproduce.file_sha256(crlf) != reproduce.file_sha256(lf)


def _cfg(tmp_path):
    y = tmp_path / "exp.yml"
    y.write_text(
        "name: t\ndevice: cpu\nruns: 1\n"
        "detector: {conf: 0.25, imgsz: 640}\n"
        f"trackeval: {{root: '{(tmp_path / 'TrackEval').as_posix()}', python: python}}\n"
        f"paths: {{prepared_root: '{(tmp_path / 'prepared').as_posix()}'}}\n"
        "detectors: [{family: yolo11, scales: [n], modes: [nms]}]\n"
        "trackers: [bytetrack]\n"
        "sequences: {MOT17: [MOT17-02, MOT17-04, MOT17-05]}\n",
        encoding="utf-8")
    return load_config(y)


def test_check_files_compares_weights_and_prepared_gt(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "yolo11n.pt").write_bytes(b"weights")
    for seq, content in (("MOT17-02", b"1,1\r\n"), ("MOT17-04", b"changed\n")):
        gt = tmp_path / "prepared" / "MOT17" / seq / "gt" / "gt.txt"
        gt.parent.mkdir(parents=True)
        gt.write_bytes(content)
    manifest = {
        "weights_sha256": {"yolo11n.pt": reproduce.file_sha256(tmp_path / "yolo11n.pt"),
                           "yolo26x.pt": "0" * 64},       # not used by this config
        "prepared_gt_sha256": {"MOT17-02": hashlib.sha256(b"1,1\n").hexdigest(),  # LF form
                               "MOT17-04": "0" * 64,
                               "MOT17-05": "0" * 64},
    }
    status = {item: s for s, item, _ in reproduce.check_files(cfg, manifest)}
    assert status == {"yolo11n.pt": "PASS", "GT MOT17-02": "PASS",
                      "GT MOT17-04": "FAIL", "GT MOT17-05": "WARN"}


def test_paper_environment_covers_the_paper_grid():
    env = json.loads(reproduce.PAPER_ENVIRONMENT.read_text(encoding="utf-8"))
    cfg = load_config(Path(__file__).resolve().parent.parent / "experiments" / "paper_main.yml")
    assert set(env["prepared_gt_sha256"]) == {s for seqs in cfg.sequences.values() for s in seqs}
    weights = {weight_filename(d.family, s) for d in cfg.detectors for s in d.scales}
    assert set(env["weights_sha256"]) == weights | {Path(cfg.reid_weights).name}


def _results(tmp_path, name, shift=0.0):
    rows = []
    for fam, sc, hota in (("rtdetr", "l", 40.0), ("yolo11", "n", 35.0)):
        for ds, dens in (("MOT17", 30.0), ("MOT20", 150.0)):
            for seq in ("A", "B"):
                for run in (1, 2):
                    rows.append(dict(model_family=fam, model_scale=sc, mode="nms",
                                     tracker="bytetrack", dataset=ds, sequence=f"{ds}-{seq}",
                                     run_id=run, density=dens, HOTA=hota + shift, IDF1=hota,
                                     MOTA=hota, IDSW=10.0, Frag=20.0, avg_total_ms=10.0,
                                     fps_total=100.0))
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "all_results.csv", index=False)
    return d


def test_compare_passes_within_tolerance_and_fails_beyond(tmp_path):
    ref = _results(tmp_path, "ref")
    ok, lines = reproduce.compare_results(ref, _results(tmp_path, "close", shift=0.3))
    assert ok
    assert any("best HOTA on MOT20: RT-DETR-L (reference RT-DETR-L)" in line for line in lines)
    ok, lines = reproduce.compare_results(ref, _results(tmp_path, "far", shift=1.5))
    assert not ok
    assert any(line.startswith("[FAIL]") and " HOTA " in line for line in lines)


def test_resolve_python_keeps_the_venv_symlink(tmp_path):
    import pytest
    base = tmp_path / "base" / "python"
    base.parent.mkdir()
    base.write_text("")
    link = tmp_path / "venv" / "bin" / "python"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(base)
    except OSError:
        pytest.skip("symbolic links are not available")
    assert evaluate.resolve_python(str(link)) == str(link.absolute())


def test_check_reports_a_missing_trackeval_interpreter(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.trackeval_python = str(tmp_path / "nowhere" / "bin" / "python")
    rows = {item: (s, detail) for s, item, detail in reproduce.check_software(cfg, {})}
    status, detail = rows["trackeval python"]
    assert status == "FAIL" and "not found" in detail


def test_compare_restricts_a_partial_run_to_the_shared_sequences(tmp_path):
    ref = _results(tmp_path, "ref")
    full = pd.read_csv(ref / "all_results.csv")
    full.loc[full["sequence"].str.endswith("-B"), "HOTA"] += 10.0    # the two sequences differ a lot
    full.to_csv(ref / "all_results.csv", index=False)
    part = tmp_path / "part"
    part.mkdir()
    full[full["sequence"] == "MOT17-A"].to_csv(part / "all_results.csv", index=False)
    ok, lines = reproduce.compare_results(ref, part)
    assert ok, lines
    assert any(line.startswith("[INFO] comparing") for line in lines)
