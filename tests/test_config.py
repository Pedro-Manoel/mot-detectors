from pathlib import Path

import pytest

from core.config import expand_grid, is_valid_combination, load_config, weight_filename
from core.runner import dev_slug, run_dir_name

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "experiments" / "paper_main.yml"
SMOKE = ROOT / "experiments" / "smoke.yml"

PAPER_SEQUENCES = {
    "MOT17":   ["MOT17-02", "MOT17-04", "MOT17-05"],
    "MOT20":   ["MOT20-02", "MOT20-03", "MOT20-05"],
    "SOMPT22": ["SOMPT22-07", "SOMPT22-10", "SOMPT22-12"],
}


def _configs(path) -> set:
    return {(g.dataset, g.sequence, g.family, g.scale, g.mode, g.tracker)
            for g in expand_grid(load_config(path))}


def test_weight_filename():
    assert weight_filename("yolo11", "n") == "yolo11n.pt"
    assert weight_filename("yolo26", "x") == "yolo26x.pt"
    assert weight_filename("rtdetr", "l") == "rtdetr-l.pt"


def test_validity_rule():
    assert is_valid_combination("yolo26", "end2end") is True
    assert is_valid_combination("yolo11", "end2end") is False
    assert is_valid_combination("rtdetr", "end2end") is False
    assert is_valid_combination("yolo11", "nms") is True


def test_paper_config_matches_the_sequences_reported_in_paper_tex():
    assert load_config(PAPER).sequences == PAPER_SEQUENCES


def test_paper_grid_is_306_configs_times_runs():
    cfg = load_config(PAPER)
    assert len(_configs(PAPER)) == 306
    assert len(expand_grid(cfg)) == 306 * cfg.runs
    assert cfg.runs == 5


def test_smoke_grid_is_part_of_the_paper_grid():
    assert _configs(SMOKE) <= _configs(PAPER)


def test_end2end_only_yolo26():
    grid = expand_grid(load_config(PAPER))
    assert {g.family for g in grid if g.mode == "end2end"} == {"yolo26"}


def test_reid_only_for_botsort():
    for g in expand_grid(load_config(PAPER)):
        if g.tracker == "botsort":
            assert g.reid_weights
        else:
            assert g.reid_weights is None


def test_run_dir_naming_matches_legacy():
    cfg = load_config(PAPER)
    g = next(x for x in expand_grid(cfg)
             if x.family == "yolo11" and x.scale == "n" and x.tracker == "bytetrack" and x.run_id == 1)
    assert run_dir_name(g, dev_slug(cfg.device)) == "yolo11_n_nms_bytetrack_0_run01"


def _yaml(tmp_path, text):
    y = tmp_path / "exp.yml"
    y.write_text(text, encoding="utf-8")
    return y


def test_unknown_keys_and_modes_are_rejected(tmp_path):
    paper = PAPER.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key 'keep_track'"):
        load_config(_yaml(tmp_path, paper + "keep_track: false\n"))
    with pytest.raises(ValueError, match="mode 'NMS'"):
        load_config(_yaml(tmp_path, paper.replace("modes: [nms] }", "modes: [NMS] }", 1)))


def test_mot17_detector_suffix_is_dropped(tmp_path):
    paper = PAPER.read_text(encoding="utf-8")
    cfg = load_config(_yaml(tmp_path, paper.replace("[MOT17-02,", "[MOT17-02-FRCNN,")))
    assert cfg.sequences["MOT17"] == ["MOT17-02", "MOT17-04", "MOT17-05"]
