from pathlib import Path
from core.config import (
    load_config, expand_grid, weight_filename, is_valid_combination,
)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "experiments" / "paper_main.yml"
BALANCED = ROOT / "experiments" / "balanced_12seq.yml"
SMOKE = ROOT / "experiments" / "smoke.yml"

PAPER_SEQUENCES = {
    "MOT17":   ["MOT17-02", "MOT17-04", "MOT17-05"],
    "MOT20":   ["MOT20-02", "MOT20-03", "MOT20-05"],
    "SOMPT22": ["SOMPT22-07", "SOMPT22-10", "SOMPT22-12"],
}


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
    grid = expand_grid(cfg)
    configs = {(g.dataset, g.sequence, g.family, g.scale, g.mode, g.tracker) for g in grid}
    assert len(configs) == 306
    assert len(grid) == 306 * cfg.runs
    assert cfg.runs == 5


def test_balanced_grid_is_408_configs_and_is_a_separate_experiment():
    cfg = load_config(BALANCED)
    grid = expand_grid(cfg)
    configs = {(g.dataset, g.sequence, g.family, g.scale, g.mode, g.tracker) for g in grid}
    assert len(configs) == 408
    assert cfg.name != load_config(PAPER).name


def test_paper_and_balanced_differ_only_in_sequences():
    paper, balanced = load_config(PAPER), load_config(BALANCED)
    for field in ("device", "runs", "conf", "imgsz", "half", "do_preproc",
                  "keep_tracks", "trackers", "reid_weights", "save_detections",
                  "shared_detection"):
        assert getattr(paper, field) == getattr(balanced, field), field
    assert paper.sequences != balanced.sequences


def test_save_detections_flag():
    from core.config import Config
    assert Config.__dataclass_fields__["save_detections"].default is False
    for path in (PAPER, BALANCED, SMOKE):
        assert load_config(path).save_detections is True, path


def test_end2end_only_yolo26():
    cfg = load_config(PAPER)
    grid = expand_grid(cfg)
    e2e_families = {g.family for g in grid if g.mode == "end2end"}
    assert e2e_families == {"yolo26"}


def test_reid_only_for_botsort():
    cfg = load_config(PAPER)
    grid = expand_grid(cfg)
    for g in grid:
        if g.tracker == "botsort":
            assert g.reid_weights
        else:
            assert g.reid_weights is None


def test_run_dir_naming_matches_legacy():
    from core.runner import dev_slug, run_dir_name
    cfg = load_config(PAPER)
    g = next(x for x in expand_grid(cfg)
             if x.family == "yolo11" and x.scale == "n" and x.tracker == "bytetrack" and x.run_id == 1)
    assert run_dir_name(g, dev_slug(cfg.device)) == "yolo11_n_nms_bytetrack_0_run01"


def test_shared_detection_is_opt_in(tmp_path):
    from core.config import Config
    assert Config.__dataclass_fields__["shared_detection"].default is False
    for path in (PAPER, BALANCED, SMOKE):
        assert load_config(path).shared_detection is False, path
    y = tmp_path / "exp.yml"
    y.write_text(PAPER.read_text(encoding="utf-8").replace("shared_detection: false",
                                                           "shared_detection: true"),
                 encoding="utf-8")
    assert load_config(y).shared_detection is True


def _yaml(tmp_path, text):
    y = tmp_path / "exp.yml"
    y.write_text(text, encoding="utf-8")
    return y


def test_unknown_keys_and_modes_are_rejected(tmp_path):
    import pytest
    paper = PAPER.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="shared_detections"):
        load_config(_yaml(tmp_path, paper + "shared_detections: true\n"))
    with pytest.raises(ValueError, match="mode 'NMS'"):
        load_config(_yaml(tmp_path, paper.replace("modes: [nms] }", "modes: [NMS] }", 1)))


def test_mot17_detector_suffix_is_dropped(tmp_path):
    paper = PAPER.read_text(encoding="utf-8")
    cfg = load_config(_yaml(tmp_path, paper.replace("[MOT17-02,", "[MOT17-02-FRCNN,")))
    assert cfg.sequences["MOT17"] == ["MOT17-02", "MOT17-04", "MOT17-05"]
