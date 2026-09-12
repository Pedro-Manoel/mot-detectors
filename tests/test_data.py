import json

from core.data import load_density_registry, resolve_density


def test_load_density_registry_nested_by_dataset(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"MOT17": {"MOT17-02": 31.0}, "MOT20": {"MOT20-05": 226.6}}),
                 encoding="utf-8")
    assert load_density_registry(p) == {"MOT17-02": 31.0, "MOT20-05": 226.6}


def test_load_density_registry_missing_file_is_empty(tmp_path):
    assert load_density_registry(tmp_path / "nope.json") == {}


def test_resolve_density_registered_wins_else_computed():
    reg = {"MOT17-02": 31.0}
    assert resolve_density("MOT17-02", 28.5, reg) == (31.0, True)
    assert resolve_density("NEW-SEQ-01", 12.3, reg) == (12.3, False)


PAPER_DENSITIES = {
    "MOT17":   {"MOT17-02": 31.0, "MOT17-04": 45.3, "MOT17-05": 8.3},
    "MOT20":   {"MOT20-02": 72.7, "MOT20-03": 148.3, "MOT20-05": 226.6},
    "SOMPT22": {"SOMPT22-07": 46.0, "SOMPT22-10": 57.0, "SOMPT22-12": 40.0},
}


def test_shipped_registry_is_exactly_the_paper_densities():
    reg = load_density_registry()
    assert reg == {s: d for seqs in PAPER_DENSITIES.values() for s, d in seqs.items()}


def test_shipped_registry_reproduces_the_densities_quoted_in_the_paper():
    reg = load_density_registry()
    avg = {ds: sum(reg[s] for s in seqs) / len(seqs) for ds, seqs in PAPER_DENSITIES.items()}
    assert round(avg["MOT17"], 1) == 28.2
    assert round(avg["SOMPT22"], 1) == 47.7
    assert round(avg["MOT20"], 1) == 149.2
