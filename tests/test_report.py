import re
from pathlib import Path

import pandas as pd
import pytest

from core.report import (_efficiency_to_latex, _quality_to_latex, benchmark_order, build_tables,
                         efficiency_table, quality_table)

PAPER_RESULTS = Path(__file__).resolve().parent.parent / "paper_results"


def _all_results() -> pd.DataFrame:
    configs = [("yolo11", "n", "nms", 30.0), ("rtdetr", "l", "nms", 40.0),
               ("yolo26", "n", "nms", 32.0), ("yolo26", "n", "end2end", 31.0)]
    rows = []
    for tracker in ("bytetrack", "botsort"):
        for ds, density, drop in (("MOT20", 150.0, 15.0), ("MOT17", 30.0, 0.0)):
            for seq in ("A", "B"):
                for run in (1, 2):
                    for fam, sc, mode, base in configs:
                        h = base - drop + (0.5 if run == 2 else -0.5)
                        e2e = mode == "end2end"
                        rows.append(dict(
                            model_family=fam, model_scale=sc, mode=mode, tracker=tracker,
                            dataset=ds, sequence=f"{ds}-{seq}", run_id=run, density=density,
                            HOTA=h, IDF1=h + 5, MOTA=h,
                            IDSW=23.0 if e2e else 20.0, Frag=50.0 if e2e else 40.0,
                            avg_total_ms=10.0 + run, fps_total=1000.0 / (10.0 + run)))
    return pd.DataFrame(rows)


def test_quality_table_averages_runs_and_orders_benchmarks_by_density():
    q = quality_table(_all_results(), "bytetrack")
    assert list(dict.fromkeys(q.columns.get_level_values("dataset"))) == ["MOT17", "MOT20"]
    assert q.loc[("yolo11", "n", "nms"), ("MOT17", "HOTA")] == pytest.approx(30.0)
    assert q.loc[("rtdetr", "l", "nms"), ("MOT20", "HOTA")] == pytest.approx(25.0)
    assert q[("MOT17", "HOTA")].idxmax() == ("rtdetr", "l", "nms")


def test_build_tables_writes_the_paper_tables(tmp_path):
    csv = tmp_path / "all_results.csv"
    _all_results().to_csv(csv, index=False)
    written = build_tables(csv, tmp_path / "tables")
    assert {p.name for p in written} == {
        "quality_bytetrack.csv", "quality_bytetrack.tex", "quality_botsort.csv",
        "quality_botsort.tex", "efficiency.csv", "efficiency.tex"}
    for p in written:
        assert bytes([13, 13]) not in p.read_bytes(), f"{p.name} has a doubled carriage return"


def test_quality_latex_follows_the_paper_format():
    tex = _quality_to_latex(quality_table(_all_results(), "botsort"), "botsort")
    assert r"\caption{Results with BoT-SORT.}" in tex and r"\label{tab:quality_botsort}" in tex
    assert r"\multicolumn{5}{c}{MOT17} & \multicolumn{5}{c}{MOT20}" in tex
    assert r"HOTA$\uparrow$" in tex and r"Frag$\downarrow$" in tex
    for label in ("YOLO11-N", "RT-DETR-L", "YOLO26-N (NMS)", "YOLO26-N (E2E)"):
        assert re.search("\n" + re.escape(label) + " +& ", tex)
    assert r"\textbf{40.00}" in tex
    assert tex.count(r"\midrule") == 3


def test_efficiency_latex_follows_the_paper_format():
    df = _all_results()
    tex = _efficiency_to_latex(efficiency_table(df), benchmark_order(df))
    assert r"\caption{Computational efficiency.}" in tex and r"\label{tab:efficiency}" in tex
    assert r"\multicolumn{2}{c}{ByteTrack} & \multicolumn{2}{c}{BoT-SORT}" in tex
    assert r"FPS$\uparrow$ & Lat. (ms)$\downarrow$" in tex
    assert r"{\scriptsize$\pm$" in tex


def test_efficiency_fps_is_reciprocal_of_latency():
    rows = []
    for seq, lat in [("S1", 5.0), ("S2", 20.0)]:
        for run in (1, 2):
            ms = lat + 0.1 * run
            rows.append(dict(model_family="yolo11", model_scale="n", mode="nms",
                             tracker="bytetrack", dataset="D", sequence=seq,
                             avg_total_ms=ms, fps_total=1000.0 / ms))
    eff = efficiency_table(pd.DataFrame(rows))
    r = eff.iloc[0]
    assert r.fps_mean * r.lat_mean == pytest.approx(1000.0)


def test_paper_results_hold_the_paper_numbers():
    q = quality_table(pd.read_csv(PAPER_RESULTS / "all_results.csv"), "bytetrack")
    row = q.loc[("yolo11", "n", "nms")]
    assert round(row[("MOT17", "HOTA")], 2) == 34.43
    assert round(row[("MOT20", "HOTA")], 2) == 12.60
    assert q[("MOT17", "HOTA")].idxmax() == ("rtdetr", "x", "nms")


def test_paper_tables_follow_from_all_results():
    df = pd.read_csv(PAPER_RESULTS / "all_results.csv")
    tables = PAPER_RESULTS / "tables"
    for trk in ("bytetrack", "botsort"):
        shipped = (tables / f"quality_{trk}.tex").read_text(encoding="utf-8")
        assert _quality_to_latex(quality_table(df, trk), trk) == shipped
    shipped = (tables / "efficiency.tex").read_text(encoding="utf-8")
    assert _efficiency_to_latex(efficiency_table(df), benchmark_order(df)) == shipped


def test_latex_marks_missing_cells_and_single_runs():
    df = _all_results()
    df = df[~((df.model_family == "yolo11") & (df.dataset == "MOT20"))]
    tex = _quality_to_latex(quality_table(df, "botsort"), "botsort")
    assert "--" in tex and "nan" not in tex
    assert r"\textbf{25.00}" in tex
    one = df[df.run_id == 1]
    assert "nan" not in _efficiency_to_latex(efficiency_table(one), benchmark_order(one))
