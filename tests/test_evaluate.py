from pathlib import Path

from core.evaluate import parse_trackeval_summary


def test_parse_summary_row(tmp_path: Path):
    (tmp_path / "pedestrian_summary.txt").write_text(
        "HOTA IDF1 MOTA IDSW Frag\n34.43 40.20 30.24 45 81\n", encoding="utf-8")
    parsed = parse_trackeval_summary(tmp_path)
    assert parsed["HOTA"] == 34.43
    assert parsed["IDSW"] == 45
    assert parsed["IDF1"] == 40.20


def test_summary_aliases_fill_the_short_names(tmp_path: Path):
    (tmp_path / "pedestrian_summary.txt").write_text(
        "HOTA CLR_Re CLR_Pr CLR_FP CLR_FN\n50.5 60.5 70.5 5 7\n", encoding="utf-8")
    m = parse_trackeval_summary(tmp_path)
    assert (m["Recall"], m["Precision"], m["FP"], m["FN"]) == (60.5, 70.5, 5, 7)
