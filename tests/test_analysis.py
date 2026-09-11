import math

import pandas as pd

from core import analysis


def _row(family, scale, mode, dataset, seq, run, **metrics):
    base = dict(model_family=family, model_scale=scale, mode=mode,
                dataset=dataset, sequence=seq, run_id=run, tracker="bytetrack")
    base.update(metrics)
    return base


def _frame():
    rows = []
    for ds, rt, yo in [("MOT17", 50.0, 45.0), ("MOT20", 45.0, 30.0)]:
        for seq in ("A", "B"):
            for run in (1, 2):
                rows.append(_row("rtdetr", "l", "nms", ds, seq, run,
                                 HOTA=rt, DetA=rt - 2, AssA=rt + 2, LocA=80.0,
                                 IDF1=rt, IDR=rt, IDP=rt, Recall=rt, Precision=90.0,
                                 MT=10, ML=1, Frag=20, IDSW=5, MOTA=rt))
                rows.append(_row("yolo11", "n", "nms", ds, seq, run,
                                 HOTA=yo, DetA=yo - 5, AssA=yo + 1, LocA=78.0,
                                 IDF1=yo, IDR=yo, IDP=yo, Recall=yo, Precision=85.0,
                                 MT=8, ML=3, Frag=30, IDSW=9, MOTA=yo))
    return pd.DataFrame(rows)


def _yolo26_frame():
    rows = []
    for scale in ("n", "m"):
        for seq in ("A", "B"):
            for run in (1, 2):
                rows.append(_row("yolo26", scale, "nms", "MOT17", seq, run,
                                 HOTA=40.0, IDF1=50.0, Frag=20, IDSW=5))
                rows.append(_row("yolo26", scale, "end2end", "MOT17", seq, run,
                                 HOTA=39.0, IDF1=49.0, Frag=30, IDSW=6))
    return pd.DataFrame(rows)


def test_confidence_interval_basic_stats():
    ci = analysis.confidence_interval([2.0, 4.0])
    assert ci["mean"] == 3.0
    assert math.isclose(ci["std"], math.sqrt(2.0))
    assert ci["n"] == 2
    assert ci["ci_low"] < 3.0 < ci["ci_high"]
    assert math.isclose(ci["ci_high"] - ci["mean"], ci["mean"] - ci["ci_low"])
    assert ci["half_width"] > 0


def test_confidence_interval_single_value_has_no_width():
    ci = analysis.confidence_interval([7.0])
    assert ci["mean"] == 7.0
    assert ci["n"] == 1
    assert ci["half_width"] == 0.0
    assert ci["ci_low"] == ci["ci_high"] == 7.0


def test_confidence_interval_ignores_nan():
    ci = analysis.confidence_interval([2.0, float("nan"), 4.0])
    assert ci["n"] == 2
    assert ci["mean"] == 3.0


def test_hota_decomposition_has_deta_assa_loca_per_benchmark():
    t = analysis.hota_decomposition_table(_frame(), "bytetrack")
    row = t.loc[("rtdetr", "l", "nms")]
    assert math.isclose(row[("MOT17", "HOTA")], 50.0)
    assert math.isclose(row[("MOT17", "DetA")], 48.0)
    assert math.isclose(row[("MOT17", "AssA")], 52.0)


def test_benchmark_order_is_by_density_not_alpha():
    from core.report import benchmark_order
    rows = []
    for ds, dens in [("ZZZ", 5.0), ("AAA", 200.0)]:
        for run in (1, 2):
            rows.append(_row("yolo11", "n", "nms", ds, "S", run, HOTA=1.0, density=dens))
    df = pd.DataFrame(rows)
    assert benchmark_order(df) == ["ZZZ", "AAA"]


def test_degradation_uses_density_endpoints_not_hardcoded():
    rows = []
    for ds, dens, hota in [("MID", 50.0, 40.0), ("LOW", 10.0, 60.0), ("HIGH", 150.0, 20.0)]:
        for run in (1, 2):
            rows.append(_row("rtdetr", "l", "nms", ds, "S", run,
                             HOTA=hota, DetA=hota, AssA=hota, density=dens))
    d = analysis.degradation_table(pd.DataFrame(rows), "bytetrack", metrics=("HOTA",))
    row = d.iloc[0]
    assert row["sparse_benchmark"] == "LOW" and row["dense_benchmark"] == "HIGH"
    assert math.isclose(row["HOTA_LOW"], 60.0) and math.isclose(row["HOTA_HIGH"], 20.0)
    assert math.isclose(row["HOTA_delta"], -40.0)


def test_degradation_reports_sparse_dense_delta():
    d = analysis.degradation_table(_frame(), "bytetrack", metrics=("HOTA",))
    rt = d[(d.family == "rtdetr")].iloc[0]
    yo = d[(d.family == "yolo11")].iloc[0]
    assert math.isclose(rt["HOTA_MOT17"], 50.0)
    assert math.isclose(rt["HOTA_MOT20"], 45.0)
    assert math.isclose(rt["HOTA_delta"], -5.0)
    assert math.isclose(yo["HOTA_delta"], -15.0)
    assert rt["HOTA_delta"] > yo["HOTA_delta"]


def test_nms_vs_e2e_paired_diff_and_pvalue():
    s = analysis.nms_vs_e2e_significance(_yolo26_frame(), "bytetrack",
                                         metrics=("HOTA", "Frag"))
    overall = s[s["scale"] == "ALL"].iloc[0]
    assert math.isclose(overall["HOTA_e2e_minus_nms"], -1.0)
    assert math.isclose(overall["Frag_e2e_minus_nms"], 10.0)
    assert 0.0 <= overall["HOTA_pvalue"] <= 1.0
    assert overall["n_pairs"] == 4          # 2 scales x 2 sequences; the 2 runs collapse


def test_nms_vs_e2e_pairs_sequences_not_runs():
    base = _yolo26_frame()
    more_runs = pd.concat([base, base.assign(run_id=base["run_id"] + 2)], ignore_index=True)
    a = analysis.nms_vs_e2e_significance(base, "bytetrack", metrics=("HOTA",))
    b = analysis.nms_vs_e2e_significance(more_runs, "bytetrack", metrics=("HOTA",))
    assert a["n_pairs"].tolist() == b["n_pairs"].tolist()
    assert a["HOTA_pvalue"].tolist() == b["HOTA_pvalue"].tolist()


def test_tracker_hyperparameters_lists_effective_values():
    h = analysis.tracker_hyperparameters()
    val = {(r.tracker, r.parameter): r.value for r in h.itertuples()}
    assert val[("bytetrack", "min_conf")] == 0.1
    assert val[("bytetrack", "track_thresh")] == 0.45
    assert val[("bytetrack", "det_thresh")] == 0.45      # derived from track_thresh
    assert val[("botsort", "track_buffer")] == 30
    assert val[("botsort", "cmc_method")] == "ecc"
    assert val[("botsort", "with_reid")] is True


def test_nms_vs_e2e_empty_when_no_e2e():
    df = _frame()
    s = analysis.nms_vs_e2e_significance(df, "bytetrack", metrics=("HOTA",))
    assert s.empty


def test_bootstrap_ci_constant_is_zero_width():
    ci = analysis.bootstrap_ci([5.0, 5.0, 5.0])
    assert ci["boot_ci_low"] == 5.0 and ci["boot_ci_high"] == 5.0


def test_bootstrap_ci_deterministic_and_brackets_mean():
    a = analysis.bootstrap_ci([40.0, 50.0], seed=0)
    b = analysis.bootstrap_ci([40.0, 50.0], seed=0)
    assert a == b
    assert a["boot_ci_low"] <= 45.0 <= a["boot_ci_high"]
    assert 40.0 <= a["boot_ci_low"] and a["boot_ci_high"] <= 50.0


def test_between_sequence_ci_collapses_runs_to_sequences():
    rows = []
    for seq, val in [("A", 40.0), ("B", 50.0)]:
        for run in (1, 2):
            rows.append(_row("rtdetr", "l", "nms", "MOT17", seq, run, HOTA=val))
    df = pd.DataFrame(rows)
    t = analysis.between_sequence_ci(df, "bytetrack", "HOTA")
    row = t.iloc[0]
    assert row["n_sequences"] == 2
    assert math.isclose(row["mean"], 45.0)
    assert math.isclose(row["std_between_seq"], math.sqrt(50.0))
    assert "boot_ci_low" in row and "t_ci_low" in row


def test_attribution_table_carries_detection_and_association_columns():
    a = analysis.attribution_table(_frame(), "bytetrack")
    for col in ("Recall", "Precision", "DetA", "AssA", "IDF1", "MT", "ML"):
        assert col in a.columns
    rt17 = a[(a.family == "rtdetr") & (a.dataset == "MOT17")].iloc[0]
    assert math.isclose(rt17["Recall"], 50.0)
    assert math.isclose(rt17["Precision"], 90.0)
