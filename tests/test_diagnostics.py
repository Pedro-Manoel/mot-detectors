import math

import numpy as np

from core import diagnostics as dg


def _box(x1, y1, x2, y2, conf=1.0, cls=0):
    return [x1, y1, x2, y2, conf, cls]


def test_iou_matrix_identical_disjoint_and_half():
    a = np.array([_box(0, 0, 10, 10)])
    b = np.array([_box(0, 0, 10, 10),
                  _box(100, 100, 110, 110),
                  _box(5, 0, 15, 10)])
    m = dg.iou_matrix(a, b)
    assert math.isclose(m[0, 0], 1.0)
    assert math.isclose(m[0, 1], 0.0)
    assert math.isclose(m[0, 2], 1.0 / 3.0, rel_tol=1e-6)


def test_duplicate_rate_counts_overlapping_pairs():
    frame_dup = np.array([_box(0, 0, 10, 10), _box(0, 0, 10, 10)])
    frame_clean = np.array([_box(0, 0, 10, 10), _box(100, 100, 110, 110)])
    d = dg.duplicate_rate([frame_dup, frame_clean], iou_thresh=0.7)
    assert math.isclose(d["mean_dup_pairs_per_frame"], 0.5)
    assert d["num_frames"] == 2
    assert math.isclose(d["dup_box_fraction"], 0.5)


def test_duplicate_rate_empty_frames_safe():
    d = dg.duplicate_rate([np.empty((0, 6)), np.empty((0, 6))], iou_thresh=0.7)
    assert d["mean_dup_pairs_per_frame"] == 0.0
    assert d["dup_box_fraction"] == 0.0


def _track_rows(track_id, boxes):
    return [[f, track_id, x, y, w, h, 1.0, -1, -1, -1] for (f, x, y, w, h) in boxes]


def test_box_jitter_zero_for_static_box():
    rows = _track_rows(1, [(1, 0, 0, 10, 10), (2, 0, 0, 10, 10), (3, 0, 0, 10, 10)])
    j = dg.box_jitter(rows)
    assert math.isclose(j["center_jitter"], 0.0)
    assert math.isclose(j["size_jitter"], 0.0)


def test_box_jitter_zero_for_constant_velocity_motion():
    rows = _track_rows(1, [(1, 0, 0, 10, 10), (2, 10, 0, 10, 10), (3, 20, 0, 10, 10)])
    j = dg.box_jitter(rows)
    assert math.isclose(j["center_jitter"], 0.0, abs_tol=1e-9)
    assert math.isclose(j["size_jitter"], 0.0)


def test_box_jitter_positive_for_wobble():
    rows = _track_rows(1, [(1, 0, 0, 10, 10), (2, 10, 0, 10, 10), (3, 0, 0, 10, 10)])
    j = dg.box_jitter(rows)
    assert j["center_jitter"] > 0.0


def test_detection_pr_perfect_match():
    det = [np.array([_box(0, 0, 10, 10, conf=0.9)])]
    gt = [np.array([[0, 0, 10, 10]])]
    pr = dg.detection_pr(det, gt, iou_thresh=0.5)
    assert pr["tp"] == 1 and pr["fp"] == 0 and pr["fn"] == 0
    assert math.isclose(pr["precision"], 1.0) and math.isclose(pr["recall"], 1.0)


def test_detection_pr_false_positive_and_negative():
    det = [np.array([_box(100, 100, 110, 110, conf=0.9)])]
    gt = [np.array([[0, 0, 10, 10]])]
    pr = dg.detection_pr(det, gt, iou_thresh=0.5)
    assert pr["tp"] == 0 and pr["fp"] == 1 and pr["fn"] == 1
    assert math.isclose(pr["precision"], 0.0) and math.isclose(pr["recall"], 0.0)


def test_average_precision_perfect_is_one():
    det = [np.array([_box(0, 0, 10, 10, conf=0.9)]),
           np.array([_box(0, 0, 10, 10, conf=0.8)])]
    gt = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])]
    ap = dg.average_precision(det, gt, iou_thresh=0.5)
    assert math.isclose(ap, 1.0, rel_tol=1e-6)


def test_save_load_detections_round_trip(tmp_path):
    boxes = [np.array([_box(0, 0, 10, 10, 0.9), _box(5, 5, 15, 15, 0.8)]),
             np.empty((0, 6)),
             np.array([_box(1, 2, 3, 4, 0.7)])]
    p = tmp_path / "run01.npz"
    dg.save_detections(p, boxes)
    loaded = dg.load_detections(p)
    assert len(loaded) == 3
    assert loaded[1].shape[0] == 0
    assert np.allclose(loaded[0], boxes[0])
    assert np.allclose(loaded[2], boxes[2])


def test_sequence_diagnostics_combines_all():
    boxes = [np.array([_box(0, 0, 10, 10, 0.9)]), np.array([_box(0, 0, 10, 10, 0.9)])]
    gt = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])]
    track_rows = _track_rows(1, [(1, 0, 0, 10, 10), (2, 0, 0, 10, 10)])
    d = dg.sequence_diagnostics(boxes, track_rows, gt)
    for k in ("mean_dup_pairs_per_frame", "dup_box_fraction", "center_jitter",
              "size_jitter", "det_recall", "det_precision", "det_ap50"):
        assert k in d
    assert math.isclose(d["det_recall"], 1.0)
    assert math.isclose(d["det_ap50"], 1.0, rel_tol=1e-6)


def test_build_diagnostics_table_walks_tree(tmp_path):
    det_dir = tmp_path / "results" / "MOT17" / "MOT17-05" / "_detections"
    dg.save_detections(det_dir / "yolo11_n_nms.npz", [np.array([_box(0, 0, 10, 10, 0.9)])])
    dg.save_detections(det_dir / "rtdetr_l_nms.npz", [np.array([_box(0, 0, 10, 10, 0.9)])])
    gt = tmp_path / "prepared" / "MOT17" / "MOT17-05" / "gt"
    gt.mkdir(parents=True)
    (gt / "gt.txt").write_text("1,1,0,0,10,10,1,1,1\n", encoding="utf-8")

    df = dg.build_diagnostics_table(tmp_path / "results", tmp_path / "prepared")
    assert len(df) == 2
    row = df[df["model_family"] == "yolo11"].iloc[0]
    assert row["dataset"] == "MOT17" and row["sequence"] == "MOT17-05"
    assert row["model_scale"] == "n" and row["mode"] == "nms"
    assert math.isclose(row["det_recall"], 1.0)
    assert math.isclose(row["det_ap50"], 1.0, rel_tol=1e-6)


def test_build_jitter_table_walks_run_dirs(tmp_path):
    import pandas as pd
    run_dir = tmp_path / "results" / "MOT17" / "MOT17-05" / "yolo11_n_nms_bytetrack_0_run01"
    run_dir.mkdir(parents=True)
    pd.DataFrame([dict(dataset="MOT17", sequence="MOT17-05", model_family="yolo11",
                       model_scale="n", mode="nms", tracker="bytetrack", run_id=1)]
                 ).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "mot_results.txt").write_text(
        "1,1,0,0,10,10,1,-1,-1,-1\n2,1,10,0,10,10,1,-1,-1,-1\n3,1,0,0,10,10,1,-1,-1,-1\n",
        encoding="utf-8")
    df = dg.build_jitter_table(tmp_path / "results")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["tracker"] == "bytetrack" and row["model_family"] == "yolo11"
    assert row["dataset"] == "MOT17" and row["sequence"] == "MOT17-05"
    assert row["center_jitter"] > 0.0


def test_box_jitter_ignores_the_jump_across_a_track_gap():
    rows = _track_rows(1, [(1, 0, 0, 10, 10), (2, 0, 0, 10, 10), (3, 0, 0, 10, 10),
                           (9, 200, 50, 30, 30), (10, 200, 50, 30, 30), (11, 200, 50, 30, 30)])
    j = dg.box_jitter(rows)
    assert math.isclose(j["center_jitter"], 0.0) and math.isclose(j["size_jitter"], 0.0)
