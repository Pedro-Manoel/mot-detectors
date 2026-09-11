from __future__ import annotations

from pathlib import Path

import numpy as np


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = a[:, :4]
    b = b[:, :4]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def duplicate_rate(per_frame_boxes, iou_thresh: float = 0.7) -> dict:
    total_pairs = 0
    dup_boxes = 0
    total_boxes = 0
    n_frames = 0
    for boxes in per_frame_boxes:
        b = np.asarray(boxes, dtype=float)
        n_frames += 1
        n = len(b)
        total_boxes += n
        if n < 2:
            continue
        iou = iou_matrix(b, b)
        np.fill_diagonal(iou, 0.0)
        pair_mask = iou > iou_thresh
        total_pairs += int(np.triu(pair_mask, k=1).sum())
        dup_boxes += int(np.any(pair_mask, axis=1).sum())
    return {
        "mean_dup_pairs_per_frame": (total_pairs / n_frames) if n_frames else 0.0,
        "dup_box_fraction": (dup_boxes / total_boxes) if total_boxes else 0.0,
        "num_frames": n_frames,
    }


def _tracks_by_id(track_rows):
    tracks: dict[int, list] = {}
    for r in track_rows:
        frame, tid, x, y, w, h = r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
        tracks.setdefault(tid, []).append((int(frame), x, y, w, h))
    for tid in tracks:
        tracks[tid].sort(key=lambda t: t[0])
    return tracks


def _segments(track):
    # Split a track where frames are missing (lost and found again): the box change across the
    # gap is not frame-to-frame jitter.
    out, cur = [], [track[0]]
    for prev, row in zip(track, track[1:]):
        if row[0] - prev[0] == 1:
            cur.append(row)
        else:
            out.append(cur)
            cur = [row]
    out.append(cur)
    return out


def box_jitter(track_rows) -> dict:
    tracks = _tracks_by_id(track_rows)
    center_vals, size_vals = [], []
    for seq in (s for track in tracks.values() for s in _segments(track)):
        if len(seq) < 2:
            continue
        cx = np.array([x + w / 2.0 for (_, x, y, w, h) in seq])
        cy = np.array([y + h / 2.0 for (_, x, y, w, h) in seq])
        ws = np.array([w for (_, x, y, w, h) in seq], dtype=float)
        hs = np.array([h for (_, x, y, w, h) in seq], dtype=float)
        diag = np.sqrt(ws ** 2 + hs ** 2)
        mean_diag = float(np.mean(diag)) or 1.0
        if len(seq) >= 3:
            ax = cx[2:] - 2 * cx[1:-1] + cx[:-2]
            ay = cy[2:] - 2 * cy[1:-1] + cy[:-2]
            accel = np.sqrt(ax ** 2 + ay ** 2)
            center_vals.append(float(np.mean(accel)) / mean_diag)
        dw = np.abs(np.diff(ws)) / np.clip(ws[:-1], 1e-9, None)
        dh = np.abs(np.diff(hs)) / np.clip(hs[:-1], 1e-9, None)
        size_vals.append(float(np.mean((dw + dh) / 2.0)))
    return {
        "center_jitter": float(np.mean(center_vals)) if center_vals else 0.0,
        "size_jitter": float(np.mean(size_vals)) if size_vals else 0.0,
        "num_tracks": len(tracks),
    }


def _greedy_match(det_boxes, gt_boxes, iou_thresh):
    n_gt = len(gt_boxes)
    if len(det_boxes) == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, len(det_boxes), 0
    iou = iou_matrix(det_boxes, gt_boxes)
    gt_taken = np.zeros(n_gt, dtype=bool)
    tp = 0
    for di in range(len(det_boxes)):
        order = np.argsort(-iou[di])
        matched = False
        for gj in order:
            if iou[di, gj] < iou_thresh:
                break
            if not gt_taken[gj]:
                gt_taken[gj] = True
                tp += 1
                matched = True
                break
    fp = len(det_boxes) - tp
    fn = n_gt - int(gt_taken.sum())
    return tp, fp, fn


def detection_pr(per_frame_det, per_frame_gt, iou_thresh: float = 0.5) -> dict:
    TP = FP = FN = 0
    for det, gt in zip(per_frame_det, per_frame_gt):
        d = np.asarray(det, dtype=float)
        g = np.asarray(gt, dtype=float)
        if len(d) > 1:
            d = d[np.argsort(-d[:, 4])] if d.shape[1] > 4 else d
        tp, fp, fn = _greedy_match(d, g, iou_thresh)
        TP += tp; FP += fp; FN += fn
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    return {"tp": TP, "fp": FP, "fn": FN, "precision": precision, "recall": recall}


def average_precision(per_frame_det, per_frame_gt, iou_thresh: float = 0.5) -> float:
    records = []
    total_gt = 0
    for det, gt in zip(per_frame_det, per_frame_gt):
        d = np.asarray(det, dtype=float)
        g = np.asarray(gt, dtype=float)
        total_gt += len(g)
        if len(d) == 0:
            continue
        order = np.argsort(-d[:, 4]) if d.shape[1] > 4 else np.arange(len(d))
        gt_taken = np.zeros(len(g), dtype=bool)
        iou = iou_matrix(d, g) if len(g) else np.zeros((len(d), 0))
        for di in order:
            conf = float(d[di, 4]) if d.shape[1] > 4 else 1.0
            is_tp = 0
            if len(g):
                gj = int(np.argmax(iou[di])) if iou.shape[1] else -1
                if gj >= 0 and iou[di, gj] >= iou_thresh and not gt_taken[gj]:
                    gt_taken[gj] = True
                    is_tp = 1
            records.append((conf, is_tp))
    if total_gt == 0 or not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp_cum = np.cumsum([r[1] for r in records])
    fp_cum = np.cumsum([1 - r[1] for r in records])
    recall = tp_cum / total_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def save_detections(path, per_frame_boxes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, boxes in enumerate(per_frame_boxes):
        b = np.asarray(boxes, dtype=np.float32)
        for r in b:
            r = list(r) + [0.0] * (6 - len(r))
            rows.append([i, r[0], r[1], r[2], r[3], r[4], r[5]])
    arr = np.array(rows, dtype=np.float32) if rows else np.empty((0, 7), np.float32)
    np.savez_compressed(path, dets=arr, n_frames=np.int64(len(per_frame_boxes)))
    return path


def load_detections(path) -> list[np.ndarray]:
    data = np.load(path)
    arr, n_frames = data["dets"], int(data["n_frames"])
    out = [np.empty((0, 6), np.float32) for _ in range(n_frames)]
    if len(arr):
        for i in range(n_frames):
            sel = arr[arr[:, 0] == i]
            if len(sel):
                out[i] = sel[:, 1:7].astype(np.float32)
    return out


def sequence_diagnostics(per_frame_boxes, track_rows, per_frame_gt,
                         dup_iou: float = 0.7, ap_iou: float = 0.5) -> dict:
    dup = duplicate_rate(per_frame_boxes, dup_iou)
    jit = box_jitter(track_rows)
    pr = detection_pr(per_frame_boxes, per_frame_gt, ap_iou)
    ap = average_precision(per_frame_boxes, per_frame_gt, ap_iou)
    return {
        "mean_dup_pairs_per_frame": dup["mean_dup_pairs_per_frame"],
        "dup_box_fraction": dup["dup_box_fraction"],
        "center_jitter": jit["center_jitter"],
        "size_jitter": jit["size_jitter"],
        "num_tracks": jit["num_tracks"],
        "det_recall": pr["recall"],
        "det_precision": pr["precision"],
        "det_ap50": ap,
    }


def _gt_per_frame(prepared_root, dataset, sequence, n_frames) -> list[np.ndarray]:
    p = Path(prepared_root) / dataset / sequence / "gt" / "gt.txt"
    per: list[list] = [[] for _ in range(n_frames)]
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            c = line.split(",")
            f = int(float(c[0]))
            x, y, w, h = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            if 1 <= f <= n_frames:
                per[f - 1].append([x, y, x + w, y + h])
    return [np.array(b, dtype=float) if b else np.empty((0, 4)) for b in per]


def _read_mot_rows(path) -> list[list]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        c = line.split(",")
        rows.append([float(c[0]), float(c[1]), float(c[2]),
                     float(c[3]), float(c[4]), float(c[5])])
    return rows


def build_jitter_table(results_dir):
    import pandas as pd

    results_dir = Path(results_dir)
    recs = []
    for mot in sorted(results_dir.glob("**/mot_results.txt")):
        metrics = mot.parent / "metrics.csv"
        if not metrics.exists():
            continue
        m = pd.read_csv(metrics).iloc[0]
        jit = box_jitter(_read_mot_rows(mot))
        recs.append(dict(
            dataset=m["dataset"], sequence=m["sequence"], model_family=m["model_family"],
            model_scale=m["model_scale"], mode=m["mode"], tracker=m["tracker"],
            run_id=int(m["run_id"]), center_jitter=jit["center_jitter"],
            size_jitter=jit["size_jitter"], num_tracks=jit["num_tracks"]))
    df = pd.DataFrame.from_records(recs)
    if df.empty:
        return df
    keys = ["dataset", "sequence", "model_family", "model_scale", "mode", "tracker"]
    return df.groupby(keys, as_index=False)[
        ["center_jitter", "size_jitter", "num_tracks"]].mean()


def build_diagnostics_table(results_dir, prepared_root,
                            dup_iou: float = 0.7, ap_iou: float = 0.5):
    import pandas as pd

    results_dir = Path(results_dir)
    recs = []
    for npz in sorted(results_dir.glob("*/*/_detections/*.npz")):
        dataset, sequence = npz.parts[-4], npz.parts[-3]
        bits = npz.stem.split("_")
        if len(bits) != 3:
            continue
        family, scale, mode = bits
        boxes = load_detections(npz)
        gt = _gt_per_frame(prepared_root, dataset, sequence, len(boxes))
        dup = duplicate_rate(boxes, dup_iou)
        pr = detection_pr(boxes, gt, ap_iou)
        ap = average_precision(boxes, gt, ap_iou)
        recs.append(dict(
            dataset=dataset, sequence=sequence, model_family=family,
            model_scale=scale, mode=mode, num_frames=len(boxes),
            mean_dup_pairs_per_frame=dup["mean_dup_pairs_per_frame"],
            dup_box_fraction=dup["dup_box_fraction"],
            det_recall=pr["recall"], det_precision=pr["precision"], det_ap50=ap))
    return pd.DataFrame.from_records(recs)
