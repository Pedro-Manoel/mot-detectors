from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_tracker(tracker_name: str, reid_weights: str | None,
                 device: str, frame_rate: int):
    from boxmot import ByteTrack, BotSort
    import torch

    raw = str(device).strip().lower()
    if raw in {"", "cpu"}:
        torch_device = torch.device("cpu")
    elif raw.isdigit():
        torch_device = torch.device(f"cuda:{raw}")
    else:
        torch_device = torch.device(raw)

    name = tracker_name.lower()

    if name == "bytetrack":
        return ByteTrack(frame_rate=frame_rate, per_class=False)
    elif name == "botsort":
        use_reid = bool(reid_weights)
        reid_path = Path(reid_weights) if reid_weights else Path("placeholder.pt")
        return BotSort(
            reid_weights=reid_path, device=torch_device, half=False,
            frame_rate=frame_rate, with_reid=use_reid, per_class=False,
        )
    else:
        raise ValueError(f"Unsupported tracker: {tracker_name}. Use: bytetrack, botsort")


def parse_track_outputs(frame_id: int, outputs: np.ndarray) -> list[list]:
    rows = []
    if outputs is None:
        return rows
    outputs = np.asarray(outputs)
    if outputs.ndim == 0 or outputs.size == 0:
        return rows
    if outputs.ndim == 1:
        outputs = outputs.reshape(1, -1)

    for row in outputs:
        if len(row) < 5:
            continue
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        track_id = int(float(row[4]))
        score = float(row[5]) if len(row) > 5 else 1.0
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        rows.append([frame_id, track_id, x1, y1, w, h, score, -1, -1, -1])
    return rows


def write_mot_txt(mot_rows: list[list], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(mot_rows)
