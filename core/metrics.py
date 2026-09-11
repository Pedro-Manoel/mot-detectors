from __future__ import annotations

from statistics import mean, stdev


def block(per_frame_ms: list[float]):
    if not per_frame_ms:
        return (None, None, None)
    avg = mean(per_frame_ms)
    std = stdev(per_frame_ms) if len(per_frame_ms) >= 2 else None
    fps = (1000.0 / avg) if avg and avg > 0 else None
    return (avg, std, fps)


def reconstruct_total(imread: list[float], det: list[float], track: list[float]) -> list[float]:
    return [a + b + c for a, b, c in zip(imread, det, track)]


def run_timing_fields(imread, det, track, pre, inf, post) -> dict:
    total = reconstruct_total(imread, det, track)
    fields: dict = {}
    for name, series in (("det", det), ("track", track), ("total", total),
                         ("preprocess", pre), ("inference", inf), ("postprocess", post)):
        avg, std, fps = block(series)
        fields[f"avg_{name}_ms"] = avg
        fields[f"std_{name}_ms"] = std
        fields[f"fps_{name}"] = fps
    fields["frame1_det_ms"] = det[0] if det else None
    fields["frame1_total_ms"] = total[0] if total else None
    fields["frame1_postprocess_ms"] = post[0] if post else None
    return fields
