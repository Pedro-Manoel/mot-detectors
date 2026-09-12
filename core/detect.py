from __future__ import annotations

import numpy as np

PERSON_CLASS_INDEX = 0
NMS_IOU = 0.7
_HAS_CUDA = False


def setup_gpu(device: str, half: bool) -> None:
    global _HAS_CUDA
    import torch

    _HAS_CUDA = torch.cuda.is_available()
    if _HAS_CUDA and device != "cpu":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)} ({mem_gb:.1f} GB)")
        print("  cudnn.benchmark: ON")
        print(f"  FP16 (half):     {'ON' if half else 'OFF'}")


def cuda_sync() -> None:
    if _HAS_CUDA:
        import torch
        torch.cuda.synchronize()


def load_detector(model_name: str, mode: str):
    from ultralytics import RTDETR, YOLO

    print(f"Loading {model_name}")
    model = RTDETR(model_name) if "rtdetr" in model_name.lower() else YOLO(model_name)
    inner = getattr(model, "model", None)
    if "yolo26" in model_name.lower() and inner is not None and hasattr(inner, "end2end"):
        inner.end2end = mode != "nms"
        print(f"  YOLO26 head set before fuse: {describe_active_head(model)}")
    try:
        model.fuse()
    except Exception as e:
        print(f"  Warning: fuse() skipped: {e}")
    return model


def predict_frame(model, frame_bgr: np.ndarray, conf: float,
                  use_nms: bool, half: bool, imgsz: int) -> list:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "end2end"):
        inner.end2end = not use_nms
    return model.predict(source=frame_bgr, conf=conf, iou=NMS_IOU, classes=[PERSON_CLASS_INDEX],
                         verbose=False, augment=False, stream=False, half=half, imgsz=imgsz)


def get_active_end2end_flag(model) -> bool | None:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "end2end"):
        return bool(inner.end2end)
    return None


def describe_active_head(model) -> str:
    active_end2end = get_active_end2end_flag(model)
    if active_end2end is None:
        return "NMS/default (model without end2end switch)"
    return "End-to-End (one-to-one)" if active_end2end else "NMS (one-to-many)"


def extract_ultralytics_speed(results) -> dict[str, float | None]:
    speed = getattr(results[0], "speed", None) or {}
    return {"pre": speed.get("preprocess"), "inf": speed.get("inference"),
            "post": speed.get("postprocess")}


def parse_ultralytics_results(results, conf_threshold: float) -> np.ndarray:
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    mask = (classes == PERSON_CLASS_INDEX) & (scores >= conf_threshold)
    if not mask.any():
        return np.empty((0, 6), dtype=np.float32)

    return np.column_stack([xyxy[mask], scores[mask], classes[mask]]).astype(np.float32)
