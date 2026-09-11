from __future__ import annotations

import numpy as np

PERSON_CLASS_INDEX = 0
NMS_IOU = 0.7
_HAS_CUDA = False


def setup_gpu(device: str, half: bool) -> None:
    global _HAS_CUDA
    try:
        import torch
        _HAS_CUDA = torch.cuda.is_available()
        if _HAS_CUDA and device != "cpu":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  GPU: {gpu_name} ({mem_gb:.1f} GB)")
            print(f"  cudnn.benchmark: ON")
            print(f"  FP16 (half):     {'ON' if half else 'OFF'}")
    except ImportError:
        pass


def cuda_sync() -> None:
    if _HAS_CUDA:
        import torch
        torch.cuda.synchronize()


def patch_numpy_compat() -> None:
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]


def load_detector(model_name: str, device: str, half: bool, mode: str):
    from ultralytics import YOLO, RTDETR

    if "rtdetr" in model_name.lower():
        print(f"Loading RT-DETR: {model_name}")
        model = RTDETR(model_name)
    else:
        print(f"Loading YOLO: {model_name}")
        model = YOLO(model_name)

    inner_model = getattr(model, "model", None)
    if "yolo26" in model_name.lower() and inner_model is not None and hasattr(inner_model, "end2end"):
        inner_model.end2end = (mode != "nms")
        actual_mode = "NMS (one-to-many)" if mode == "nms" else "End-to-End (one-to-one)"
        print(f"  YOLO26 head mode set before fuse: {actual_mode}")

    try:
        model.fuse()
    except Exception as e:
        print(f"  Warning: fuse() skipped: {e}")

    return model


def model_complexity(model_name: str, imgsz: int) -> tuple[float, float]:
    # Parameters (millions) and GFLOPs after layer fusion, as Ultralytics reports them; for
    # YOLO11 and YOLO26 these are the figures Ultralytics publishes.
    from ultralytics import YOLO, RTDETR

    model = RTDETR(model_name) if "rtdetr" in model_name.lower() else YOLO(model_name)
    model.fuse()
    _, n_params, _, gflops = model.model.info(verbose=True, imgsz=imgsz)
    return n_params / 1e6, gflops


def predict_frame(model, frame_bgr: np.ndarray, conf: float,
                  use_nms: bool, half: bool, imgsz: int) -> list:
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "end2end"):
        inner_model.end2end = (not use_nms)

    kwargs = dict(
        source=frame_bgr,
        conf=conf,
        iou=NMS_IOU,
        classes=[PERSON_CLASS_INDEX],
        verbose=False,
        augment=False,
        stream=False,
        half=half,
        imgsz=imgsz,
    )

    return model.predict(**kwargs)


def get_active_end2end_flag(model) -> bool | None:
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "end2end"):
        return bool(inner_model.end2end)
    return None


def describe_active_head(model) -> str:
    active_end2end = get_active_end2end_flag(model)
    if active_end2end is None:
        return "NMS/default (model without end2end switch)"
    return "End-to-End (one-to-one)" if active_end2end else "NMS (one-to-many)"


def extract_ultralytics_speed(results) -> dict[str, float | None]:
    result = results[0] if isinstance(results, list) else results
    speed = getattr(result, "speed", None) or {}
    return {
        "ultra_pre_ms": speed.get("preprocess"),
        "ultra_inf_ms": speed.get("inference"),
        "ultra_post_ms": speed.get("postprocess"),
    }


def _to_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def parse_ultralytics_results(results, conf_threshold: float) -> np.ndarray:
    result = results[0] if isinstance(results, list) else results
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.empty((0, 6), dtype=np.float32)

    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    mask = (classes == PERSON_CLASS_INDEX) & (scores >= conf_threshold)
    if not mask.any():
        return np.empty((0, 6), dtype=np.float32)

    return np.column_stack([xyxy[mask], scores[mask], classes[mask]]).astype(np.float32)
