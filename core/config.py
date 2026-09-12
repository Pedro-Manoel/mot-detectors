from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .data import canonical_name


@dataclass
class DetectorSpec:
    family: str
    scales: list[str]
    modes: list[str]


@dataclass
class Config:
    name: str
    device: str
    runs: int
    conf: float
    imgsz: int
    half: bool
    trackeval_root: str
    trackeval_python: str
    do_preproc: bool
    keep_tracks: bool
    reid_weights: str
    detectors: list[DetectorSpec]
    trackers: list[str]
    sequences: dict[str, list[str]]
    output_root: str
    dataset_root: str
    prepared_root: str


@dataclass
class RunSpec:
    dataset: str
    sequence: str
    family: str
    scale: str
    mode: str
    tracker: str
    run_id: int
    model: str
    reid_weights: str | None


def is_valid_combination(family: str, mode: str) -> bool:
    return mode != "end2end" or family == "yolo26"


def weight_filename(family: str, scale: str) -> str:
    if family == "rtdetr":
        return f"rtdetr-{scale}.pt"
    return f"{family}{scale}.pt"


_TOP_KEYS = {"name", "device", "runs", "detector", "trackeval", "keep_tracks", "reid_weights",
             "detectors", "trackers", "sequences", "paths"}
_SECTION_KEYS = {"detector": {"conf", "imgsz", "half"},
                 "trackeval": {"root", "python", "do_preproc"},
                 "paths": {"output_root", "dataset_root", "prepared_root"}}
MODES = ("nms", "end2end")


def _validate(data: dict, path) -> None:
    problems = [f"unknown key '{k}'" for k in sorted(set(data) - _TOP_KEYS)]
    for section, keys in _SECTION_KEYS.items():
        problems += [f"unknown key '{section}.{k}'"
                     for k in sorted(set(data.get(section) or {}) - keys)]
    for d in data.get("detectors") or []:
        problems += [f"unknown key 'detectors[].{k}'"
                     for k in sorted(set(d) - {"family", "scales", "modes"})]
        problems += [f"mode '{m}' of {d.get('family')} is not one of {', '.join(MODES)}"
                     for m in d.get("modes", []) if m not in MODES]
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _validate(data, path)
    det = data["detector"]
    te = data["trackeval"]
    paths = data.get("paths", {})
    detectors = [DetectorSpec(d["family"], list(d["scales"]), list(d["modes"]))
                 for d in data["detectors"]]
    return Config(
        name=data["name"],
        device=str(data["device"]),
        runs=int(data["runs"]),
        conf=float(det["conf"]),
        imgsz=int(det["imgsz"]),
        half=bool(det.get("half", False)),
        trackeval_root=te["root"],
        trackeval_python=te["python"],
        do_preproc=bool(te.get("do_preproc", False)),
        keep_tracks=bool(data.get("keep_tracks", True)),
        reid_weights=data.get("reid_weights", ""),
        detectors=detectors,
        trackers=list(data["trackers"]),
        sequences={k: [canonical_name(k, s) for s in v] for k, v in data["sequences"].items()},
        output_root=paths.get("output_root", "results"),
        dataset_root=paths.get("dataset_root", "datasets"),
        prepared_root=paths.get("prepared_root", "prepared"),
    )


def expand_grid(cfg: Config) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for dataset, seqs in cfg.sequences.items():
        for sequence in seqs:
            for det in cfg.detectors:
                for scale in det.scales:
                    for mode in det.modes:
                        if not is_valid_combination(det.family, mode):
                            continue
                        for tracker in cfg.trackers:
                            for run_id in range(1, cfg.runs + 1):
                                reid = cfg.reid_weights if tracker == "botsort" else None
                                specs.append(RunSpec(
                                    dataset=dataset, sequence=sequence,
                                    family=det.family, scale=scale, mode=mode,
                                    tracker=tracker, run_id=run_id,
                                    model=weight_filename(det.family, scale),
                                    reid_weights=reid,
                                ))
    return specs
