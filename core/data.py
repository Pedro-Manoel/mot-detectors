from __future__ import annotations

import configparser
import json
import platform
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PEDESTRIAN_CLASS_ID = 1
MOT17_DETECTOR_PREFERENCE = ["FRCNN", "DPM", "SDP"]
IS_WINDOWS = platform.system() == "Windows"

DENSITY_REGISTRY = Path(__file__).resolve().parent.parent / "densities.json"


def load_density_registry(path=DENSITY_REGISTRY) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, float] = {}

    def _add(seq, val) -> None:
        if val is None or (isinstance(val, str) and not val.strip()):
            return
        out[str(seq)] = float(val)

    for key, val in raw.items():
        if isinstance(val, dict):
            for s, d in val.items():
                _add(s, d)
        else:
            _add(key, val)
    return out


def resolve_density(sequence: str, computed: float, registry: dict) -> tuple[float, bool]:
    if sequence in registry:
        return float(registry[sequence]), True
    return float(computed), False


@dataclass
class SeqInfo:
    name: str
    frame_rate: int
    seq_length: int
    width: int
    height: int
    im_dir: str
    im_ext: str


@dataclass
class GTRow:
    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: int
    class_id: int
    visibility: float


def strip_mot17_suffix(name: str) -> str:
    for suffix in ["-FRCNN", "-DPM", "-SDP"]:
        if name.upper().endswith(suffix):
            return name[: -len(suffix)]
    return name


def canonical_name(dataset: str, raw_name: str) -> str:
    if dataset.upper() == "MOT17":
        return strip_mot17_suffix(raw_name)
    return raw_name


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.optionxform = str.lower  # type: ignore[assignment]
    parser.read(path, encoding="utf-8")

    section = None
    for sec_name in ["sequence", "Sequence"]:
        if parser.has_section(sec_name):
            section = parser[sec_name]
            break
    if section is None:
        sections = parser.sections()
        if not sections:
            raise ValueError(f"No sections found in seqinfo.ini: {path}")
        section = parser[sections[0]]

    def get_int(key: str, fb: int = 0) -> int:
        for k in [key, key.lower(), key.upper()]:
            if k in section:
                return int(section[k])
        return fb

    def get_str(key: str, fb: str = "") -> str:
        for k in [key, key.lower(), key.upper()]:
            if k in section:
                return section[k]
        return fb

    return SeqInfo(
        name=get_str("name", path.parent.name),
        frame_rate=get_int("framerate", 25),
        seq_length=get_int("seqlength"),
        width=get_int("imwidth"),
        height=get_int("imheight"),
        im_dir=get_str("imdir", "img1"),
        im_ext=get_str("imext", ".jpg"),
    )


def read_gt(gt_path: Path) -> list[GTRow]:
    rows: list[GTRow] = []
    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            rows.append(GTRow(
                frame=int(float(parts[0])),
                track_id=int(float(parts[1])),
                left=float(parts[2]),
                top=float(parts[3]),
                width=float(parts[4]),
                height=float(parts[5]),
                conf=int(float(parts[6])),
                class_id=int(float(parts[7])),
                visibility=float(parts[8]),
            ))
    return rows


def resolve_sequence_path(dataset_root: Path, dataset: str, sequence: str) -> Path:
    ds = dataset.upper()

    if ds == "SOMPT22":
        bases = [
            dataset_root / "SOMPT22" / "train",
            dataset_root / "SOMPT22",
            dataset_root / "SOMPT22" / "SOMPT22",
            dataset_root,
        ]
    else:
        bases = [dataset_root / dataset / "train", dataset_root / dataset]

    for base in bases:
        if not base.is_dir():
            continue
        exact = base / sequence
        if exact.is_dir() and (exact / "seqinfo.ini").exists():
            return exact
        if ds == "MOT17":
            bare = strip_mot17_suffix(sequence)
            for det in MOT17_DETECTOR_PREFERENCE:
                candidate = base / f"{bare}-{det}"
                if candidate.is_dir() and (candidate / "seqinfo.ini").exists():
                    return candidate
        if ds == "SOMPT22":
            bare_num = sequence.replace("SOMPT22-", "").replace("SOMPT22_", "")
            for alt in [bare_num, f"seq{bare_num}", f"Seq{bare_num}"]:
                candidate = base / alt
                if candidate.is_dir() and (candidate / "seqinfo.ini").exists():
                    return candidate

    searched = "\n  ".join(str(b / sequence) for b in bases)
    raise FileNotFoundError(
        f"Sequence folder not found for {dataset}/{sequence} (a folder with seqinfo.ini).\n"
        f"Searched:\n  {searched}\n"
        f"Download the train split into {dataset_root} as described in README.md (Datasets)."
    )


def write_seqinfo(info: SeqInfo, path: Path) -> None:
    parser = configparser.RawConfigParser()
    parser.optionxform = str  # type: ignore[assignment]
    parser["Sequence"] = {
        "name": info.name, "imDir": info.im_dir,
        "frameRate": str(info.frame_rate), "seqLength": str(info.seq_length),
        "imWidth": str(info.width), "imHeight": str(info.height),
        "imExt": info.im_ext,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        parser.write(f)


def copy_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if IS_WINDOWS:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)


def prepare_sequence(dataset_root: Path, dataset: str, sequence: str, prepared_root: Path) -> Path:
    raw_seq_path = resolve_sequence_path(dataset_root, dataset, sequence)
    seq_name = canonical_name(dataset, sequence)

    seqinfo = read_seqinfo(raw_seq_path / "seqinfo.ini")
    seqinfo.name = seq_name

    all_gt = read_gt(raw_seq_path / "gt" / "gt.txt")
    ped_gt = [r for r in all_gt if r.conf == 1 and r.class_id == PEDESTRIAN_CLASS_ID]

    out_dir = prepared_root / dataset / seq_name
    img_dir = out_dir / "img1"
    gt_dir = out_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    src_img_dir = raw_seq_path / seqinfo.im_dir
    for i in range(1, seqinfo.seq_length + 1):
        fname = f"{i:06d}{seqinfo.im_ext}"
        src = src_img_dir / fname
        if not src.exists():
            raise FileNotFoundError(f"Frame not found: {src}")
        copy_frame(src, img_dir / fname)

    gt_lines = [
        f"{r.frame},{r.track_id},{r.left:.3f},{r.top:.3f},"
        f"{r.width:.3f},{r.height:.3f},1,1,{r.visibility:.6f}"
        for r in ped_gt
    ]
    (gt_dir / "gt.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    write_seqinfo(seqinfo, out_dir / "seqinfo.ini")

    frame_counts = Counter(r.frame for r in ped_gt)
    avg_density = sum(frame_counts.values()) / seqinfo.seq_length if seqinfo.seq_length > 0 else 0.0
    computed = round(avg_density, 2)
    density, registered = resolve_density(seq_name, computed, load_density_registry())
    if not registered:
        print(f"[density] WARNING: '{seq_name}' is not registered in "
              f"{DENSITY_REGISTRY.name}; using GT-computed density {computed:.2f}. "
              f"Add it there for the dataset's published value.")

    meta = {
        "dataset": dataset, "sequence": seq_name,
        "raw_sequence": raw_seq_path.name,
        "raw_sequence_path": str(raw_seq_path.resolve()),
        "seq_length": seqinfo.seq_length, "frame_rate": seqinfo.frame_rate,
        "width": seqinfo.width, "height": seqinfo.height,
        "im_ext": seqinfo.im_ext,
        "num_gt_rows_original": len(all_gt),
        "num_gt_rows_pedestrian": len(ped_gt),
        "unique_track_ids": len({r.track_id for r in ped_gt}),
        "density_avg_computed": computed,
        "density_avg": density,
        "density_registered": registered,
        "prepared_dir": str(out_dir.resolve()),
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def load_sequence(prepared_root: Path, dataset: str, sequence: str) -> tuple[list[Path], dict]:
    seq_dir = prepared_root / dataset / sequence
    if not seq_dir.exists():
        bare = strip_mot17_suffix(sequence)
        seq_dir = prepared_root / dataset / bare

    if not seq_dir.exists():
        raise FileNotFoundError(
            f"Prepared sequence not found: {prepared_root / dataset / sequence}\n"
            f"Run `mot.py prepare` first."
        )

    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Frame directory not found: {img_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not frames:
        raise RuntimeError(f"No image files found in {img_dir}")

    meta_path = seq_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    return frames, meta
