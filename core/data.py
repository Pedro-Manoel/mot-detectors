from __future__ import annotations

import configparser
import json
import platform
import shutil
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
    registry = json.loads(p.read_text(encoding="utf-8"))
    return {seq: float(d) for seqs in registry.values() for seq, d in seqs.items()}


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
    for det in MOT17_DETECTOR_PREFERENCE:
        if name.upper().endswith(f"-{det}"):
            return name[: -len(det) - 1]
    return name


def canonical_name(dataset: str, raw_name: str) -> str:
    if dataset.upper() == "MOT17":
        return strip_mot17_suffix(raw_name)
    return raw_name


def read_seqinfo(path: Path) -> SeqInfo:
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if not parser.sections():
        raise ValueError(f"No sections found in seqinfo.ini: {path}")
    section = parser[parser.sections()[0]]
    return SeqInfo(
        name=section.get("name", path.parent.name),
        frame_rate=section.getint("framerate", 25),
        seq_length=section.getint("seqlength", 0),
        width=section.getint("imwidth", 0),
        height=section.getint("imheight", 0),
        im_dir=section.get("imdir", "img1"),
        im_ext=section.get("imext", ".jpg"),
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
    base = dataset_root / dataset / "train"
    names = [sequence]
    if dataset.upper() == "MOT17":
        bare = strip_mot17_suffix(sequence)
        names += [f"{bare}-{det}" for det in MOT17_DETECTOR_PREFERENCE]
    for name in names:
        if (base / name / "seqinfo.ini").exists():
            return base / name
    raise FileNotFoundError(
        f"Sequence folder not found for {dataset}/{sequence} (a folder with seqinfo.ini).\n"
        f"Searched: {', '.join(str(base / n) for n in names)}\n"
        f"Download the train split into {dataset_root} as described in README.md (Datasets)."
    )


def write_seqinfo(info: SeqInfo, path: Path) -> None:
    parser = configparser.RawConfigParser()
    parser.optionxform = str
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

    computed = round(len(ped_gt) / seqinfo.seq_length, 2) if seqinfo.seq_length > 0 else 0.0
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
    img_dir = seq_dir / "img1"
    if not img_dir.exists():
        raise FileNotFoundError(f"Prepared sequence not found: {seq_dir}\n"
                                f"Run `mot.py prepare` first.")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not frames:
        raise RuntimeError(f"No image files found in {img_dir}")
    meta_path = seq_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return frames, meta
