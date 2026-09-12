from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

TRACKEVAL_WANTED_METRICS = [
    "HOTA", "DetA", "AssA", "LocA",
    "MOTA", "MOTP", "IDSW", "Frag",
    "CLR_FP", "CLR_FN", "FP", "FN",
    "MT", "PT", "ML", "MTR", "MLR",
    "mostly_tracked_ratio", "mostly_lost_ratio",
    "GT_IDs", "num_unique_objects",
    "CLR_Re", "CLR_Pr", "Recall", "Precision",
    "IDF1", "IDR", "IDP",
]
_ALIASES = {"CLR_Re": "Recall", "CLR_Pr": "Precision", "CLR_FP": "FP", "CLR_FN": "FN",
            "GT_IDs": "num_unique_objects", "MTR": "mostly_tracked_ratio",
            "MLR": "mostly_lost_ratio"}


def resolve_python(path: str) -> str:
    p = Path(path)
    candidates = [p]
    if len(p.parts) >= 2 and p.parts[-2].lower() == "scripts":
        candidates.append(p.parent.parent / "bin" / "python")
    elif len(p.parts) >= 2 and p.parts[-2] == "bin":
        candidates.append(p.parent.parent / "Scripts" / "python.exe")
    for c in candidates:
        if c.exists():
            return str(c.absolute())
    return path


def stage_trackeval_workspace(prepared_root: Path, dataset: str, sequence: str,
                              tracker_name: str, mot_results_path: Path,
                              workspace_root: Path) -> dict[str, Path]:
    gt_root = workspace_root / "gt" / "mot_challenge"
    trackers_root = workspace_root / "trackers" / "mot_challenge"
    seq_gt_dir = gt_root / f"{dataset}-train" / sequence / "gt"
    seqmaps_dir = gt_root / "seqmaps"
    tracker_data_dir = trackers_root / f"{dataset}-train" / tracker_name / "data"
    for d in (seq_gt_dir, seqmaps_dir, tracker_data_dir):
        d.mkdir(parents=True, exist_ok=True)

    seq_dir = prepared_root / dataset / sequence
    shutil.copy2(seq_dir / "gt" / "gt.txt", seq_gt_dir / "gt.txt")
    shutil.copy2(seq_dir / "seqinfo.ini", seq_gt_dir.parent / "seqinfo.ini")
    (seqmaps_dir / f"{dataset}-train.txt").write_text(f"name\n{sequence}\n", encoding="utf-8",
                                                      newline="\n")
    shutil.copy2(mot_results_path, tracker_data_dir / f"{sequence}.txt")
    return {"gt_root": gt_root, "trackers_root": trackers_root}


def call_trackeval(trackeval_root: Path, trackeval_python: str, dataset: str, tracker_name: str,
                   workspace: dict[str, Path], do_preproc: bool = False) -> dict:
    script = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script.exists():
        raise FileNotFoundError(f"TrackEval script not found: {script}")

    cmd = [
        trackeval_python, str(script),
        "--BENCHMARK", dataset, "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", tracker_name,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False", "--NUM_PARALLEL_CORES", "1",
        "--DO_PREPROC", "True" if do_preproc else "False",
        "--GT_FOLDER", str(workspace["gt_root"]),
        "--TRACKERS_FOLDER", str(workspace["trackers_root"]),
    ]
    print("\n[TrackEval] Running evaluation...")
    print(f"  Command: {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, cwd=str(trackeval_root), capture_output=True, text=True,
                            check=False)

    for line in result.stdout.splitlines():
        line = line.strip()
        if "Config" not in line and any(kw in line for kw in (
                "eval_sequence", "Evaluating", "finished in", "COMBINED", "pedestrian")):
            print(f"  {line}")

    if result.returncode != 0:
        for line in result.stderr.strip().splitlines()[-5:]:
            print(f"  [TrackEval ERR] {line.strip()}")
        raise RuntimeError(
            f"TrackEval exited with code {result.returncode}.\n"
            f"Common causes:\n"
            f"  - Wrong trackeval.python in the YAML\n"
            f"  - TrackEval not installed in that venv\n"
            f"  - Empty tracking results (no detections)"
        )

    return parse_trackeval_summary(workspace["trackers_root"] / f"{dataset}-train" / tracker_name)


def parse_trackeval_summary(tracker_dir: Path) -> dict:
    path = Path(tracker_dir) / "pedestrian_summary.txt"
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Empty TrackEval summary: {path}")
    summary: dict = {}
    for key, val in zip(re.split(r"[\s,]+", lines[0]), re.split(r"[\s,]+", lines[1])):
        try:
            f = float(val)
            summary[key] = int(f) if f == int(f) else f
        except ValueError:
            summary[key] = val
    for src, dst in _ALIASES.items():
        if src in summary and dst not in summary:
            summary[dst] = summary[src]
    return summary
