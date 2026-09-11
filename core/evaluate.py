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


def resolve_python(path: str) -> str:
    # A venv interpreter lives in Scripts/python.exe on Windows and in bin/python elsewhere.
    # Accept either spelling so one YAML works on both; a bare command such as "python" is
    # returned unchanged.
    p = Path(path)
    candidates = [p]
    if len(p.parts) >= 2 and p.parts[-2].lower() == "scripts":
        candidates.append(p.parent.parent / "bin" / "python")
    elif len(p.parts) >= 2 and p.parts[-2] == "bin":
        candidates.append(p.parent.parent / "Scripts" / "python.exe")
    for c in candidates:
        if c.exists():
            # absolute(), not resolve(): a Linux venv's bin/python is a symlink to the base
            # interpreter, and following it would run TrackEval outside its environment.
            return str(c.absolute())
    return path


def stage_trackeval_workspace(
    prepared_root: Path, dataset: str, sequence: str,
    tracker_name: str, mot_results_path: Path, workspace_root: Path,
) -> dict[str, Path]:
    gt_root = workspace_root / "gt" / "mot_challenge"
    trackers_root = workspace_root / "trackers" / "mot_challenge"
    output_root = workspace_root / "output"

    seq_gt_dir = gt_root / f"{dataset}-train" / sequence / "gt"
    seqmaps_dir = gt_root / "seqmaps"
    tracker_data_dir = trackers_root / f"{dataset}-train" / tracker_name / "data"

    for d in [seq_gt_dir, seqmaps_dir, tracker_data_dir, output_root]:
        d.mkdir(parents=True, exist_ok=True)

    seq_dir = prepared_root / dataset / sequence
    src_gt = seq_dir / "gt" / "gt.txt"
    src_seqinfo = seq_dir / "seqinfo.ini"

    if not src_gt.exists():
        raise FileNotFoundError(f"GT not found: {src_gt}")
    if not src_seqinfo.exists():
        raise FileNotFoundError(f"seqinfo.ini not found: {src_seqinfo}")

    shutil.copy2(src_gt, seq_gt_dir / "gt.txt")
    shutil.copy2(src_seqinfo, gt_root / f"{dataset}-train" / sequence / "seqinfo.ini")

    seqmap = seqmaps_dir / f"{dataset}-train.txt"
    with seqmap.open("w", encoding="utf-8", newline="\n") as f:
        f.write("name\n")
        f.write(f"{sequence}\n")

    shutil.copy2(mot_results_path, tracker_data_dir / f"{sequence}.txt")

    return {"gt_root": gt_root, "trackers_root": trackers_root, "output_root": output_root}


def call_trackeval(
    trackeval_root: Path, trackeval_python: str,
    dataset: str, tracker_name: str, workspace: dict[str, Path],
    do_preproc: bool = False,
) -> dict:
    script = trackeval_root / "scripts" / "run_mot_challenge.py"
    if not script.exists():
        alt = trackeval_root / "TrackEval" / "scripts" / "run_mot_challenge.py"
        if alt.exists():
            script = alt
            trackeval_root = alt.parents[1]
        else:
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

    print(f"\n[TrackEval] Running evaluation...")
    print(f"  Command: {' '.join(cmd[:6])} ...")

    result = subprocess.run(
        cmd, cwd=str(trackeval_root),
        capture_output=True, text=True, check=False,
    )

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            stripped = line.strip()
            if any(kw in stripped for kw in [
                "eval_sequence", "Evaluating", "finished in", "COMBINED", "pedestrian",
            ]) and "Config" not in stripped:
                print(f"  {stripped}")

    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"  [TrackEval ERR] {line.strip()}")
        raise RuntimeError(
            f"TrackEval exited with code {result.returncode}.\n"
            f"Common causes:\n"
            f"  - Wrong trackeval.python in the YAML\n"
            f"  - TrackEval not installed in that venv\n"
            f"  - Empty tracking results (no detections)"
        )

    trackers_dir = workspace["trackers_root"] / f"{dataset}-train" / tracker_name
    output_dir = workspace["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_files = list(trackers_dir.rglob("*summary*.txt"))
    for sf in summary_files:
        shutil.copy2(sf, output_dir / sf.name)

    return parse_trackeval_summary(output_dir)


def _parse_one_summary_file(path: Path) -> dict:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    header = re.split(r"[\s,]+", lines[0])
    values = re.split(r"[\s,]+", lines[1])
    result = {}
    for key, val in zip(header, values):
        try:
            f = float(val)
            result[key] = int(f) if f == int(f) else f
        except ValueError:
            result[key] = val
    return result


def parse_trackeval_summary(output_root: Path) -> dict:
    all_files = sorted(output_root.rglob("*summary*.txt"))
    if not all_files:
        raise FileNotFoundError(f"No TrackEval summary files under {output_root}")

    def sort_key(p: Path) -> int:
        n = p.name.lower()
        if "pedestrian" in n: return 0
        if "clear" in n: return 1
        if "hota" in n: return 2
        if "identity" in n: return 3
        return 4

    merged: dict = {}
    for path in sorted(all_files, key=sort_key):
        parsed = _parse_one_summary_file(path)
        if parsed:
            print(f"  [TrackEval] Parsed: {path.name} ({len(parsed)} fields)")
            for k, v in parsed.items():
                if k not in merged:
                    merged[k] = v

    if not merged:
        raise RuntimeError(f"All summary files empty under {output_root}")

    aliases = [
        ("CLR_Re", "Recall"), ("CLR_Pr", "Precision"),
        ("CLR_FP", "FP"), ("CLR_FN", "FN"),
        ("GT_IDs", "num_unique_objects"),
        ("MTR", "mostly_tracked_ratio"), ("MLR", "mostly_lost_ratio"),
    ]
    for src, dst in aliases:
        if src in merged and dst not in merged:
            merged[dst] = merged[src]

    return merged
