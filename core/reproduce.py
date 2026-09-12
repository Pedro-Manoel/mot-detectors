from __future__ import annotations

import hashlib
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import pandas as pd

from . import evaluate, report
from .config import Config, weight_filename
from .runner import PROJECT_ROOT

PAPER_ENVIRONMENT = PROJECT_ROOT / "experiments" / "paper_environment.json"
TABLE_TOL = {"HOTA": 0.6, "IDF1": 0.8, "MOTA": 0.6, "IDSW": 4, "Frag": 10}
RUN_KEYS = ["model_family", "model_scale", "mode", "tracker", "dataset", "sequence"]
_LABELS = {(f, s, m): label for f, s, m, label in report.MODEL_ORDER}


def file_sha256(path: str | Path, text: bool = False) -> str:
    data =Path(path).read_bytes()
    if text:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git_commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def check_software(cfg: Config, env: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    exp_py, cur_py = env.get("python", ""), platform.python_version()
    same_minor = cur_py.split(".")[:2] == exp_py.split(".")[:2]
    rows.append(("PASS" if same_minor else "WARN", "python", f"{cur_py} (paper {exp_py})"))

    for dist, exp in env.get("library_versions", {}).items():
        if dist.endswith("_note"):
            continue
        try:
            cur = metadata.version(dist)
        except metadata.PackageNotFoundError:
            rows.append(("FAIL", dist, f"not installed (paper {exp})"))
            continue
        rows.append(("PASS" if cur == exp else "WARN", dist, f"{cur} (paper {exp})"))

    if str(cfg.device).lower() != "cpu":
        paper_gpu = env.get("hardware", {}).get("gpu", "?")
        try:
            import torch
            if torch.cuda.is_available():
                rows.append(("PASS", "cuda", f"{torch.cuda.get_device_name(0)} (paper {paper_gpu}); "
                                             "FPS and latency depend on the hardware"))
            else:
                rows.append(("FAIL", "cuda", "not available, but the config runs on a GPU"))
        except Exception as exc:
            rows.append(("FAIL", "cuda", f"torch cannot be imported: {exc}"))

    te = env.get("trackeval", {})
    root = Path(cfg.trackeval_root)
    if not (root / "scripts" / "run_mot_challenge.py").exists():
        rows.append(("FAIL", "trackeval", f"no TrackEval clone at {root}"))
    else:
        commit, exp = _git_commit(root), te.get("commit", "")
        rows.append(("PASS" if commit == exp else "WARN", "trackeval commit",
                     f"{commit[:12]} (paper {exp[:12]})"))

    py = evaluate.resolve_python(cfg.trackeval_python)
    if Path(py).parent != Path(".") and not Path(py).exists():
        rows.append(("FAIL", "trackeval python", f"{py} not found; create the TrackEval "
                                                  "environment (README, Installation)"))
        return rows
    probe = ("import sys, numpy, scipy, trackeval; "
             "print(sys.version.split()[0], numpy.__version__, scipy.__version__)")
    try:
        out =subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=120,
                             cwd=str(root) if root.is_dir() else None)
        lines = out.stdout.strip().splitlines()
        got = lines[-1].split() if out.returncode == 0 and lines else []
        err = (out.stderr.strip().splitlines() or ["unknown error"])[-1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        got, err = [], str(exc)
    if len(got) != 3:
        rows.append(("FAIL", "trackeval python", f"{py}: cannot import trackeval ({err})"))
    else:
        for label, cur, key in zip(("trackeval python", "trackeval numpy", "trackeval scipy"),
                                   got, ("python", "numpy", "scipy")):
            exp = str(te.get(key, ""))
            rows.append(("PASS" if cur == exp else "WARN", label, f"{cur} (paper {exp})"))
    return rows


def check_files(cfg: Config, env: dict) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    needed = {weight_filename(d.family, s): Path(weight_filename(d.family, s))
              for d in cfg.detectors for s in d.scales}
    if "botsort" in cfg.trackers and cfg.reid_weights:
        needed[Path(cfg.reid_weights).name] = Path(cfg.reid_weights)
    for name, expected in env.get("weights_sha256", {}).items():
        if name not in needed:
            continue
        path = needed[name]
        if not path.exists():
            rows.append(("WARN", name, "missing; it is downloaded on first use, re-run check "
                                       "afterwards"))
        elif file_sha256(path) == expected:
            rows.append(("PASS", name, "sha256 matches the paper's weights"))
        else:
            rows.append(("FAIL", name, "sha256 differs from the paper's weights"))

    gt_hashes = env.get("prepared_gt_sha256", {})
    for dataset, seqs in cfg.sequences.items():
        for seq in seqs:
            if seq not in gt_hashes:
                continue
            gt = Path(cfg.prepared_root) / dataset / seq / "gt" / "gt.txt"
            if not gt.exists():
                rows.append(("WARN", f"GT {seq}", "not prepared yet; run mot.py prepare"))
            elif file_sha256(gt, text=True) == gt_hashes[seq]:
                rows.append(("PASS", f"GT {seq}", "filtered GT matches the paper's"))
            else:
                rows.append(("FAIL", f"GT {seq}", "filtered GT differs from the paper's "
                                                  "(dataset version or preparation)"))
    return rows


def compare_results(reference_dir: str | Path, candidate_dir: str | Path) -> tuple[bool, list[str]]:
    ref = pd.read_csv(Path(reference_dir) / "all_results.csv")
    new = pd.read_csv(Path(candidate_dir) / "all_results.csv")
    pairs = ref[RUN_KEYS].drop_duplicates()
    common = pairs.merge(new[RUN_KEYS].drop_duplicates())
    if common.empty:
        return False, [f"[FAIL] no configuration and sequence in common with {reference_dir}"]
    lines: list[str] = []
    if len(common) < len(pairs):
        lines.append(f"[INFO] comparing the {len(common)} of {len(pairs)} configuration-sequence "
                     f"pairs present in {candidate_dir}")
    ref, new = ref.merge(common, on=RUN_KEYS), new.merge(common, on=RUN_KEYS)
    ok = True

    for trk in sorted(set(ref["tracker"])):
        qr, qn = report.quality_table(ref, trk), report.quality_table(new, trk)
        diff = (qn - qr).abs()
        for metric, t in TABLE_TOL.items():
            d = diff.xs(metric, axis=1, level="metric")
            n_bad = int((d > t).sum().sum())
            ok = ok and n_bad == 0
            lines.append(f"[{'PASS' if n_bad == 0 else 'FAIL'}] {trk:9s} {metric:4s} "
                         f"max |diff| {d.max().max():6.2f} over {int(d.notna().sum().sum())} "
                         f"cells, {n_bad} beyond {t}")
        hr = qr.xs("HOTA", axis=1, level="metric")
        hn = qn.xs("HOTA", axis=1, level="metric")
        for ds in hr.columns:
            if hr[ds].isna().all():
                continue
            br, bn = hr[ds].idxmax(), hn[ds].idxmax()
            lines.append(f"[{'PASS' if br == bn else 'WARN'}] {trk:9s} best HOTA on {ds}: "
                         f"{_LABELS.get(bn, bn)} (reference {_LABELS.get(br, br)})")

    keys = ["family", "scale", "mode", "tracker", "dataset"]
    eff = report.efficiency_table(ref).merge(report.efficiency_table(new), on=keys,
                                             suffixes=("_ref", "_new"))
    if not eff.empty:
        r = eff["fps_mean_new"] / eff["fps_mean_ref"]
        lines.append(f"[INFO] FPS candidate/reference: median {r.median():.2f} "
                     f"(range {r.min():.2f}-{r.max():.2f}); hardware-dependent, not checked")
    return ok, lines
