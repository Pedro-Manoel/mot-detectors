#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from core.config import load_config, environment_manifest
from core import data, diagnostics, report, runner


def _prepare(cfg) -> None:
    for dataset, seqs in cfg.sequences.items():
        for seq in seqs:
            out = Path(cfg.prepared_root) / dataset / seq
            if (out / "meta.json").exists():
                print(f"[skip] {dataset}/{seq}")
                continue
            print(f"[prepare] {dataset}/{seq}")
            data.prepare_sequence(Path(cfg.dataset_root), dataset, seq, Path(cfg.prepared_root))


def _results_dir(cfg) -> Path:
    return Path(cfg.output_root) / cfg.name


def _load(args):
    # --name writes the same grid to results/<name>/, e.g. a second run of the paper's grid
    # to compare with the first.
    cfg = load_config(args.config)
    if getattr(args, "name", ""):
        cfg.name = args.name
    return cfg


def cmd_prepare(args) -> None:
    cfg = load_config(args.config)
    with runner.log_to(_results_dir(cfg) / "logs", "prepare"):
        _prepare(cfg)


def cmd_run(args) -> None:
    cfg = _load(args)
    quicktest = bool(args.max_frames and args.max_frames > 0)
    rdir = _results_dir(cfg)
    with runner.log_to(runner.results_root_for(cfg, quicktest) / "logs", "run"):
        _prepare(cfg)
        if not quicktest and (args.force or runner.status(cfg)["pending"]):
            rdir.mkdir(parents=True, exist_ok=True)
            mpath = rdir / "manifest.json"
            if mpath.exists():
                # Keep the first manifest of a results folder; record later invocations
                # (resumed or forced runs) next to it.
                mpath = rdir / f"manifest_{datetime.now():%Y%m%d_%H%M%S}.json"
            runner.atomic_write_text(mpath, json.dumps(environment_manifest(cfg), indent=2))
        # A quick test always starts over, so a longer --max-frames never reuses shorter runs.
        summary = runner.run_batch(cfg, force=args.force or quicktest, max_frames=args.max_frames)
        present = f"{summary['done_before']} runs already present"
        if args.force or quicktest:
            present += " (re-run)"
        print(f"\nDone: {summary['completed']} runs written, {summary['failed']} groups failed, "
              f"{present}. Results -> {summary['results_root']}")
        allcsv = Path(summary["results_root"]) / "all_results.csv"
        if not quicktest and summary["completed"] and allcsv.exists():
            print(f"Aggregated -> {allcsv}")


def cmd_status(args) -> None:
    st = runner.status(_load(args))
    print(f"total {st['total']} | done {st['done']} | pending {st['pending']} "
          f"| failed-records {len(st['failed_records'])}")
    for r in st["failed_records"]:
        print(f"  FAILED {r.get('key')}: {r.get('error')}")


def cmd_analyze(args) -> None:
    cfg = _load(args)
    rdir = Path(args.results) if args.results else _results_dir(cfg)
    with runner.log_to(rdir / "logs", "analyze"):
        if not args.results:
            n, done = sum(1 for _ in rdir.rglob("metrics.csv")), runner.status(cfg)["done"]
            if n != done:
                print(f"[WARN] {rdir} holds {n} runs but the grid of {args.config} has {done}; "
                      "the tables include runs outside the grid (for instance an older device "
                      "name or run count).")
        # paper_results/ ships all_results.csv without the per-run folders it was built from.
        allcsv = rdir / "all_results.csv"
        if any(rdir.rglob("metrics.csv")) or not allcsv.exists():
            allcsv = report.aggregate(rdir)
        written = report.build_tables(allcsv, rdir / "tables")
        diag = diagnostics.build_diagnostics_table(rdir, cfg.prepared_root)
        if not diag.empty:
            dpath = rdir / "tables" / "detection_diagnostics.csv"
            diag.to_csv(dpath, index=False)
            written.append(dpath)
        jit = diagnostics.build_jitter_table(rdir)
        if not jit.empty:
            jpath = rdir / "tables" / "box_jitter.csv"
            jit.to_csv(jpath, index=False)
            written.append(jpath)
        print("Tables written:")
        for p in written:
            print("  ", p)


def cmd_models(args) -> None:
    cfg = load_config(args.config)
    out = (Path(args.results) if args.results else _results_dir(cfg)) / "tables"
    table = report.models_table(cfg)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "models.csv", index=False)
    (out / "models.tex").write_text(report.models_to_latex(table), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"Tables written: {out / 'models.csv'}, {out / 'models.tex'}")


def cmd_check(args) -> None:
    from core.reproduce import PAPER_ENVIRONMENT, check_files, check_software
    cfg = load_config(args.config)
    mpath = Path(args.manifest) if args.manifest else PAPER_ENVIRONMENT
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    rows = check_software(cfg, manifest) + check_files(cfg, manifest)
    for status, item, detail in rows:
        print(f"[{status}] {item:22s} {detail}")
    n_fail = sum(s == "FAIL" for s, _, _ in rows)
    n_warn = sum(s == "WARN" for s, _, _ in rows)
    print(f"\nCHECK: {'FAIL' if n_fail else 'PASS'} ({n_fail} failed, {n_warn} warnings) "
          f"against {mpath}")
    raise SystemExit(1 if n_fail else 0)


def cmd_compare(args) -> None:
    from core.reproduce import compare_results
    ok, lines = compare_results(args.reference, args.candidate)
    print("\n".join(lines))
    print("\nCOMPARE:", "PASS" if ok else "FAIL (quality beyond tolerance)")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description="config-driven MOT study runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run", "status", "analyze", "models", "check"):
        sp = sub.add_parser(name)
        sp.add_argument("config")
        if name in ("run", "status", "analyze"):
            sp.add_argument("--name", default="",
                            help="use results/<name>/ instead of the YAML's name")
        if name == "run":
            sp.add_argument("--force", action="store_true")
            sp.add_argument("--max-frames", type=int, default=0)
        if name == "analyze":
            sp.add_argument("--results", default="",
                            help="folder to analyze, e.g. paper_results (default: results/<name>/)")
        if name == "models":
            sp.add_argument("--results", default="",
                            help="folder whose tables/ receives models.csv and models.tex")
        if name == "check":
            sp.add_argument("--manifest", default="",
                            help="reference environment (default: experiments/paper_environment.json)")
    sp = sub.add_parser("compare", help="compare the quality tables of two results folders")
    sp.add_argument("reference", help="e.g. paper_results")
    sp.add_argument("candidate", help="e.g. results/paper_main")
    args = p.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "status": cmd_status, "analyze": cmd_analyze,
     "models": cmd_models, "check": cmd_check,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
