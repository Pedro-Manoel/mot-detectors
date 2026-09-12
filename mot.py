from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import data, report, runner
from core.config import load_config
from core.reproduce import PAPER_ENVIRONMENT, check_files, check_software, compare_results


def _prepare(cfg) -> None:
    for dataset, seqs in cfg.sequences.items():
        for seq in seqs:
            out = Path(cfg.prepared_root) / dataset / seq
            if (out / "meta.json").exists():
                print(f"[skip] {dataset}/{seq}")
                continue
            print(f"[prepare] {dataset}/{seq}")
            data.prepare_sequence(Path(cfg.dataset_root), dataset, seq, Path(cfg.prepared_root))


def _load(args):
    cfg = load_config(args.config)
    if getattr(args, "name", ""):
        cfg.name = args.name
    return cfg


def cmd_prepare(args) -> None:
    cfg = load_config(args.config)
    with runner.log_to(runner.results_root_for(cfg) / "logs", "prepare"):
        _prepare(cfg)


def cmd_run(args) -> None:
    cfg = _load(args)
    quicktest = bool(args.max_frames and args.max_frames > 0)
    with runner.log_to(runner.results_root_for(cfg, quicktest) / "logs", "run"):
        _prepare(cfg)
        summary =runner.run_batch(cfg, force=args.force or quicktest, max_frames=args.max_frames)
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
    rdir = Path(args.results) if args.results else runner.results_root_for(cfg)
    with runner.log_to(rdir / "logs", "analyze"):
        n_runs = sum(1 for _ in rdir.rglob("metrics.csv"))
        if not args.results:
            done = runner.status(cfg)["done"]
            if n_runs != done:
                print(f"[WARN] {rdir} holds {n_runs} runs but the grid of {args.config} has "
                      f"{done}; the tables include runs outside the grid (for instance an older "
                      "device name or run count).")
        allcsv = rdir / "all_results.csv"
        if n_runs or not allcsv.exists():
            allcsv = report.aggregate(rdir)
        written = report.build_tables(allcsv, rdir / "tables")
        print("Tables written:")
        for p in written:
            print("  ", p)


def cmd_check(args) -> None:
    cfg = load_config(args.config)
    env = json.loads(PAPER_ENVIRONMENT.read_text(encoding="utf-8"))
    rows = check_software(cfg, env) + check_files(cfg, env)
    for status, item, detail in rows:
        print(f"[{status}] {item:22s} {detail}")
    n_fail = sum(s == "FAIL" for s, _, _ in rows)
    n_warn = sum(s == "WARN" for s, _, _ in rows)
    print(f"\nCHECK: {'FAIL' if n_fail else 'PASS'} ({n_fail} failed, {n_warn} warnings) "
          f"against {PAPER_ENVIRONMENT.name}")
    raise SystemExit(1 if n_fail else 0)


def cmd_compare(args) -> None:
    ok, lines = compare_results(args.reference, args.candidate)
    print("\n".join(lines))
    print("\nCOMPARE:", "PASS" if ok else "FAIL (quality beyond tolerance)")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description="config-driven MOT study runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "run", "status", "analyze", "check"):
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
    sp = sub.add_parser("compare", help="compare the quality tables of two results folders")
    sp.add_argument("reference", help="e.g. paper_results")
    sp.add_argument("candidate", help="e.g. results/paper_main")
    args = p.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "status": cmd_status, "analyze": cmd_analyze,
     "check": cmd_check, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
