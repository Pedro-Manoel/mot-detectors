from __future__ import annotations

from pathlib import Path

import pandas as pd

QUALITY_METRICS = ["HOTA", "IDF1", "MOTA", "IDSW", "Frag"]
LOWER_IS_BETTER = {"IDSW", "Frag"}


def benchmark_order(df: pd.DataFrame) -> list[str]:
    if "density" in df.columns and df["density"].notna().any():
        return list(df.groupby("dataset")["density"].mean().sort_values().index)
    return sorted(df["dataset"].astype(str).unique())


MODEL_ORDER = [
    ("yolo11", "n", "nms", "YOLO11-N"), ("yolo11", "s", "nms", "YOLO11-S"),
    ("yolo11", "m", "nms", "YOLO11-M"), ("yolo11", "l", "nms", "YOLO11-L"),
    ("yolo11", "x", "nms", "YOLO11-X"),
    ("rtdetr", "l", "nms", "RT-DETR-L"), ("rtdetr", "x", "nms", "RT-DETR-X"),
    ("yolo26", "n", "nms", "YOLO26-N (NMS)"), ("yolo26", "n", "end2end", "YOLO26-N (E2E)"),
    ("yolo26", "s", "nms", "YOLO26-S (NMS)"), ("yolo26", "s", "end2end", "YOLO26-S (E2E)"),
    ("yolo26", "m", "nms", "YOLO26-M (NMS)"), ("yolo26", "m", "end2end", "YOLO26-M (E2E)"),
    ("yolo26", "l", "nms", "YOLO26-L (NMS)"), ("yolo26", "l", "end2end", "YOLO26-L (E2E)"),
    ("yolo26", "x", "nms", "YOLO26-X (NMS)"), ("yolo26", "x", "end2end", "YOLO26-X (E2E)"),
]

TRACKER_NAMES = {"bytetrack": "ByteTrack", "botsort": "BoT-SORT"}


def _ordered_trackers(names) -> list[str]:
    present = set(names)
    return [t for t in TRACKER_NAMES if t in present] + sorted(present - set(TRACKER_NAMES))


def aggregate(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    rows = [pd.read_csv(p) for p in sorted(results_dir.rglob("metrics.csv"))]
    if not rows:
        raise FileNotFoundError(f"No metrics.csv under {results_dir}")
    df = pd.concat(rows, ignore_index=True)
    out = results_dir / "all_results.csv"
    df.to_csv(out, index=False)
    return out


def quality_table(df: pd.DataFrame, tracker: str) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    order = benchmark_order(df)
    g = (d.groupby(["model_family", "model_scale", "mode", "dataset"])[QUALITY_METRICS]
           .mean())
    out = g.unstack("dataset")
    out = out.swaplevel(axis=1)
    cols = pd.MultiIndex.from_product([order, QUALITY_METRICS],
                                      names=["dataset", "metric"])
    out = out.reindex(columns=cols)
    out.index = out.index.set_names(["family", "scale", "mode"])
    return out


def efficiency_table(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (fam, sc, mode, trk, ds), grp in df.groupby(
            ["model_family", "model_scale", "mode", "tracker", "dataset"]):
        per_seq = grp.groupby("sequence")
        lat_mean = grp["avg_total_ms"].mean()
        recs.append(dict(
            family=fam, scale=sc, mode=mode, tracker=trk, dataset=ds,
            fps_mean=1000.0 / lat_mean,
            fps_std=per_seq["fps_total"].std(ddof=1).mean(),
            lat_mean=lat_mean,
            lat_std=per_seq["avg_total_ms"].std(ddof=1).mean(),
        ))
    return pd.DataFrame.from_records(recs)


def _ordered_quality_rows(q: pd.DataFrame):
    for fam, sc, mode, label in MODEL_ORDER:
        if (fam, sc, mode) in q.index:
            yield fam, label, q.loc[(fam, sc, mode)]


def _best(values: dict, lower: bool) -> set:
    values = {k: v for k, v in values.items() if pd.notna(v)}
    if not values:
        return set()
    best = (min if lower else max)(values.values())
    return {k for k, v in values.items() if v == best}


def _body(rows) -> list[str]:
    width = max((len(label) for _, label, _ in rows), default=0)
    lines, prev = [], None
    for fam, label, cells in rows:
        if prev is not None and fam != prev:
            lines.append(r"\midrule")
        lines.append(f"{label.ljust(width)} & " + " & ".join(cells) + r" \\")
        prev = fam
    return lines


def _quality_to_latex(q: pd.DataFrame, tracker: str) -> str:
    order = list(dict.fromkeys(q.columns.get_level_values("dataset")))
    rows = list(_ordered_quality_rows(q))
    cols = [(b, m) for b in order for m in QUALITY_METRICS]
    text = {(label, c): "--" if pd.isna(row[c]) else f"{row[c]:.2f}"
            for _, label, row in rows for c in cols}
    bold = {c: _best({label: float(text[(label, c)]) for _, label, _ in rows
                      if text[(label, c)] != "--"}, c[1] in LOWER_IS_BETTER) for c in cols}
    width = {c: max((len(text[(label, c)]) for _, label, _ in rows), default=0) for c in cols}

    def cell(label, c):
        value = text[(label, c)]
        return rf"\textbf{{{value}}}" if label in bold[c] else value.rjust(width[c])

    k = len(QUALITY_METRICS)
    heads = [" & " + " & ".join(m + (r"$\downarrow$" if m in LOWER_IS_BETTER else r"$\uparrow$")
                                for m in QUALITY_METRICS) for _ in order]
    heads[-1] += r" \\"
    lines = [r"\begin{table*}[!thb]", r"\centering",
             rf"\caption{{Results with {TRACKER_NAMES.get(tracker, tracker)}.}}",
             rf"\label{{tab:quality_{tracker}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             rf"\begin{{tabular}}{{l *{{{k * len(order)}}}{{c}}}}", r"\toprule",
             r"\multirow{2}{*}{Model} & "
             + " & ".join(rf"\multicolumn{{{k}}}{{c}}{{{b}}}" for b in order) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + k * i}-{1 + k * (i + 1)}}}" for i in range(len(order))),
             *heads, r"\midrule",
             *_body([(fam, label, [cell(label, c) for c in cols]) for fam, label, _ in rows]),
             r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def _efficiency_to_latex(eff: pd.DataFrame, order: list[str]) -> str:
    trackers = _ordered_trackers(eff["tracker"])
    e = eff.set_index(["family", "scale", "mode", "tracker", "dataset"])
    models = [(f, s, m, label) for f, s, m, label in MODEL_ORDER
              if any((f, s, m, t, b) in e.index for t in trackers for b in order)]
    cols = [(b, t, kind) for b in order for t in trackers for kind in ("FPS", "Lat")]
    text = {}
    for f, s, m, label in models:
        for b, t, kind in cols:
            if (f, s, m, t, b) in e.index:
                r = e.loc[(f, s, m, t, b)]
                mean, std = (r.fps_mean, r.fps_std) if kind == "FPS" else (r.lat_mean, r.lat_std)
                text[(label, (b, t, kind))] = (f"{mean:.1f}", "" if pd.isna(std) else f"{std:.1f}")
    bold = {c: _best({label: float(text[(label, c)][0]) for *_, label in models
                      if (label, c) in text}, c[2] == "Lat") for c in cols}

    def cell(label, c):
        if (label, c) not in text:
            return "--"
        mean, std = text[(label, c)]
        spread = rf"{{\scriptsize$\pm${std}}}" if std else ""
        return (rf"\textbf{{{mean}}}" if label in bold[c] else mean) + spread

    per = 2 * len(trackers)
    names = " & ".join(rf"\multicolumn{{2}}{{c}}{{{TRACKER_NAMES.get(t, t)}}}" for t in trackers)
    units = " & ".join(r"FPS$\uparrow$ & Lat. (ms)$\downarrow$" for _ in trackers)
    lines = [r"\begin{table*}[!thb]", r"\centering", r"\caption{Computational efficiency.}",
             r"\label{tab:efficiency}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\resizebox{\textwidth}{!}{", r"\begin{tabular}{l" + "c" * len(cols) + "}",
             r"\toprule", r"\multirow{3}{*}{Model} "]
    lines += [rf"& \multicolumn{{{per}}}{{c}}{{{b}}} " for b in order]
    lines[-1] = lines[-1].rstrip() + r" \\"
    lines.append(" ".join(rf"\cmidrule(lr){{{2 + per * i}-{1 + per * (i + 1)}}}"
                          for i in range(len(order))))
    lines += [f"& {names}" for _ in order]
    lines[-1] += r" \\"
    for i in range(len(order)):
        start = 2 + per * i
        lines.append(" ".join(rf"\cmidrule(lr){{{start + 2 * j}-{start + 2 * j + 1}}}"
                              for j in range(len(trackers))))
    lines += [f"& {units}" for _ in order]
    lines[-1] += r" \\"
    lines += [r"\midrule",
              *_body([(f, label, [cell(label, c) for c in cols]) for f, s, m, label in models]),
              r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    return "\n".join(lines)


def build_tables(all_results_csv: Path, out_dir: Path) -> list[Path]:
    df = pd.read_csv(all_results_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for trk in _ordered_trackers(df["tracker"]):
        q = quality_table(df, trk)
        q.to_csv(out_dir / f"quality_{trk}.csv")
        (out_dir / f"quality_{trk}.tex").write_text(_quality_to_latex(q, trk), encoding="utf-8")
        written += [out_dir / f"quality_{trk}.csv", out_dir / f"quality_{trk}.tex"]
    efficiency = efficiency_table(df)
    efficiency.to_csv(out_dir / "efficiency.csv", index=False)
    (out_dir / "efficiency.tex").write_text(_efficiency_to_latex(efficiency, benchmark_order(df)),
                                            encoding="utf-8")
    return written + [out_dir / "efficiency.csv", out_dir / "efficiency.tex"]
