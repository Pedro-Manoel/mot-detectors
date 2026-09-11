from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .report import benchmark_order, MODEL_ORDER

DECOMP_METRICS = ["HOTA", "DetA", "AssA", "LocA"]
ATTRIBUTION_METRICS = ["Recall", "Precision", "DetA", "AssA", "IDF1", "IDR", "IDP",
                       "MOTA", "MT", "ML", "IDSW", "Frag"]


def confidence_interval(values, confidence: float = 0.95) -> dict:
    from scipy import stats

    a = np.asarray([v for v in values
                    if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    n = int(a.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan"), "half_width": float("nan")}
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "std": float("nan"), "n": n,
                "ci_low": mean, "ci_high": mean, "half_width": 0.0}
    std = float(a.std(ddof=1))
    se = std / np.sqrt(n)
    t = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1))
    hw = t * se
    return {"mean": mean, "std": std, "n": n,
            "ci_low": mean - hw, "ci_high": mean + hw, "half_width": hw}


def bootstrap_ci(values, confidence: float = 0.95, n_boot: int = 10000, seed: int = 0) -> dict:
    a = np.asarray([v for v in values
                    if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    n = int(a.size)
    if n == 0:
        return {"mean": float("nan"), "n": 0,
                "boot_ci_low": float("nan"), "boot_ci_high": float("nan")}
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "n": n, "boot_ci_low": mean, "boot_ci_high": mean}
    rng = np.random.default_rng(seed)
    boot_means = a[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {"mean": mean, "n": n,
            "boot_ci_low": float(np.percentile(boot_means, 100 * alpha)),
            "boot_ci_high": float(np.percentile(boot_means, 100 * (1 - alpha)))}


def between_sequence_ci(df: pd.DataFrame, tracker: str, metric: str,
                        confidence: float = 0.95, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    recs = []
    for (fam, sc, mode, ds), grp in d.groupby(
            ["model_family", "model_scale", "mode", "dataset"]):
        per_seq = grp.groupby("sequence")[metric].mean().tolist()
        t = confidence_interval(per_seq, confidence)
        b = bootstrap_ci(per_seq, confidence, n_boot, seed)
        recs.append(dict(
            family=fam, scale=sc, mode=mode, label=_label_for(fam, sc, mode),
            dataset=ds, metric=metric, n_sequences=t["n"], mean=t["mean"],
            std_between_seq=t["std"], t_ci_low=t["ci_low"], t_ci_high=t["ci_high"],
            boot_ci_low=b["boot_ci_low"], boot_ci_high=b["boot_ci_high"]))
    return pd.DataFrame.from_records(recs)


def _label_for(family: str, scale: str, mode: str) -> str:
    for f, s, m, label in MODEL_ORDER:
        if (f, s, m) == (family, scale, mode):
            return label
    return f"{family}{scale}" + ("" if mode == "nms" else f" ({mode})")


def summary_ci_table(df: pd.DataFrame, tracker: str, metric: str,
                     confidence: float = 0.95) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    recs = []
    for (fam, sc, mode, ds), grp in d.groupby(
            ["model_family", "model_scale", "mode", "dataset"]):
        ci = confidence_interval(grp[metric].tolist(), confidence)
        recs.append(dict(family=fam, scale=sc, mode=mode,
                         label=_label_for(fam, sc, mode), dataset=ds, metric=metric,
                         mean=ci["mean"], std=ci["std"], n=ci["n"],
                         ci_low=ci["ci_low"], ci_high=ci["ci_high"],
                         ci_half_width=ci["half_width"]))
    return pd.DataFrame.from_records(recs)


def _per_benchmark_block(df: pd.DataFrame, tracker: str, metrics: list[str]) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    g = (d.groupby(["model_family", "model_scale", "mode", "dataset"])[metrics].mean())
    out = g.unstack("dataset").swaplevel(axis=1)
    cols = pd.MultiIndex.from_product([benchmark_order(df), metrics], names=["dataset", "metric"])
    out = out.reindex(columns=cols)
    out.index = out.index.set_names(["family", "scale", "mode"])
    return out


def hota_decomposition_table(df: pd.DataFrame, tracker: str) -> pd.DataFrame:
    return _per_benchmark_block(df, tracker, DECOMP_METRICS)


def degradation_table(df: pd.DataFrame, tracker: str,
                      metrics=("HOTA", "DetA", "AssA")) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    order = benchmark_order(df)
    sparse_ds, dense_ds = order[0], order[-1]
    metrics = list(metrics)
    recs = []
    for (fam, sc, mode), grp in d.groupby(["model_family", "model_scale", "mode"]):
        rec = dict(family=fam, scale=sc, mode=mode, label=_label_for(fam, sc, mode),
                   sparse_benchmark=sparse_ds, dense_benchmark=dense_ds)
        for m in metrics:
            sparse = grp[grp["dataset"] == sparse_ds][m].mean()
            dense = grp[grp["dataset"] == dense_ds][m].mean()
            rec[f"{m}_{sparse_ds}"] = sparse
            rec[f"{m}_{dense_ds}"] = dense
            rec[f"{m}_delta"] = dense - sparse
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


def attribution_table(df: pd.DataFrame, tracker: str) -> pd.DataFrame:
    d = df[df["tracker"] == tracker]
    present = [m for m in ATTRIBUTION_METRICS if m in d.columns]
    recs = []
    for (fam, sc, mode, ds), grp in d.groupby(
            ["model_family", "model_scale", "mode", "dataset"]):
        rec = dict(family=fam, scale=sc, mode=mode,
                   label=_label_for(fam, sc, mode), dataset=ds)
        for m in present:
            rec[m] = grp[m].mean()
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


def nms_vs_e2e_significance(df: pd.DataFrame, tracker: str,
                            metrics=("HOTA", "IDF1", "Frag", "IDSW")) -> pd.DataFrame:
    from scipy import stats

    d = df[(df["tracker"] == tracker) & (df["model_family"] == "yolo26")]
    if d.empty or not {"nms", "end2end"} <= set(d["mode"]):
        return pd.DataFrame()

    # Pair at the sequence level: the repetitions of one configuration are near-identical
    # (cudnn noise only), so pairing run by run would count each sequence `runs` times and
    # inflate the significance.
    keys = ["model_scale", "dataset", "sequence"]
    metrics = [m for m in metrics if m in d.columns]
    per_seq = d.groupby(["mode"] + keys)[metrics].mean().reset_index()
    nms = per_seq[per_seq["mode"] == "nms"]
    e2e = per_seq[per_seq["mode"] == "end2end"]
    merged = pd.merge(nms[keys + metrics], e2e[keys + metrics],
                      on=keys, suffixes=("_nms", "_e2e"))
    if merged.empty:
        return pd.DataFrame()

    def _stats(sub: pd.DataFrame, scale_label: str) -> dict:
        rec = {"scale": scale_label, "n_pairs": len(sub)}
        for m in metrics:
            diff = (sub[f"{m}_e2e"] - sub[f"{m}_nms"]).to_numpy(dtype=float)
            rec[f"{m}_e2e_minus_nms"] = float(np.mean(diff)) if len(diff) else float("nan")
            if len(diff) >= 1 and np.any(diff != 0):
                try:
                    p = float(stats.wilcoxon(diff, zero_method="wilcox",
                                             alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
            else:
                p = float("nan")
            rec[f"{m}_pvalue"] = p
        return rec

    recs = [_stats(sub, scale) for scale, sub in merged.groupby("model_scale")]
    recs.append(_stats(merged, "ALL"))
    return pd.DataFrame.from_records(recs)


# Arguments that core.track.load_tracker sets explicitly; everything else is left at the
# BoxMOT default.
TRACKER_SETTINGS = {
    "bytetrack": {"per_class": False, "frame_rate": "seqinfo.ini frameRate"},
    "botsort": {"per_class": False, "half": False, "with_reid": True,
                "frame_rate": "seqinfo.ini frameRate"},
}


def tracker_hyperparameters() -> pd.DataFrame:
    import inspect

    from boxmot import BotSort, ByteTrack
    from boxmot.trackers.basetracker import BaseTracker

    # Live instances expose values BoxMOT derives from other arguments (ByteTrack sets
    # det_thresh from track_thresh). BoT-SORT is built without ReID so no weights are needed;
    # with_reid is reported from TRACKER_SETTINGS.
    live = {
        "bytetrack": ByteTrack(frame_rate=30, per_class=False),
        "botsort": BotSort(reid_weights=Path("placeholder.pt"), device="cpu", half=False,
                           frame_rate=30, with_reid=False, per_class=False),
    }
    recs = []
    for name, cls in (("bytetrack", ByteTrack), ("botsort", BotSort)):
        defaults: dict = {}
        for c in (BaseTracker, cls):          # subclass defaults override the base ones
            for p in inspect.signature(c.__init__).parameters.values():
                if p.default is not inspect.Parameter.empty:
                    defaults[p.name] = p.default
        settings = TRACKER_SETTINGS[name]
        for pname in list(defaults) + [k for k in settings if k not in defaults]:
            if pname in settings:
                value, source = settings[pname], "set by core.track"
            else:
                value = getattr(live[name], pname, defaults[pname])
                source = "BoxMOT default" if value == defaults[pname] else "BoxMOT, derived"
            recs.append(dict(tracker=name, parameter=pname, value=value, source=source))
    return pd.DataFrame.from_records(recs)
