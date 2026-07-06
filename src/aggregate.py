import csv
import json
import statistics
import sys
from pathlib import Path

import yaml

METRICS = [
    "auroc",
    "acc",
    "fpr",
    "oscr",
    "norm_gap_l2",
    "norm_gap_l1",
    "maxcos_id",
    "maxcos_ood",
    "conf_ood",
]
# Config columns captured per run. ``seed`` is the only one aggregated OVER in
# the by-seed mode; the rest (plus alpha / gate / scorer_frozen, which some
# ablations vary) together identify one configuration.
CONFIG_COLS = [
    "corruption",
    "method",
    "score",
    "gmm_fit",
    "ood_op",
    "label",
    "lam",
    "marginal_lambda",
    "warmup_K",
    "reliability_gate",
    "scorer_frozen",
    "alpha",
    "seed",
]
# Everything that identifies a configuration (group key for by-seed aggregation).
GROUP_COLS = [c for c in CONFIG_COLS if c != "seed"]


def _row(run_dir: Path) -> dict:
    """One flat CSV row for a run dir: its config columns + every metric at t0/tT."""
    summ = json.loads((run_dir / "summary.json").read_text())
    cfg = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    m = cfg.get("method", {})
    row = {
        "corruption": cfg.get("corruption"),
        "method": m.get("name"),
        "score": m.get("score"),
        "gmm_fit": m.get("gmm_fit"),
        "ood_op": m.get("ood_op"),
        "label": m.get("label"),
        "lam": m.get("lam"),
        "marginal_lambda": m.get("marginal_lambda"),
        "warmup_K": m.get("warmup_K"),
        "reliability_gate": m.get("reliability_gate"),
        "scorer_frozen": m.get("scorer_frozen"),
        "alpha": cfg.get("alpha"),
        "seed": cfg.get("seed"),
    }
    # Tolerant of older runs that predate fpr/oscr (their keys are simply absent).
    for k in METRICS:
        mk = summ.get(k)
        row[f"{k}_t0"] = mk["t0"] if mk else None
        row[f"{k}_tT"] = mk["tT"] if mk else None
    return row


def _collect(multirun_dir: str) -> list[dict]:
    """All run rows under ``multirun_dir`` (any dir holding summary.json + .hydra)."""
    root = Path(multirun_dir)
    run_dirs = sorted(
        p.parent
        for p in root.rglob("summary.json")
        if (p.parent / ".hydra" / "config.yaml").exists()
    )
    return [_row(d) for d in run_dirs]


def aggregate(multirun_dir: str, out_csv: str) -> int:
    """Flat dump: one row per run (config cols + every metric at t0/tT).

    :param multirun_dir: root of a Hydra multirun tree holding the run dirs.
    :type multirun_dir: str
    :param out_csv: destination CSV path (parent dirs are created).
    :type out_csv: str
    :returns: the number of rows written.
    :rtype: int
    """
    rows = _collect(multirun_dir)
    rows.sort(key=lambda r: (str(r["corruption"]), str(r["method"]), str(r["seed"])))
    cols = CONFIG_COLS + [f"{k}_{t}" for k in METRICS for t in ("t0", "tT")]
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _mean_std(vals: list) -> tuple:
    """Mean and sample (ddof=1) std of the non-None values; std=0 if <2 points."""
    xs = [v for v in vals if v is not None]
    if not xs:
        return None, None
    mean = statistics.fmean(xs)
    std = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return mean, std


def aggregate_by_seed(multirun_dir: str, out_csv: str) -> int:
    """Group runs by configuration (all of GROUP_COLS) and reduce over seeds.

    Emits one row per configuration with ``n_seeds`` and, for every metric and
    endpoint, ``<metric>_<t0|tT>_mean`` / ``<metric>_<t0|tT>_std`` (sample std).

    :param multirun_dir: root of a Hydra multirun tree holding the run dirs.
    :type multirun_dir: str
    :param out_csv: destination CSV path (parent dirs are created).
    :type out_csv: str
    :returns: the number of grouped configurations written.
    :rtype: int
    """
    rows = _collect(multirun_dir)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r[c] for c in GROUP_COLS)
        groups.setdefault(key, []).append(r)

    metric_cols = [
        f"{k}_{t}_{stat}"
        for k in METRICS
        for t in ("tT", "t0")
        for stat in ("mean", "std")
    ]
    cols = GROUP_COLS + ["n_seeds"] + metric_cols

    out_rows = []
    for key, grp in groups.items():
        row = dict(zip(GROUP_COLS, key))
        row["n_seeds"] = len(grp)
        for k in METRICS:
            for t in ("tT", "t0"):
                mean, std = _mean_std([g[f"{k}_{t}"] for g in grp])
                row[f"{k}_{t}_mean"] = mean
                row[f"{k}_{t}_std"] = std
        out_rows.append(row)
    out_rows.sort(key=lambda r: (str(r["corruption"]), str(r["method"]), str(r["alpha"]), str(r["lam"])))

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    return len(out_rows)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--by-seed"]
    if "--by-seed" in sys.argv:
        n = aggregate_by_seed(args[0], args[1])
        print(f"wrote {n} grouped (by-seed) configs to {args[1]}")
    else:
        n = aggregate(args[0], args[1])
        print(f"wrote {n} rows to {args[1]}")
