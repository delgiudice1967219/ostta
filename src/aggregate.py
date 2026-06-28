import csv
import json
import sys
from pathlib import Path

import yaml

METRICS = [
    "auroc",
    "acc",
    "norm_gap_l2",
    "norm_gap_l1",
    "maxcos_id",
    "maxcos_ood",
    "conf_ood",
]
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
    "seed",
]


def _row(run_dir: Path) -> dict:
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
        "seed": cfg.get("seed"),
    }
    for k in METRICS:
        row[f"{k}_t0"] = summ[k]["t0"]
        row[f"{k}_tT"] = summ[k]["tT"]
    return row


def aggregate(multirun_dir: str, out_csv: str) -> int:
    root = Path(multirun_dir)
    run_dirs = sorted(
        p.parent
        for p in root.rglob("summary.json")
        if (p.parent / ".hydra" / "config.yaml").exists()
    )
    rows = [_row(d) for d in run_dirs]
    rows.sort(key=lambda r: (str(r["corruption"]), str(r["method"])))
    cols = CONFIG_COLS + [f"{k}_{t}" for k in METRICS for t in ("t0", "tT")]
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    n = aggregate(sys.argv[1], sys.argv[2])
    print(f"wrote {n} rows to {sys.argv[2]}")
