"""Paper figures rendered from saved results (no GPU, no re-running).

Each function loads saved ``.npz`` / ``.csv`` artifacts and writes a figure to
``paper/figures/``. Run ``python src/figures.py`` to render all of them.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("paper/figures")

# Per-method visual identity (consistent across every figure).
STYLE = {
    "bnadapt":     dict(label="BN-adapt",  color="#9aa3b0", ls="--", lw=1.6, z=1),
    "tent":        dict(label="Tent",      color="#c0392b", ls="-",  lw=2.0, z=2),
    "unient":      dict(label="UniEnt",    color="#d98a2b", ls="-",  lw=1.8, z=3),
    "unient_plus": dict(label="UniEnt+",   color="#e0b066", ls="--", lw=1.8, z=3),
    "nova":        dict(label="NOVA",      color="#3a3f9c", ls="-",  lw=2.6, z=5),
}
ORDER = ["bnadapt", "tent", "unient", "unient_plus", "nova"]


def _load_traj(corruption: str = "gaussian_noise") -> dict:
    """method -> timetrack arrays, from the stable sweep output dirs."""
    out = {}
    for m in ORDER:
        hits = glob.glob(f"multirun/sweep/{corruption}/{m}_{corruption}/timetrack.npz")
        if hits:
            out[m] = dict(np.load(hits[0]))
    return out


def fig_geometry(corruption: str = "gaussian_noise") -> Path:
    """The diagnosis: norm-gap, csOOD confidence, detection AUROC over t."""
    traj = _load_traj(corruption)
    if not traj:
        raise FileNotFoundError(f"no timetrack.npz under multirun/sweep/{corruption}/")

    panels = [
        ("norm_gap_l2", "feature-norm gap  (csID − csOOD)", "NOVA's lever: open the gap"),
        ("conf_ood",    "csOOD confidence",                       "Tent's failure: inflates it"),
        ("auroc",       "detection AUROC (energy)",               "the outcome"),
    ]
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 2.6))

    for ax, (key, ylab, sub) in zip(axes, panels):
        for m in ORDER:
            if m not in traj:
                continue
            s = STYLE[m]
            d = traj[m]
            ax.plot(d["t"], d[key], color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["z"])
        ax.set_title(sub, color="#444", fontsize=10.5, pad=6)
        ax.set_xlabel("adaptation step  t")
        ax.set_ylabel(ylab)
        ax.grid(True, color="#e8eaef", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.02)

    # one shared legend
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                          lw=STYLE[m]["lw"], label=STYLE[m]["label"]) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    # suptitle omitted: the paper caption carries this message (keeps the float compact).
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig_geometry.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_geometry.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def _read_csv(path: str) -> list[dict]:
    import csv
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fig_ablation() -> Path:
    """Lever-causal headline: detector fixed, only the csOOD op changes."""
    rows = {r["method"]: r for r in _read_csv("results/ablation_grid/summary.csv")}
    # Lever ablation (NOVA detector fixed): none -> entropy-max -> L1.
    lever = [("det_only", "none", "#9aa3b0"),
             ("novadet_entmax", "entropy-max", "#d98a2b"),
             ("nova", "L1-suppress\n(NOVA)", "#3a3f9c")]
    names = [lab for _, lab, _ in lever]
    vals = [float(rows[m]["auroc_tT"]) for m, _, _ in lever]
    cols = [c for _, _, c in lever]

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(names, vals, color=cols, width=0.62, zorder=3)
    ax.axhline(float(rows["tent"]["auroc_tT"]), color="#c0392b", ls=":", lw=1.5,
               zorder=2, label=f"Tent {float(rows['tent']['auroc_tT']):.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.80, 0.95)
    ax.set_ylabel("detection AUROC (t=80)")
    ax.set_title("csOOD lever, detector held fixed\n(max-cosine → pooled GMM → soft)",
                 fontsize=11)
    ax.grid(True, axis="y", color="#e8eaef", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig_ablation.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def fig_pooling() -> Path:
    """Pooling dynamics: held-out GMM NLL, pooled vs per-batch, at N=20 and N=200."""
    def load(dirpat):
        hits = glob.glob(f"{dirpat}/**/posterior_quality.npz", recursive=True)
        return dict(np.load(hits[0])) if hits else None

    data = [("N = 20  (starved per-batch fit)", load("results/dyn_N20")),
            ("N = 200  (natural)", load("results/dyn_N200"))]
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, (title, d) in zip(axes, data):
        if d is None:
            ax.set_title(title + "  (missing)")
            continue
        t = d["t"]
        ax.plot(t, d["perbatch_nll"], color="#9aa3b0", lw=1.8, label="per-batch (UniEnt)")
        ax.plot(t, d["pooled_nll"], color="#3a3f9c", lw=2.4, label="pooled (NOVA)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("adaptation step  t")
        ax.grid(True, color="#e8eaef", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("held-out GMM NLL  (lower = better fit)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.suptitle("Pooling gives a better, more stable detector fit — decisively when per-batch samples are scarce",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig_pooling.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_pooling.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


if __name__ == "__main__":
    for fn in (fig_geometry, fig_ablation, fig_pooling):
        print("wrote", fn())
