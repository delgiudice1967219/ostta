"""Paper figures rendered from saved results (no GPU, no re-running).

Each function loads saved ``.npz`` / ``summary.json`` artifacts under ``multirun/``
and writes a figure to ``paper/figures/``. Run ``python src/figures.py`` to render
all of them. All trajectory/aggregate data comes from the seeded multi-seed runs
(``multirun/{bench,lever,openness,dist}``) so every figure shares one provenance.
"""
from __future__ import annotations

import csv
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

FIG_DIR = Path("paper/figures")
MR = "multirun"

# Per-method visual identity (consistent across every figure).
STYLE = {
    "bnadapt":     dict(label="BN-adapt",  color="#9aa3b0", ls="--", lw=1.6, z=1),
    "tent":        dict(label="Tent",      color="#c0392b", ls="-",  lw=2.0, z=2),
    "unient":      dict(label="UniEnt",    color="#d98a2b", ls="-",  lw=1.8, z=3),
    "unient_plus": dict(label="UniEnt+",   color="#e0b066", ls="--", lw=1.8, z=3),
    "nova":        dict(label="NOVA",      color="#3a3f9c", ls="-",  lw=2.6, z=5),
}
ORDER = ["bnadapt", "tent", "unient", "unient_plus", "nova"]
ID_C, OOD_C = "#3a6ea5", "#c0392b"   # csID / csOOD histogram colours


# --------------------------------------------------------------------------- #
#  loaders
# --------------------------------------------------------------------------- #
def _load_summaries(subdir: str) -> list[dict]:
    """Flat rows {method, corruption, seed, alpha, auroc, acc} from a multirun dir."""
    rows = []
    for sj in glob.glob(f"{MR}/{subdir}/**/summary.json", recursive=True):
        p = Path(sj)
        cfgp = p.parent / ".hydra" / "config.yaml"
        if not cfgp.exists():
            continue
        s = json.loads(p.read_text())
        c = yaml.safe_load(cfgp.read_text())
        m = c.get("method", {})
        if "auroc" not in s:          # skip continual (different schema)
            continue
        rows.append(dict(method=m.get("name"), corruption=c.get("corruption"),
                         seed=c.get("seed"), alpha=c.get("alpha"),
                         auroc=s["auroc"]["tT"], acc=s["acc"]["tT"]))
    return rows


def _seed_stable(rows, metric="auroc"):
    """Mean over corruptions per seed -> (mean, std) across seeds. Seed-stable error bar."""
    g = defaultdict(list)
    for r in rows:
        if r[metric] is not None:
            g[r["seed"]].append(r[metric])
    sm = [st.fmean(v) for v in g.values()]
    return st.fmean(sm), (st.stdev(sm) if len(sm) > 1 else 0.0)


def _traj(corruption="gaussian_noise", seed=0) -> dict:
    """method -> timetrack arrays from the seeded bench runs (per-corruption protocol)."""
    out = {}
    for m in ORDER:
        hits = glob.glob(f"{MR}/bench/{m}_{corruption}_s{seed}/timetrack.npz")
        if hits:
            out[m] = dict(np.load(hits[0]))
    return out


def _traj_bands(corruption, methods, key):
    """method -> (t, mean, std) for one metric, aggregated over all available seeds.

    Loads every ``{method}_{corruption}_s*`` timetrack, stacks the per-seed arrays
    on the shared ``t`` grid, and returns the across-seed mean and std so a figure
    can draw a confidence band. Methods/seeds with no artifact are skipped.
    """
    out = {}
    for m in methods:
        hits = sorted(glob.glob(f"{MR}/bench/{m}_{corruption}_s*/timetrack.npz"))
        arrs, t_ref = [], None
        for h in hits:
            d = dict(np.load(h))
            if key not in d:
                continue
            arrs.append(d[key]); t_ref = d["t"]
        if arrs:
            L = min(len(a) for a in arrs)                 # guard ragged lengths
            stack = np.stack([a[:L] for a in arrs])
            out[m] = (t_ref[:L], stack.mean(0), stack.std(0))
    return out


def _finish(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def fig_geometry(corruption="gaussian_noise") -> Path:
    """The diagnosis over adaptation step t, as a 2x2 panel that fits one column:
    the norm gap and novel-input alignment (the two factors of the logit), the
    novel-input confidence they induce, and the outcome (detection AUROC). Lines
    are the across-seed mean with a +/-std band. Tent rotates novel features onto
    the known classes and inflates their confidence, collapsing detection even as
    the norm gap widens; NOVA opens the gap widest and detects best. UniEnt is
    dropped in favour of UniEnt+ (near-identical behaviour, fewer lines)."""
    # One representative baseline (UniEnt+, the soft-split analogue of NOVA's rule)
    # instead of both UniEnt variants, which overlap.
    order = ["bnadapt", "tent", "unient_plus", "nova"]
    panels = [("norm_gap_l2", "norm gap"),
              ("maxcos_ood",  "novel align."),
              ("conf_ood",    "novel conf."),
              ("auroc",       "AUROC")]
    if not _traj_bands(corruption, order, "auroc"):
        raise FileNotFoundError(f"no timetrack under {MR}/bench/*_{corruption}_s*/")
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.9))
    for ax, (key, ylab) in zip(axes.ravel(), panels):
        bands = _traj_bands(corruption, order, key)
        for m in order:
            if m in bands:
                t, mu, sd = bands[m]
                s = STYLE[m]
                ax.fill_between(t, mu - sd, mu + sd, color=s["color"],
                                alpha=0.18, lw=0, zorder=s["z"])
                ax.plot(t, mu, color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["z"])
        ax.set_xlabel("step  t", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(True, color="#e8eaef", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.02)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                          lw=STYLE[m]["lw"], label=STYLE[m]["label"]) for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=6.6, columnspacing=1.0,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _finish(fig, "fig_geometry")


def fig_ablation() -> Path:
    """Lever-causal headline across 15 corruptions x 3 seeds, with seed error bars."""
    rows = _load_summaries("bench") + _load_summaries("lever")
    lever = [("det_only", "none", "#9aa3b0"),
             ("novadet_entmax", "entropy-max", "#d98a2b"),
             ("nova", "feature-norm\nsuppression\n(NOVA)", "#3a3f9c")]
    names, vals, errs, cols = [], [], [], []
    for m, lab, col in lever:
        mean, sd = _seed_stable([r for r in rows if r["method"] == m])
        names.append(lab); vals.append(mean); errs.append(sd); cols.append(col)
    tent_mean, _ = _seed_stable([r for r in rows if r["method"] == "tent"])

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    bars = ax.bar(names, vals, yerr=errs, color=cols, width=0.62, zorder=3,
                  capsize=4, error_kw=dict(lw=1.3, ecolor="#333"))
    # Tent reference (no detector, no split): a labelled band + line, drawn well
    # inside the frame so it reads clearly against the three lever bars.
    ax.axhspan(0.58, tent_mean, color="#c0392b", alpha=0.06, zorder=0)
    ax.axhline(tent_mean, color="#c0392b", ls="--", lw=2.0, zorder=2)
    ax.text(len(names) - 0.5, tent_mean + 0.004,
            f"Tent (no detector)  {tent_mean:.3f}", ha="right", va="bottom",
            fontsize=9.5, color="#c0392b", fontweight="bold")
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    ax.set_ylim(0.58, 0.94)
    ax.set_ylabel("novelty-detection AUROC")
    ax.grid(True, axis="y", color="#e8eaef", lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return _finish(fig, "fig_ablation")


def fig_distributions(corruption="gaussian_noise") -> Path:
    """Per-sample energy & feature-norm of known vs novel inputs, Tent vs NOVA (t=T)."""
    def load(m):
        hits = glob.glob(f"{MR}/dist/{m}_{corruption}/dump.npz")
        return dict(np.load(hits[0])) if hits else None
    dump = {m: load(m) for m in ("tent", "nova")}
    if not all(dump.values()):
        raise FileNotFoundError(f"missing dist dumps for {corruption}")

    plt.rcParams.update({"font.size": 10.5, "font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.0))
    rowspec = [("energy", "energy  −logΣexp(logits)  (higher ⇒ novel)"),
               ("norm",   "feature-norm  ‖g(x)‖₂")]
    for i, (q, ylab) in enumerate(rowspec):
        for j, m in enumerate(("tent", "nova")):
            ax = axes[i][j]
            d = dump[m]
            idv, oodv = d[f"{q}_id_tT"], d[f"{q}_ood_tT"]
            lo = float(min(idv.min(), oodv.min())); hi = float(max(idv.max(), oodv.max()))
            bins = np.linspace(lo, hi, 40)
            ax.hist(idv, bins=bins, color=ID_C, alpha=0.6, label="known (csID)", density=True)
            ax.hist(oodv, bins=bins, color=OOD_C, alpha=0.6, label="novel (csOOD)", density=True)
            if i == 0:
                # Neutral title colour: the bar colours already key csID/csOOD,
                # so a coloured method title would clash with the population legend.
                ax.set_title(f"under {STYLE[m]['label']}", fontsize=11.5,
                             color="#222", fontweight="bold")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9.8)
            ax.set_yticks([])
            ax.grid(True, axis="x", color="#eef0f4", lw=0.7)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _finish(fig, "fig_distributions")


def fig_openness() -> Path:
    """Detection AUROC vs openness alpha for NOVA and UniEnt (subset corruptions, 3 seeds)."""
    rows = _load_summaries("openness")
    alphas = sorted({r["alpha"] for r in rows if r["alpha"] is not None})
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for m in ("unient", "nova"):
        ys, es = [], []
        for a in alphas:
            mean, sd = _seed_stable([r for r in rows if r["method"] == m and r["alpha"] == a])
            ys.append(mean); es.append(sd)
        s = STYLE[m]
        ax.errorbar(alphas, ys, yerr=es, color=s["color"], lw=s["lw"], marker="o",
                    capsize=3, label=s["label"])
    ax.set_xlabel("openness  α  (fraction of novel inputs per batch)")
    ax.set_ylabel("novelty-detection AUROC")
    ax.set_xticks(alphas)
    ax.grid(True, color="#e8eaef", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return _finish(fig, "fig_openness")


def fig_tsne(corruption="gaussian_noise", n=900, seed=0) -> Path:
    """t-SNE of the latent space (UniEnt Fig. 6 style): csID by class, csOOD yellow.

    Four panels --- BN-adapt (source), Tent, UniEnt, NOVA --- each a separate t-SNE
    of the held-out diagnostic features (theta_T, except BN-adapt = theta_0).
    """
    from sklearn.manifold import TSNE

    def load(m):
        hits = glob.glob(f"{MR}/tsne/{m}/dump.npz")
        return dict(np.load(hits[0])) if hits else None
    nova, tent, unient = load("nova"), load("tent"), load("unient")
    if not all([nova, tent, unient]):
        raise FileNotFoundError(f"missing t-SNE dumps under {MR}/tsne/")

    # (title, method-key, csID feats, csOOD feats, csID labels)
    panels = [
        ("BN-adapt (t=0)", "bnadapt", nova["feat_id_t0"], nova["feat_ood_t0"], nova["y_id"]),
        ("Tent",              "tent",    tent["feat_id_tT"], tent["feat_ood_tT"], tent["y_id"]),
        ("UniEnt",            "unient",  unient["feat_id_tT"], unient["feat_ood_tT"], unient["y_id"]),
        ("NOVA",              "nova",    nova["feat_id_tT"], nova["feat_ood_tT"], nova["y_id"]),
    ]
    rng = np.random.default_rng(seed)
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    sc = None
    for ax, (title, key, fid, food, yid) in zip(axes.ravel(), panels):
        i_id = rng.choice(len(fid), min(n, len(fid)), replace=False)
        i_od = rng.choice(len(food), min(n, len(food)), replace=False)
        fid_s, yid_s, food_s = fid[i_id], yid[i_id], food[i_od]
        X = np.concatenate([fid_s, food_s], 0).astype(np.float32)
        Z = TSNE(n_components=2, perplexity=30, init="pca",
                 learning_rate="auto", random_state=seed).fit_transform(X)
        n_id = len(fid_s)
        ax.scatter(Z[n_id:, 0], Z[n_id:, 1], s=7, c="#f2c200", marker="o",
                   alpha=0.55, linewidths=0, label="csOOD (novel)")
        sc = ax.scatter(Z[:n_id, 0], Z[:n_id, 1], s=7, c=yid_s, cmap="coolwarm",
                        vmin=0, vmax=9, alpha=0.85, linewidths=0)
        ax.set_title(title, fontsize=12.5, fontweight="bold", color=STYLE[key]["color"])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
    axes.ravel()[0].legend(loc="upper right", fontsize=8.5, frameon=True,
                           markerscale=1.5)
    fig.tight_layout()
    return _finish(fig, "fig_tsne")


def fig_absorption(corruption="gaussian_noise") -> Path:
    """Relabelling & absorption diagnostics over adaptation (App. figure).

    Two panels from the absorb reruns: the csOOD cosine to each input's
    theta_0-predicted class (falls under Tent while max-cosine rises = features
    rotate onto *other* prototypes: relabelling) and the csOOD distance to the
    nearest frozen clean-CIFAR centroid (falls under Tent = absorption into the
    source classes; NOVA halts it)."""
    traj = {}
    for m in ORDER:
        hits = glob.glob(f"{MR}/absorb/{m}_{corruption}_s0/timetrack.npz")
        if hits:
            traj[m] = dict(np.load(hits[0]))
    if not traj:
        raise FileNotFoundError(f"no timetrack under {MR}/absorb/*_{corruption}_s0/")
    panels = [("srccos_ood", "novel src-cos"),
              ("cdist_ood",  "novel cdist")]
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.55))
    for ax, (key, ylab) in zip(axes, panels):
        for m in ORDER:
            if m in traj and key in traj[m]:
                s = STYLE[m]
                ax.plot(traj[m]["t"], traj[m][key], color=s["color"], ls=s["ls"],
                        lw=s["lw"], zorder=s["z"])
        ax.set_xlabel("step  t", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.grid(True, color="#e8eaef", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.02)
        ax.tick_params(labelsize=7.5)
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                          lw=STYLE[m]["lw"], label=STYLE[m]["label"]) for m in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=6.6, columnspacing=1.0,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _finish(fig, "fig_absorption")


def fig_score_density(corruption="gaussian_noise") -> Path:
    """Frozen max-cosine score densities at t=0 (App. figure): the score NOVA's
    detection rule reads already separates csID from csOOD before any adaptation
    -- the observation that motivates the score choice."""
    hits = glob.glob(f"{MR}/absorb/*_{corruption}_s0/dump.npz")
    if not hits:
        raise FileNotFoundError(f"no dump.npz under {MR}/absorb/*_{corruption}_s0/")
    d = dict(np.load(hits[0]))
    id_s, ood_s = d["maxcos_id_t0"], d["maxcos_ood_t0"]

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    lo = float(min(id_s.min(), ood_s.min())); hi = float(max(id_s.max(), ood_s.max()))
    bins = np.linspace(lo, hi, 40)
    ax.hist(id_s, bins=bins, color=ID_C, alpha=0.6, label="known (csID)", density=True)
    ax.hist(ood_s, bins=bins, color=OOD_C, alpha=0.6, label="novel (csOOD)", density=True)
    ax.axvline(float(np.mean(id_s)), color=ID_C, ls=":", lw=1.2)
    ax.axvline(float(np.mean(ood_s)), color=OOD_C, ls=":", lw=1.2)
    ax.set_xlabel("frozen max-cosine score  $s(x)$")
    ax.set_yticks([])
    ax.grid(True, axis="x", color="#eef0f4", lw=0.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return _finish(fig, "fig_score_density")


def fig_pooling() -> Path:
    """Pooling dynamics: held-out GMM NLL, pooled vs per-batch, at N=20 and N=200."""
    def load(dirpat):
        hits = glob.glob(f"{dirpat}/**/posterior_quality.npz", recursive=True)
        return dict(np.load(hits[0])) if hits else None
    data = [("N = 20  (few samples per batch)", load("results/dyn_N20")),
            ("N = 200  (default)", load("results/dyn_N200"))]
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, (title, d) in zip(axes, data):
        if d is None:
            ax.set_title(title + "  (missing)"); continue
        t = d["t"]
        ax.plot(t, d["perbatch_nll"], color="#9aa3b0", lw=1.8, label="per-batch fit (UniEnt)")
        ax.plot(t, d["pooled_nll"], color="#3a3f9c", lw=2.4, label="pooled fit (NOVA)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("adaptation step  t")
        ax.grid(True, color="#e8eaef", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("held-out fit quality\n(GMM NLL, lower = better)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    return _finish(fig, "fig_pooling")


if __name__ == "__main__":
    for fn in (fig_geometry, fig_ablation, fig_distributions, fig_openness,
               fig_pooling, fig_tsne, fig_absorption, fig_score_density):
        try:
            print("wrote", fn())
        except Exception as e:  # keep going; report which figure failed
            print(f"FAILED {fn.__name__}: {e}")
