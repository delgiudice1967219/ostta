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

# Colorblind-safe palette (Okabe-Ito), validated against the data-viz CVD checks:
# the three method hues (vermillion/green/blue) are mutually >=37 Machado-2009
# deltaE (>> 12 target); the gray baseline is an intentional neutral, distinguished
# from the green UniEnt line by its dashed style (their deltaE is in the 8-12 band).
C_BASE   = "#999999"   # BN-adapt: neutral baseline (recessive, dashed)
C_TENT   = "#D55E00"   # vermillion -- the failure mode
C_UNIENT = "#009E73"   # bluish-green -- UniEnt family (UniEnt+ = same hue, dashed)
C_NOVA   = "#0072B2"   # blue -- our method

# Per-method visual identity (consistent across every figure).
STYLE = {
    "bnadapt":     dict(label="BN-adapt",  color=C_BASE,   ls="--", lw=1.6, z=1),
    "tent":        dict(label="Tent",      color=C_TENT,   ls="-",  lw=2.0, z=2),
    "unient":      dict(label="UniEnt",    color=C_UNIENT, ls="-",  lw=1.8, z=3),
    "unient_plus": dict(label="UniEnt+",   color=C_UNIENT, ls="--", lw=1.8, z=3),
    "nova":        dict(label="NOVA",      color=C_NOVA,   ls="-",  lw=2.6, z=5),
}
ORDER = ["bnadapt", "tent", "unient", "unient_plus", "nova"]
ID_C, OOD_C = C_NOVA, C_TENT   # csID (known, blue) / csOOD (novel, vermillion)


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
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def fig_geometry(corruption="gaussian_noise") -> Path:
    """The diagnosis over adaptation step t, as a 2x2 panel that fits one column:
    the norm gap and novel-input alignment (the two factors of the logit), the
    known-class accuracy adaptation recovers, and the detection AUROC it costs.
    Lines are the across-seed mean with a +/-std band. Tent rotates novel features
    onto the known classes and collapses detection even as the norm gap widens and
    accuracy climbs -- the trade-off adaptation imposes; NOVA opens the gap widest
    and detects best at matched accuracy. UniEnt is dropped in favour of UniEnt+
    (near-identical behaviour, fewer lines)."""
    # One representative baseline (UniEnt+, the soft-split analogue of NOVA's rule)
    # instead of both UniEnt variants, which overlap. Panel 3 shows csID accuracy
    # (not novel confidence): paired with the AUROC panel it reads as the
    # accuracy-up/detection-down trade-off Tent imposes.
    order = ["bnadapt", "tent", "unient_plus", "nova"]
    panels = [("norm_gap_l2", "norm gap"),
              ("maxcos_ood",  "novel align."),
              ("acc",         "accuracy (csID)"),
              ("auroc",       "AUROC")]
    if not _traj_bands(corruption, order, "auroc"):
        raise FileNotFoundError(f"no timetrack under {MR}/bench/*_{corruption}_s*/")
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif"})
    # UniEnt+ drawn solid here (not its global dashed style) as requested.
    lsty = lambda m: "-" if m == "unient_plus" else STYLE[m]["ls"]
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.9))
    for ax, (key, ylab) in zip(axes.ravel(), panels):
        bands = _traj_bands(corruption, order, key)
        for m in order:
            if m in bands:
                t, mu, sd = bands[m]
                s = STYLE[m]
                ax.fill_between(t, mu - sd, mu + sd, color=s["color"],
                                alpha=0.18, lw=0, zorder=s["z"])
                ax.plot(t, mu, color=s["color"], ls=lsty(m), lw=s["lw"], zorder=s["z"])
        ax.set_xlabel("step  t", fontsize=8)
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(True, color="#e8eaef", lw=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.02)
        ax.tick_params(labelsize=7)
    handles = [plt.Line2D([0], [0], color=STYLE[m]["color"], ls=lsty(m),
                          lw=STYLE[m]["lw"], label=STYLE[m]["label"]) for m in order]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.03), fontsize=6.6, columnspacing=1.0,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _finish(fig, "fig_geometry")


def fig_confidence(corruption="gaussian_noise") -> Path:
    """Mean softmax confidence on novel inputs over adaptation: the inflation
    the body's diagnosis cites (Tent 0.69->0.87), the effect of the rotation
    shown in fig_geometry's alignment panel. Rendered as a single panel sized
    for its subfigure slot in the merged diagnostics figure (Fig. 8), using the
    global per-method styles so panel (a)'s legend keys it too. Tent inflates;
    UniEnt+ holds flat; NOVA pushes below the un-adapted baseline."""
    order = ["bnadapt", "tent", "unient_plus", "nova"]
    bands = _traj_bands(corruption, order, "conf_ood")
    if not bands:
        raise FileNotFoundError(f"no timetrack under {MR}/bench/*_{corruption}_s*/")
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    for m in order:
        if m in bands:
            t, mu, sd = bands[m]
            s = STYLE[m]
            ax.fill_between(t, mu - sd, mu + sd, color=s["color"],
                            alpha=0.18, lw=0, zorder=s["z"])
            ax.plot(t, mu, color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["z"])
    ax.set_xlabel("step  t")
    ax.set_ylabel("novel confidence")
    ax.grid(True, color="#e8eaef", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(x=0.02)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    return _finish(fig, "fig_confidence")


def fig_ablation() -> Path:
    """Lever-causal headline across 15 corruptions x 3 seeds, with seed error bars.

    Tent (no detector, adapts on all inputs) is the leftmost bar, set off by a gap
    and a lighter face, then the three lever variants that share NOVA's detector."""
    rows = _load_summaries("bench") + _load_summaries("lever")
    # (method key, x-position, label, colour). A gap after Tent (x=0 -> next at 1.4)
    # signals Tent is the no-detector reference, not a lever variant.
    spec = [("tent",           0.0, "Tent\n(no detector)",              C_TENT),
            ("det_only",       1.4, "none",                              C_BASE),
            ("novadet_entmax", 2.4, "entropy-max",                       C_UNIENT),
            ("nova",           3.4, "feature-norm\nsuppression\n(NOVA)", C_NOVA)]
    xs, vals, errs, cols, labels = [], [], [], [], []
    for m, x, lab, col in spec:
        mean, sd = _seed_stable([r for r in rows if r["method"] == m])
        xs.append(x); vals.append(mean); errs.append(sd); cols.append(col); labels.append(lab)

    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(xs, vals, yerr=errs, color=cols, width=0.82, zorder=3,
                  capsize=4, error_kw=dict(lw=1.3, ecolor="#333"))
    bars[0].set_alpha(0.55)                       # Tent set off as the reference
    bars[0].set_hatch("//")
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, v + e + 0.004, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold")
    # Divider between the Tent reference and the three detector-sharing variants.
    ax.axvline(0.7, color="#cccccc", ls=":", lw=1.0, zorder=1)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
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

    # figwidth chosen so that, scaled to its ~0.55\linewidth slot in the merged
    # Fig. 7, tick/label fonts render at ~7pt -- matching the absorption and
    # score-density panels it sits beside (consistent fonts across the group).
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.4))
    rowspec = [("energy", "energy  (higher = novel)"),
               ("norm",   "feature norm")]
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


def fig_tsne(corruption="gaussian_noise", n=1500, seed=0, perplexity=40) -> Path:
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
    for ax, (title, key, fid, food, yid) in zip(axes.ravel(), panels):
        i_id = rng.choice(len(fid), min(n, len(fid)), replace=False)
        i_od = rng.choice(len(food), min(n, len(food)), replace=False)
        fid_s, yid_s, food_s = fid[i_id], yid[i_id], food[i_od]
        X = np.concatenate([fid_s, food_s], 0).astype(np.float32)
        # Faithful to UniEnt Fig. 6: one t-SNE per method; csID red->blue by class
        # (radial fingers), csOOD one yellow mass. Euclidean t-SNE keeps norm
        # information, so NOVA's norm-shrunk csOOD should collapse toward a compact
        # blob -- the separation, if it shows, shows here.
        Z = TSNE(n_components=2, perplexity=perplexity, init="pca",
                 learning_rate="auto", random_state=seed).fit_transform(X)
        n_id = len(fid_s)
        ax.scatter(Z[n_id:, 0], Z[n_id:, 1], s=8, c="#f2c200", marker="o",
                   alpha=0.5, linewidths=0, label="csOOD (novel)", zorder=1)
        ax.scatter(Z[:n_id, 0], Z[:n_id, 1], s=8, c=yid_s, cmap="RdYlBu",
                   vmin=0, vmax=9, alpha=0.9, linewidths=0, zorder=2)
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
    detection rule reads already separates csID from csOOD before any adaptation."""
    hits = glob.glob(f"{MR}/absorb/*_{corruption}_s0/dump.npz")
    if not hits:
        raise FileNotFoundError(f"no dump.npz under {MR}/absorb/*_{corruption}_s0/")
    d = dict(np.load(hits[0]))
    id_s, ood_s = d["maxcos_id_t0"], d["maxcos_ood_t0"]

    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    lo = float(min(id_s.min(), ood_s.min())); hi = float(max(id_s.max(), ood_s.max()))
    bins = np.linspace(lo, hi, 40)
    ax.hist(id_s, bins=bins, color=ID_C, alpha=0.55, label="known (csID)", density=True)
    ax.hist(ood_s, bins=bins, color=OOD_C, alpha=0.55, label="novel (csOOD)", density=True)

    ax.set_xlabel("frozen max-cosine score  $s(x)$")
    ax.set_yticks([])
    ax.grid(True, axis="x", color="#eef0f4", lw=0.7)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
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
        ax.plot(t, d["perbatch_nll"], color=C_BASE, lw=1.8, label="per-batch fit (UniEnt)")
        ax.plot(t, d["pooled_nll"], color=C_NOVA, lw=2.4, label="pooled fit (NOVA)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("adaptation step  t")
        ax.grid(True, color="#e8eaef", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("held-out fit quality\n(GMM NLL, lower = better)")
    axes[0].legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    return _finish(fig, "fig_pooling")


def _lambda_curve(dirs, method_name):
    """{lam: {metric:(mean, seed-stable std)}} for one method across dirs.

    Aggregates seed-stably (mean over the 15 corruptions per seed, then mean/std
    over the 3 seeds) so the error bar is the seed stability reported in Table 1,
    not the much larger across-corruption spread."""
    # lam -> seed -> metric -> [per-corruption values]
    perseed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in dirs:
        for sj in glob.glob(f"{MR}/{d}/**/summary.json", recursive=True):
            p = Path(sj); cfgp = p.parent / ".hydra" / "config.yaml"
            if not cfgp.exists():
                continue
            c = yaml.safe_load(cfgp.read_text()); m = c.get("method", {})
            if m.get("name") != method_name or m.get("lam") is None:
                continue
            s = json.loads(p.read_text())
            if "auroc" not in s:
                continue
            lam = float(m["lam"]); seed = c.get("seed")
            for k in ("acc", "auroc", "oscr"):
                perseed[lam][seed][k].append(s[k]["tT"])
    out = {}
    for lam, seeds in perseed.items():
        out[lam] = {}
        for k in ("acc", "auroc", "oscr"):
            means = [float(np.mean(seeds[sd][k])) for sd in seeds if seeds[sd][k]]
            out[lam][k] = (float(np.mean(means)),
                           float(np.std(means)) if len(means) > 1 else 0.0)
    return out


def fig_lambda() -> Path:
    """NOVA's penalty strength lambda trades csID accuracy for detection AUROC
    (all 15 corruptions x 3 seeds, seed-stable error bars). Twin y-axes."""
    cur = _lambda_curve(["bench", "nova_lambda"], "nova")
    lams = sorted(cur)
    acc  = [cur[l]["acc"][0] * 100 for l in lams]
    accs = [cur[l]["acc"][1] * 100 for l in lams]
    au   = [cur[l]["auroc"][0] * 100 for l in lams]
    aus  = [cur[l]["auroc"][1] * 100 for l in lams]
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax1 = plt.subplots(figsize=(4.6, 3.2))
    ax2 = ax1.twinx()
    ax1.errorbar(lams, acc, yerr=accs, color=ID_C, marker="o", lw=2.0, capsize=3,
                 label="csID accuracy")
    ax2.errorbar(lams, au, yerr=aus, color=OOD_C, marker="s", lw=2.0, capsize=3,
                 label="detection AUROC")
    # Zoomed-out ranges so the (small) seed-stable error bars read as small.
    ax1.set_ylim(81, 87); ax2.set_ylim(86, 99)
    ax1.set_xlabel("penalty strength  $\\lambda$")
    ax1.set_ylabel("csID accuracy", color=ID_C)
    ax2.set_ylabel("detection AUROC", color=OOD_C)
    ax1.tick_params(axis="y", labelcolor=ID_C)
    ax2.tick_params(axis="y", labelcolor=OOD_C)
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    ax1.grid(True, axis="x", color="#eef0f4", lw=0.7)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _finish(fig, "fig_lambda")


def fig_tradeoff() -> Path:
    """csID accuracy vs detection AUROC as lambda is swept, for the two frozen
    scores (all 15 corruptions x 3 seeds). Max-cosine's frontier dominates the
    entropy score's -- higher AUROC at every matched accuracy, and it reaches
    accuracies the entropy split cannot."""
    nova = _lambda_curve(["bench", "nova_lambda"], "nova")
    ent  = _lambda_curve(["entropy_lambda", "rule_swap"], "entropy_l1")
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    for cur, color, marker, label, dy in [
            (nova, C_NOVA,   "o", "max-cosine (NOVA)",  6),
            (ent,  C_UNIENT, "s", "entropy score",     -13)]:
        lams = sorted(cur)
        xs = [cur[l]["acc"][0] * 100 for l in lams]
        ys = [cur[l]["auroc"][0] * 100 for l in lams]
        ax.plot(xs, ys, color=color, marker=marker, lw=2.2, ms=6, label=label, zorder=3)
        for l, x, y in zip(lams, xs, ys):
            ax.annotate(f"$\\lambda{{=}}{l:g}$", (x, y), textcoords="offset points",
                        xytext=(4, dy), fontsize=6.6, color=color)
    ax.axvline(83.3, color="#cccccc", ls=":", lw=0.9, zorder=1)
    ax.set_xlim(82.2, 85.0); ax.set_ylim(86.5, 96.5)
    ax.set_xlabel("csID accuracy")
    ax.set_ylabel("detection AUROC")
    ax.grid(True, color="#eef0f4", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    return _finish(fig, "fig_tradeoff")


if __name__ == "__main__":
    for fn in (fig_geometry, fig_confidence, fig_ablation, fig_distributions, fig_openness,
               fig_pooling, fig_tsne, fig_absorption, fig_score_density,
               fig_lambda, fig_tradeoff):
        try:
            print("wrote", fn())
        except Exception as e:  # keep going; report which figure failed
            print(f"FAILED {fn.__name__}: {e}")
