"""BPT diagrams of the nuclei fitted with pPXF (results/tables/ppxf_lines.csv): one spectrum per galaxy
(DESI when both DESI and SDSS exist), colour = VC interaction class of the system (class_VC of
dataset/catalogs/interbars_targets.csv), both nuclei of a pair with the same marker, joined by a line
(companions further than 1000 km/s dropped, a pair listed twice drawn once), a ring marks a detected
broad Halpha, error bars from the formal flux errors, no names. Left: [N II] BPT; right: [S II] diagram.
Points need S/N >= 3 in the four lines. `scripts/bpt_comparison.py` draws the same figure over the
contours of comparison samples through `load_sample` / `draw_panel`.

Output: results/figures/bpt_ppxf_sample.{png,pdf}, results/tables/bpt_ppxf_sample.csv (the plotted rows)
Run:  source scripts/fastspecfit_env.sh && python scripts/plot_bpt_ppxf.py
"""
import argparse
from pathlib import Path

import matplotlib
if "ipykernel" not in __import__("sys").modules:      # a notebook keeps its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COL = {"VC 0  pre-interaction": "#c8102e", "VC 1  merger": "#e8871e", "VC 2  post-merger": "#2a9d3f", "no VC class": "#7a7a7a"}
PANELS = {"n2": ("log_nii_ha", "log_nii_ha_err", "ok_n2", "log([N II] λ6584 / Hα)", (-1.3, 0.7)),
          "s2": ("log_sii_ha", "log_sii_ha_err", "ok_s2", "log([S II] λλ6716,31 / Hα)", (-1.2, 0.6))}


def kauffmann03(x): return 0.61 / (x - 0.05) + 1.3
def kewley01(x): return 0.61 / (x - 0.47) + 1.19
def schawinski07(x): return 1.05 * x + 0.45
def kewley06_s2(x): return 0.72 / (x - 0.32) + 1.30
def kewley06_s2_sy(x): return 1.89 * x + 0.76


def load_sample():
    """(pts, pairs): one row per spectrum-galaxy with the plotting columns; pairs = {frozenset(spec_id, spec_id): stage}."""
    t = pd.read_csv(ROOT / "results/tables/ppxf_lines.csv")
    tg = pd.read_csv(ROOT / "dataset/catalogs/interbars_targets.csv").set_index("system")
    sz = pd.read_csv(ROOT / "dataset/catalogs/interbars_specz.csv").set_index("system")
    t["sii_err"] = np.hypot(t.sii6716_err, t.sii6731_err)
    t["log_sii_ha_err"] = np.hypot(t.sii_err / t.sii_flux, t.ha_err / t.ha_flux) / np.log(10)
    t["ok_n2"] = t.min_snr_bpt >= 3
    t["ok_s2"] = t.ok_n2 & (t.sii_flux / t.sii_err >= 3)
    # one spectrum per galaxy (DESI first); companions further than 1000 km/s are not companions; a pair that
    # is two sample rows is keyed by its spectra so it is drawn once
    one = t.sort_values(["role", "source"], ascending=[False, True]).drop_duplicates(["system", "role"], keep="first").copy()
    one = one[~((one.role == "comp") & (one.system.map(sz.dv_comp_kms).abs() > 1000))]
    one["stage"] = one.system.map(tg.class_VC).map({0: "VC 0  pre-interaction", 1: "VC 1  merger", 2: "VC 2  post-merger"}).fillna("no VC class")
    pairs = {}
    for _, g in one.groupby("system"):
        if g.role.nunique() == 2:
            pairs[frozenset((g[g.role == "main"].spec_id.iloc[0], g[g.role == "comp"].spec_id.iloc[0]))] = g[g.role == "main"].stage.iloc[0]
    pts = one.drop_duplicates("spec_id", keep="first")
    return pts, pairs


def demarcations(ax, panel):
    if panel == "n2":
        xx = np.linspace(-1.6, 0.0, 200); ax.plot(xx[xx < 0.05], kauffmann03(xx[xx < 0.05]), "k--", lw=1, label="Kauffmann+03")
        xx = np.linspace(-1.6, 0.4, 200); ax.plot(xx, kewley01(xx), "k-", lw=1, label="Kewley+01")
        xx = np.linspace(-0.18, 0.7, 50); ax.plot(xx, schawinski07(xx), "k:", lw=1, label="Schawinski+07")
        ax.text(-1.2, 0.55, "star forming", fontsize=10); ax.text(-0.2, -1.1, "composite", fontsize=10, rotation=72); ax.text(0.15, 0.95, "Seyfert"); ax.text(0.32, -0.45, "LINER")
    else:
        xx = np.linspace(-1.5, 0.3, 200); ax.plot(xx[xx < 0.32], kewley06_s2(xx[xx < 0.32]), "k-", lw=1, label="Kewley+06")
        xx = np.linspace(-0.3, 0.6, 50); ax.plot(xx, kewley06_s2_sy(xx), "k:", lw=1, label="Seyfert / LINER")
        ax.text(-1.1, 0.55, "star forming", fontsize=10); ax.text(0.1, 0.95, "Seyfert"); ax.text(0.25, -0.5, "LINER")
    xcol, xerr, ok, xlab, xlim = PANELS[panel]
    ax.set(xlim=xlim, ylim=(-1.3, 1.3), xlabel=xlab, ylabel="log([O III] λ5007 / Hβ)")


def draw_panel(ax, pts, pairs, panel, title=True):
    """The sample on one panel: pair lines, points by VC class, broad-line rings. Returns the plotted rows."""
    xcol, xerr, ok, xlab, xlim = PANELS[panel]
    d = pts[pts[ok]].set_index("spec_id")
    for pair, stage in pairs.items():
        a, b = tuple(pair)
        if a in d.index and b in d.index:
            ax.plot([d.loc[a, xcol], d.loc[b, xcol]], [d.loc[a, "log_oiii_hb"], d.loc[b, "log_oiii_hb"]], "-", color=COL[stage], lw=1.2, alpha=0.85, zorder=5)
    for stage, g in d.groupby("stage"):
        ax.errorbar(g[xcol], g.log_oiii_hb, xerr=g[xerr], yerr=g.log_oiii_hb_err, fmt="o", ms=8, mfc=COL[stage], mec="k", mew=0.5,
                    ecolor=COL[stage], elinewidth=0.7, capsize=2, ls="none", zorder=6, label=stage)
    b = d[d.broad_adopted]
    ax.plot(b[xcol], b.log_oiii_hb, "o", ms=15, mfc="none", mec="k", mew=1.0, ls="none", zorder=7, label="broad Hα detected")
    demarcations(ax, panel)
    if title:
        npairs = sum(all(x in d.index for x in pr) for pr in pairs)
        if panel == "n2":
            n = dict(zip(*np.unique(d.bpt_class, return_counts=True)))
            ax.set_title(f"[N II] BPT: {len(d)} nuclei, {npairs} pairs joined  ({', '.join(f'{v} {k}' for k, v in n.items())})", fontsize=10)
        else:
            ax.set_title(f"[S II] diagram: {len(d)} nuclei", fontsize=10)
    return d


def sample_legend(ax, extra=None, **kw):
    h, l = ax.get_legend_handles_labels(); seen = {}
    for hh, ll in zip(h, l): seen.setdefault(ll, hh)
    order = sorted(seen, key=lambda k: (not k.startswith("VC"), k))
    ax.legend([seen[k] for k in order] + (extra or []), order + [e.get_label() for e in (extra or [])], fontsize=7.5, loc=kw.get("loc", "lower left"))


def main():
    argparse.ArgumentParser().parse_args()
    pts, pairs = load_sample()
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    d_n2 = draw_panel(axs[0], pts, pairs, "n2"); draw_panel(axs[1], pts, pairs, "s2")
    sample_legend(axs[0])
    fig.suptitle("interbars nuclei with pPXF — one spectrum per galaxy (DESI where available), colour = VC interaction class of the system, lines join the two nuclei of a pair (|Δv| < 1000 km/s)", fontsize=10.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"results/figures/bpt_ppxf_sample.{ext}", dpi=130)
    out = d_n2.reset_index()[["spec_id", "label", "system", "role", "source", "stage", "log_nii_ha", "log_nii_ha_err", "log_oiii_hb", "log_oiii_hb_err", "log_sii_ha", "bpt_class", "broad_adopted", "ha_flux", "balmer_dec"]]
    out.sort_values(["stage", "system"]).to_csv(ROOT / "results/tables/bpt_ppxf_sample.csv", index=False, float_format="%.4f")
    print(pd.crosstab([d_n2.stage, d_n2.role], d_n2.bpt_class))


if __name__ == "__main__":
    main()
