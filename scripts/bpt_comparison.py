"""The sample's BPT diagram over the contours of large comparison samples.

Comparison samples (cached in output/comparison/ by the download cells of 2026-09-28):
  sdss   MPA-JHU line fluxes of SDSS DR8 galaxies at 0.005 < z < 0.1 (Data Lab sdss_dr8.galspecline +
         galspecinfo; Brinchmann+2004 / Tremonti+2004 measurements, 3" fibres)
  desi   DESI DR1 stellar_mass_emline VAC at 0.005 < z < 0.1 (Data Lab; 1.5" fibres)
  pairs  SDSS spectroscopic pairs built here from the MPA-JHU sample: a companion within --pair-kpc
         projected and --pair-dv km/s (Ellison+2008-style, without the mass ratio cut)
  bars   Nair & Abraham (2010, ApJS 186, 427; VizieR J/ApJS/186/427) visual bar flags (Bar bits 2, 4, 8:
         strong / intermediate / weak) matched to MPA-JHU by position (< 2"); 'napair' = their Pair flag != 0
All comparison galaxies need S/N > 3 in the four lines of the panel; contours enclose 20 / 50 / 80 % of
each sample (2D histogram, 0.04 dex bins, Gaussian-smoothed).

Output: results/figures/bpt_ppxf_comparison.{png,pdf}, results/manifests/<run_id>.json
Run:  source scripts/fastspecfit_env.sh && python scripts/bpt_comparison.py [--pair-kpc 30] [--pair-dv 500]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import matplotlib
if "ipykernel" not in __import__("sys").modules:      # a notebook keeps its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_bpt_ppxf import load_sample, draw_panel, sample_legend   # noqa: E402

CMP = ROOT / "output" / "comparison"
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
C_KMS = 299792.458
LEVELS = (0.2, 0.5, 0.8)


def load_sdss():
    f = CMP / "sdss_dr8_galspec_z0p1.csv"
    d = pd.read_csv(f if f.exists() else f.with_suffix(".csv.gz"))          # the student bundle ships the gzipped copy
    d = d.rename(columns={"h_alpha_flux": "ha", "h_alpha_flux_err": "ha_err", "h_beta_flux": "hb", "h_beta_flux_err": "hb_err",
                          "nii_6584_flux": "nii", "nii_6584_flux_err": "nii_err", "oiii_5007_flux": "oiii", "oiii_5007_flux_err": "oiii_err"})
    d["sii"] = d.sii_6717_flux + d.sii_6731_flux
    return d


def load_desi():
    d = pd.read_csv(CMP / "desi_dr1_emline_z0p1.csv")
    d = d.rename(columns={"halpha_flux": "ha", "halpha_fluxerr": "ha_err", "hbeta_flux": "hb", "hbeta_fluxerr": "hb_err",
                          "nii6583_flux": "nii", "nii6583_fluxerr": "nii_err", "oiii5007_flux": "oiii", "oiii5007_fluxerr": "oiii_err",
                          "target_ra": "ra", "target_dec": "dec"})
    d["sii"] = d.sii6716_flux + d.sii6731_flux
    return d


def ratios(d):
    ok = (d.ha / d.ha_err > 3) & (d.hb / d.hb_err > 3) & (d.nii / d.nii_err > 3) & (d.oiii / d.oiii_err > 3)
    d = d[ok].copy()
    d["n2"] = np.log10(d.nii / d.ha); d["o3"] = np.log10(d.oiii / d.hb)
    d["s2"] = np.where(d.sii > 0, np.log10(d.sii.clip(lower=1e-9) / d.ha), np.nan)
    return d


def sdss_pairs(d, kpc, dv):
    """Galaxies of the MPA-JHU sample with a spectroscopic companion within kpc (projected) and dv (km/s)."""
    kpc_per_deg = 3600 / COSMO.arcsec_per_kpc_proper(d.z.values).value
    ra, dec, zv = np.radians(d.ra.values), np.radians(d.dec.values), d.z.values
    xyz = np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    tree = cKDTree(xyz)
    rmax = np.radians(kpc / kpc_per_deg.min())          # generous search (nearest z), refined per galaxy below
    pairs = tree.query_pairs(r=2 * np.sin(rmax / 2), output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    sep_deg = np.degrees(2 * np.arcsin(0.5 * np.linalg.norm(xyz[i] - xyz[j], axis=1)))
    close = (sep_deg * np.minimum(kpc_per_deg[i], kpc_per_deg[j]) < kpc) & (np.abs(C_KMS * (zv[i] - zv[j]) / (1 + zv[i])) < dv)
    has = np.zeros(len(d), bool); has[i[close]] = True; has[j[close]] = True
    return has


def nair_match(d):
    """Nair & Abraham flags attached to the MPA-JHU galaxies by position (< 2")."""
    n = pd.read_csv(CMP / "nair_abraham_2010.csv")
    tree = cKDTree(np.c_[n._RA.values, n._DE.values * 1.0])
    # small-angle match in degrees with the RA stretched by cos(dec)
    q = np.c_[d.ra.values, d.dec.values]
    dist, idx = tree.query(np.c_[q[:, 0], q[:, 1]], distance_upper_bound=2 / 3600 / np.cos(np.radians(d.dec.values)).min())
    ok = np.isfinite(dist)
    sep = np.full(len(d), np.inf)
    sep[ok] = np.hypot((d.ra.values[ok] - n._RA.values[idx[ok]]) * np.cos(np.radians(d.dec.values[ok])), d.dec.values[ok] - n._DE.values[idx[ok]]) * 3600
    ok = sep < 2
    bar = np.zeros(len(d), bool); pair = np.zeros(len(d), bool); inter = np.zeros(len(d), bool); innair = np.zeros(len(d), bool)
    bar[ok] = (n.Bar.values[idx[ok]].astype(int) & (2 | 4 | 8)) > 0
    pair[ok] = n.Pair.values[idx[ok]].astype(int) > 0
    inter[ok] = n.Int.values[idx[ok]].astype(int) > 0
    innair[ok] = True
    return innair, bar, pair, inter


def contour_levels(x, y, xlim, ylim, bins=0.04, smooth=1.2):
    ok = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=[np.arange(xlim[0], xlim[1] + bins, bins), np.arange(ylim[0], ylim[1] + bins, bins)])
    H = gaussian_filter(H, smooth)
    srt = np.sort(H.ravel())[::-1]; cum = np.cumsum(srt) / srt.sum()
    levels = [srt[np.searchsorted(cum, f)] for f in LEVELS[::-1]]       # enclosing 80, 50, 20 %
    return (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, H.T, levels


# ------------------------------------------------------------------ which stage looks like which population?
def density_grid(x, y, xlim=(-1.6, 0.9), ylim=(-1.5, 1.5), bins=0.04, smooth=1.5):
    """Smoothed, normalised 2D density of a population on a fixed grid (probability per bin)."""
    ok = np.isfinite(x) & np.isfinite(y)
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=[np.arange(xlim[0], xlim[1] + bins, bins), np.arange(ylim[0], ylim[1] + bins, bins)])
    H = gaussian_filter(H, smooth); H = H / H.sum()
    return xe, ye, H


def eval_density(grid, x, y):
    xe, ye, H = grid
    i = np.clip(np.searchsorted(xe, x) - 1, 0, H.shape[0] - 1); j = np.clip(np.searchsorted(ye, y) - 1, 0, H.shape[1] - 1)
    return H[i, j]


def enclosed_fraction(grid, x, y, frac):
    """Fraction of the points inside the contour that encloses `frac` of the population."""
    xe, ye, H = grid
    srt = np.sort(H.ravel())[::-1]; thr = srt[np.searchsorted(np.cumsum(srt), frac)]
    return float(np.mean(eval_density(grid, x, y) >= thr))


def energy_distance_2d(A, B):
    from scipy.spatial.distance import cdist
    return 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()


def stage_consistency(pts, pops, nboot=500, nperm=300, seed=1):
    """Per VC stage and population: fraction of nuclei inside the 50 % and 80 % contours, mean log-density of
    the nuclei under the population (bootstrap over the nuclei), and an energy-distance permutation test
    of the nuclei against a random subsample of the population (p = fraction of permutations at least as
    far apart). Also the non-star-forming fraction (above Kauffmann+03) of each."""
    from plot_bpt_ppxf import kauffmann03
    rng = np.random.default_rng(seed)
    d = pts[pts.ok_n2]
    grids = {k: density_grid(v.n2.values, v.o3.values) for k, v in pops.items()}
    rows = []
    groups = [(st, g) for st, g in d.groupby("stage") if st != "no VC class"] + [("all stages", d[d.stage != "no VC class"])]
    for stage, g in groups:
        x, y = g.log_nii_ha.values, g.log_oiii_hb.values
        nonsf = float(np.mean((x >= 0.05) | (y >= kauffmann03(np.minimum(x, 0.049)))))
        for pop, P in pops.items():
            grid = grids[pop]
            dens = np.log10(np.clip(eval_density(grid, x, y), 1e-12, None))
            boots = [np.mean(rng.choice(dens, len(dens))) for _ in range(nboot)]
            sub = P.sample(min(len(P), 1500), random_state=seed)[["n2", "o3"]].values
            A = np.c_[x, y]; e_obs = energy_distance_2d(A, sub)
            pool = np.vstack([A, sub]); e_perm = []
            for _ in range(nperm):
                rng.shuffle(pool); e_perm.append(energy_distance_2d(pool[:len(A)], pool[len(A):]))
            pnonsf = P.n2.values; qnonsf = P.o3.values
            rows.append(dict(stage=stage, n=len(g), population=pop, n_pop=len(P),
                             frac_in50=enclosed_fraction(grid, x, y, 0.5), frac_in80=enclosed_fraction(grid, x, y, 0.8),
                             mean_logdens=float(np.mean(dens)), mean_logdens_err=float(np.std(boots)),
                             energy_dist=float(e_obs), p_energy=float(np.mean(np.array(e_perm) >= e_obs)),
                             nonsf_frac_sample=nonsf, nonsf_frac_pop=float(np.mean((pnonsf >= 0.05) | (qnonsf >= kauffmann03(np.minimum(pnonsf, 0.049)))))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-kpc", type=float, default=30.0); ap.add_argument("--pair-dv", type=float, default=500.0)
    ap.add_argument("--layers", default="pairs,bars", help="comma list of all,pairs,bars,desi")
    a = ap.parse_args()
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_bpt_comparison"
    sdss, desi = ratios(load_sdss()), ratios(load_desi())
    sdss["pair"] = sdss_pairs(sdss, a.pair_kpc, a.pair_dv)
    sdss["in_nair"], sdss["bar"], sdss["napair"], sdss["interacting"] = nair_match(sdss)
    pts, pairs = load_sample()
    allsamples = {"all": ("SDSS all (MPA-JHU)", sdss, "0.55", "solid", True), "pairs": ("SDSS pairs (< %.0f kpc, < %.0f km/s)" % (a.pair_kpc, a.pair_dv), sdss[sdss.pair], "#1f5fbf", "solid", False),
                  "bars": ("SDSS barred (Nair & Abraham 2010)", sdss[sdss.bar], "#8e2ca3", "dashed", False), "desi": ("DESI DR1 all", desi, "#2f2f2f", "dotted", False)}
    samples = {v[0]: v[1:] for k, v in allsamples.items() if k in a.layers.split(",")}
    stats = stage_consistency(pts, {k: allsamples[k][1] for k in ("pairs", "bars", "all")}, nboot=500, nperm=300, seed=1)
    stats.to_csv(ROOT / "results/tables/bpt_stage_vs_population.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220); print(stats.round(3).to_string(index=False))
    counts = {k: int(len(v[0])) for k, v in samples.items()}
    counts["Nair & Abraham matched"] = int(sdss.in_nair.sum()); counts["Nair pairs"] = int(sdss.napair.sum()); counts["Nair interacting"] = int(sdss.interacting.sum())

    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5))
    handles = []
    for ax, panel, xcol in ((axs[0], "n2", "n2"), (axs[1], "s2", "s2")):
        xlim = (-1.3, 0.7) if panel == "n2" else (-1.2, 0.6); ylim = (-1.3, 1.3)
        for name, (d, col, ls, fill) in samples.items():
            if panel == "s2" and "pairs" in name:
                pass
            xc, yc, H, lev = contour_levels(d[xcol].values, d.o3.values, xlim, ylim)
            if fill:
                ax.contourf(xc, yc, H, levels=lev + [H.max()], colors=["#e6e6e6", "#cfcfcf", "#b5b5b5"], zorder=0)
                cs = ax.contour(xc, yc, H, levels=lev, colors=[col], linewidths=0.8, linestyles=ls, zorder=1)
            else:
                cs = ax.contour(xc, yc, H, levels=lev, colors=[col], linewidths=1.3, linestyles=ls, zorder=2)
            if panel == "n2":
                handles.append(plt.Line2D([], [], color=col, ls=ls, lw=1.3, label=f"{name}: {len(d):,} galaxies"))
        draw_panel(ax, pts, pairs, panel)
    sample_legend(axs[0], extra=handles, loc="lower right")
    fig.suptitle("interbars nuclei (pPXF) over the BPT distributions of SDSS spectroscopic pairs and SDSS barred galaxies (contours enclose 20 / 50 / 80 %)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"results/figures/bpt_ppxf_comparison.{ext}", dpi=130)
    man = dict(run_id=run_id, script="scripts/bpt_comparison.py", args=vars(a),
               inputs=["output/comparison/sdss_dr8_galspec_z0p1.csv (Data Lab sdss_dr8.galspecline+galspecinfo)", "output/comparison/desi_dr1_emline_z0p1.csv (Data Lab desi_dr1.stellar_mass_emline)",
                       "output/comparison/nair_abraham_2010.csv (VizieR J/ApJS/186/427/table2)", "results/tables/ppxf_lines.csv"],
               outputs=["results/figures/bpt_ppxf_comparison.{png,pdf}", "results/tables/bpt_stage_vs_population.csv"],
               summary=dict(counts=counts, levels=LEVELS, stage_vs_population=stats.to_dict("records")))
    (ROOT / "results/manifests" / f"{run_id}.json").write_text(json.dumps(man, indent=2, default=str))
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
