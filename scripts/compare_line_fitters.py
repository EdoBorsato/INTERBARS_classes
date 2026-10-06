"""Three checks of the pPXF line fluxes (results/tables/ppxf_lines.csv):
  (a) pPXF vs FastSpecFit on the same spectrum (DESI coadds, and the SDSS spectra that were fitted by
      FastSpecFit through the DESI-format wrapper before it was abandoned) — results/tables/fastspecfit_bpt.csv
  (b) pPXF vs the SDSS pipeline's own Gaussian fits (SPZLINE extension of the lite files)
  (c) DESI 1.5" fibre vs SDSS 3" fibre of the same galaxy, both with pPXF: the aperture effect
Outputs: results/tables/line_fitter_comparison.csv, results/figures/line_fitter_comparison.png,
         results/manifests/<run_id>.json
Run:  source scripts/fastspecfit_env.sh && python scripts/compare_line_fitters.py
"""
import datetime
import json
from pathlib import Path

import fitsio
import matplotlib
if "ipykernel" not in __import__("sys").modules:      # a notebook keeps its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINES = {"hb": "H_beta", "oiii": "[O_III] 5007", "ha": "H_alpha", "nii": "[N_II] 6583"}
LABEL = {"hb": "Hβ", "oiii": "[OIII]5007", "ha": "Hα", "nii": "[NII]6584"}


def spzline(path):
    with fitsio.FITS(path) as F:
        names = [h.get_extname() for h in F]
        d = F[names.index("SPZLINE") if "SPZLINE" in names else 3].read()
    out = {}
    for k, ln in LINES.items():
        r = d[np.char.strip(d["LINENAME"].astype(str)) == ln]
        out[f"{k}_sdsspipe"] = float(r["LINEAREA"][0]) if len(r) else np.nan
        out[f"{k}_sdsspipe_err"] = float(r["LINEAREA_ERR"][0]) if len(r) else np.nan
    return out


def robust(q):
    q = q[np.isfinite(q) & (q > 0)]
    if len(q) < 2:
        return dict(n=int(len(q)), median=np.nan, scatter_dex=np.nan)
    lq = np.log10(q)
    return dict(n=int(len(q)), median=float(10 ** np.median(lq)), scatter_dex=float(1.4826 * np.median(np.abs(lq - np.median(lq)))))


def main():
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_compare_line_fitters"
    t = pd.read_csv(ROOT / "results/tables/ppxf_lines.csv").set_index("label")
    fsf = pd.read_csv(ROOT / "results/tables/fastspecfit_bpt.csv").set_index("label")
    tg = pd.read_csv(ROOT / "output/desi_redux/sdss_wrapped.csv").set_index("label") if (ROOT / "output/desi_redux/sdss_wrapped.csv").exists() else None
    for k in LINES:
        t[f"{k}_fsf"] = t.index.map(fsf[f"{k}_flux"]); t[f"{k}_fsf_err"] = t.index.map(fsf[f"{k}_err"])
    pipe = {}
    for lab, r in t[t.source == "SDSS"].iterrows():
        p, mj, f = r.spec_id.split("-"); sysdir = ROOT / "dataset/spectra" / r.system
        files = list(sysdir.glob(f"*/spec-{int(p):04d}-{mj}-{int(f):04d}.fits"))
        if files:
            pipe[lab] = spzline(files[0])
    pipe = pd.DataFrame(pipe).T
    t = t.join(pipe)
    stats = {}
    for k in LINES:
        ok = (t[f"{k}_snr"] > 3)
        stats[f"{k}_ppxf_over_fsf_desi"] = robust((t[f"{k}_flux"] / t[f"{k}_fsf"])[ok & (t.source == "DESI")])
        stats[f"{k}_ppxf_over_fsf_sdss"] = robust((t[f"{k}_flux"] / t[f"{k}_fsf"])[ok & (t.source == "SDSS")])
        stats[f"{k}_ppxf_over_sdsspipe"] = robust((t[f"{k}_flux"] / t[f"{k}_sdsspipe"])[ok & (t[f"{k}_sdsspipe"] / t[f"{k}_sdsspipe_err"] > 3)])
    # (c) same galaxy, both fibres
    both = []
    for lab, r in t[t.source == "DESI"].iterrows():
        s = lab + "_sdss"
        if s in t.index:
            both.append(dict(galaxy=lab, n2_desi=r.log_nii_ha, n2_sdss=t.loc[s, "log_nii_ha"], o3_desi=r.log_oiii_hb, o3_sdss=t.loc[s, "log_oiii_hb"],
                             ha_sdss_over_desi=t.loc[s, "ha_flux"] / r.ha_flux, class_desi=r.bpt_class, class_sdss=t.loc[s, "bpt_class"]))
    both = pd.DataFrame(both)
    t.to_csv(ROOT / "results/tables/line_fitter_comparison.csv", float_format="%.5g")

    fig, axs = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, k in zip(axs[0], LINES):
        for src, mk, col in (("DESI", "o", "tab:blue"), ("SDSS", "^", "tab:orange")):
            g = t[(t.source == src) & (t[f"{k}_snr"] > 3)]
            ax.errorbar(g[f"{k}_fsf"], g[f"{k}_flux"], xerr=g[f"{k}_fsf_err"], yerr=g[f"{k}_err"], fmt=mk, ms=4, capsize=2, color=col, label=f"{src} spectra")
        lim = np.nanmax(t[[f"{k}_flux", f"{k}_fsf"]].values) * 1.2
        ax.plot([1, lim], [1, lim], "k--", lw=0.8); ax.set(xscale="log", yscale="log", xlim=(1, lim), ylim=(1, lim), xlabel=f"{LABEL[k]} FastSpecFit", ylabel=f"{LABEL[k]} pPXF")
        sd, ss = stats[f"{k}_ppxf_over_fsf_desi"], stats[f"{k}_ppxf_over_fsf_sdss"]
        ax.set_title(f"pPXF / FastSpecFit: DESI {sd['median']:.2f} (n={sd['n']}), SDSS {ss['median']:.2f} ±{ss['scatter_dex']:.2f} dex (n={ss['n']})", fontsize=8)
    axs[0, 0].legend(fontsize=8)
    for ax, k in zip(axs[1, :3], ("hb", "oiii", "ha")):
        g = t[(t.source == "SDSS") & (t[f"{k}_snr"] > 3)]
        ax.errorbar(g[f"{k}_sdsspipe"], g[f"{k}_flux"], xerr=g[f"{k}_sdsspipe_err"], yerr=g[f"{k}_err"], fmt="^", ms=4, capsize=2, color="tab:orange")
        lim = np.nanmax(g[[f"{k}_flux", f"{k}_sdsspipe"]].values) * 1.2
        ax.plot([1, lim], [1, lim], "k--", lw=0.8); ax.set(xscale="log", yscale="log", xlim=(1, lim), ylim=(1, lim), xlabel=f"{LABEL[k]} SDSS pipeline (SPZLINE)", ylabel=f"{LABEL[k]} pPXF")
        s = stats[f"{k}_ppxf_over_sdsspipe"]; ax.set_title(f"pPXF / SDSS pipeline: median {s['median']:.2f}, scatter {s['scatter_dex']:.2f} dex (n={s['n']})", fontsize=8)
    ax = axs[1, 3]
    if len(both):
        for r in both.itertuples():
            ax.annotate("", xy=(r.n2_sdss, r.o3_sdss), xytext=(r.n2_desi, r.o3_desi), arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
            ax.plot(r.n2_desi, r.o3_desi, "o", color="tab:blue"); ax.plot(r.n2_sdss, r.o3_sdss, "^", color="tab:orange"); ax.annotate(r.galaxy, (r.n2_desi, r.o3_desi), xytext=(4, 3), textcoords="offset points", fontsize=6)
        xx = np.linspace(-1.6, 0.0, 200); ax.plot(xx[xx < 0.05], 0.61 / (xx[xx < 0.05] - 0.05) + 1.3, "k--", lw=1)
        xx = np.linspace(-1.6, 0.4, 200); ax.plot(xx, 0.61 / (xx - 0.47) + 1.19, "k-", lw=1)
        ax.set(xlim=(-1.2, 0.5), ylim=(-1.2, 0.8), xlabel="log([NII]/Hα)", ylabel="log([OIII]/Hβ)", title="same galaxy: DESI 1.5\" (o) -> SDSS 3\" (^), both pPXF", fontsize=9) if False else ax.set(xlim=(-1.2, 0.5), ylim=(-1.2, 0.8), xlabel="log([NII]/Hα)", ylabel="log([OIII]/Hβ)")
        ax.set_title("same galaxy: DESI 1.5\" (o) → SDSS 3\" (^), both pPXF", fontsize=9)
    fig.suptitle("pPXF line fluxes: vs FastSpecFit (top), vs the SDSS pipeline (bottom left), aperture effect (bottom right)")
    fig.tight_layout(); fig.savefig(ROOT / "results/figures/line_fitter_comparison.png", dpi=120)
    man = dict(run_id=run_id, script="scripts/compare_line_fitters.py", inputs=["results/tables/ppxf_lines.csv", "results/tables/fastspecfit_bpt.csv", "dataset/spectra/*/*/spec-*.fits[SPZLINE]"],
               outputs=["results/tables/line_fitter_comparison.csv", "results/figures/line_fitter_comparison.png"], summary=dict(stats=stats, both_fibres=both.to_dict("records")))
    (ROOT / "results/manifests" / f"{run_id}.json").write_text(json.dumps(man, indent=2, default=str))
    print(pd.DataFrame(stats).T.round(3).to_string())
    if len(both):
        print("\nsame galaxy, two fibres:"); print(both.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
