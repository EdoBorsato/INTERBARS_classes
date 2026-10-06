"""Stellar continuum + emission-line fits of every nuclear spectrum of the interbars sample with pPXF
(Cappellari 2017, 2023), the same way for DESI DR1 coadds and SDSS DR17 lite files: no format conversion.

Per spectrum
  1. read: DESI arms joined (0.8 A linear grid, resolution from the RESOLUTION matrix) or the SDSS lite
     file (1e-4 dex log grid, resolution from `wdisp`); ivar -> noise; pixels with ivar = 0 excluded
  2. vacuum -> air wavelengths (the E-MILES templates are in air), log-rebin (pPXF's log_rebin), noise
     rebinned with the variance and the number of pixels per new pixel
  3. templates: E-MILES SSPs (ppxf sps_lib, degraded to the spectrum's FWHM(lambda) when coarser
     than the library) + Gaussian gas templates at the same FWHM (ppxf_util.emission_lines):
     free Balmer fluxes (Hbeta, Halpha, ...), free [SII] doublet, [OIII] / [NII] / [OI] doublets at
     their atomic ratios (one template each, flux of the strong line = template flux / 1.33)
  4. pPXF: stars (V, sigma), narrow Balmer (V, sigma), narrow forbidden (V, sigma) as three kinematic
     components (sigma <= 500 km/s); multiplicative polynomial (mdegree 10) for the flux calibration, no
     additive one. A 4th, broad Balmer component (500-5000 km/s) is tried and adopted when it lowers the
     total chi2 by > 25, has broad Halpha S/N > 5, 600 < sigma < 4500 km/s (away from the bounds) and a broad /
     narrow Halpha flux ratio > 0.3 (columns broad_adopted, ha_broad_flux, sigma_broad, *_trial)
  5. fluxes: pp.gas_flux x pixel size in A at the observed line x the spectrum normalisation
     -> 1e-17 erg/s/cm2; formal errors from pp.gas_flux_error

Inputs: dataset/catalogs/specz_matches.csv (DESI / SDSS rows within 5" of a main galaxy or companion),
        output/desi_redux/iron/healpix/ (DESI coadds), dataset/spectra/<system>/<galaxy>/spec-*.fits
        (SDSS lite; downloaded from the SAS when missing), output/ppxf/sps_models/spectra_emiles_9.0.npz
Outputs: results/tables/ppxf_lines.csv (one row per spectrum; BPT columns as in fastspecfit_bpt.csv, plus
         stellar kinematics and light-weighted age / metallicity; FastSpecFit values merged when present),
         output/ppxf/<label>.png (pp.plot: full fit + gas), results/manifests/<run_id>.json
Run:  source scripts/fastspecfit_env.sh && python scripts/run_ppxf.py [--only SYSTEM ...] [--sps emiles|fsps] [--force]
"""
import argparse
import datetime
import json
import os
import re
from pathlib import Path

import fitsio
import matplotlib
if "ipykernel" not in __import__("sys").modules:      # a notebook keeps its inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "dataset" / "spectra"
REDUX = Path(os.environ.get("DESI_SPECTRO_REDUX", ROOT / "output" / "desi_redux")) / "iron" / "healpix"
OUT = ROOT / "output" / "ppxf"
SPS = OUT / "sps_models"
SAS = "https://data.sdss.org/sas/dr17"
C_KMS = 299792.458
# pPXF gas-template names -> our keys (doublet templates hold both lines: strong line = flux / 1.33)
GAS = {"Hbeta": ("hb", 1.0), "[OIII]5007_d": ("oiii", 1.33), "Halpha": ("ha", 1.0), "[NII]6583_d": ("nii", 1.33),
       "[SII]6716": ("sii6716", 1.0), "[SII]6731": ("sii6731", 1.0), "[OI]6300_d": ("oi", 1.33), "Hgamma": ("hg", 1.0),
       "Hdelta": ("hd", 1.0), "[OII]3726": ("oii3726", 1.0), "[OII]3729": ("oii3729", 1.0)}


# ------------------------------------------------------------------ inventory
def targets():
    m = pd.read_csv(ROOT / "dataset/catalogs/specz_matches.csv", dtype={"obj_id": str})
    t = pd.read_csv(ROOT / "dataset/catalogs/interbars_targets.csv").set_index("system")
    rows = []
    d = m[(m.survey == "DESI DR1") & (m.sep_arcsec <= 5) & m.usable].copy()
    d = d[d.extra.str.contains("primary=t")].sort_values("sep_arcsec").drop_duplicates(["system", "role"])
    for r in d.itertuples():
        e = dict(re.findall(r"(\w+)=(\w+)", r.extra)); hp = int(e["healpix"])
        rows.append(dict(label=r.system + ("_comp" if r.role == "comp" else ""), system=r.system, role=r.role, source="DESI",
                         spec_id=r.obj_id, z_in=r.z, file=str(REDUX / e["survey"] / e["program"] / str(hp // 100) / str(hp) / f"coadd-{e['survey']}-{e['program']}-{hp}.fits")))
    s = m[(m.survey == "SDSS DR17") & (m.sep_arcsec <= 5) & m.good & ~m.is_star].copy()
    s = s.sort_values(["role", "sep_arcsec"], ascending=[False, True]).drop_duplicates(["system", "role"])
    for r in s.itertuples():
        e = dict(re.findall(r"(\w+)=([\w.]+)", r.extra)); p, mj, f = int(e["plate"]), int(e["mjd"]), int(e["fiberid"])
        gal = r.system if r.role == "main" else (t.comp_name.get(r.system) if isinstance(t.comp_name.get(r.system), str) else "comp")
        rows.append(dict(label=r.system + ("_comp" if r.role == "comp" else "") + "_sdss", system=r.system, role=r.role, source="SDSS",
                         spec_id=f"{p}-{mj}-{f}", z_in=r.z, run2d=e.get("run2d", ""), file=str(SPEC / r.system / gal / f"spec-{p:04d}-{mj}-{f:04d}.fits")))
    return pd.DataFrame(rows).drop_duplicates("file")


def fetch_lite(path, run2d):
    path = Path(path)
    if path.exists() and path.stat().st_size > 100000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    p, mj, f = path.stem.split("-")[1:]
    urls = ([f"{SAS}/sdss/spectro/redux/{run2d}/spectra/lite/{p}/{path.name}", f"{SAS}/eboss/spectro/redux/{run2d}/spectra/lite/{p}/{path.name}"] if run2d else []) + \
           [f"{SAS}/sdss/spectro/redux/{r}/spectra/lite/{p}/{path.name}" for r in ("26", "103", "104")] + [f"{SAS}/eboss/spectro/redux/v5_13_2/spectra/lite/{p}/{path.name}"]
    for u in urls:
        r = requests.get(u, timeout=120)
        if r.ok and r.content[:6] == b"SIMPLE":
            path.write_bytes(r.content); return
    raise RuntimeError(f"cannot download {path.name}")


# ------------------------------------------------------------------ readers -> (lam_vac, flux, ivar, fwhm_A, mask)
def read_desi(coadd, targetid):
    fm = fitsio.read(coadd, "FIBERMAP", columns=["TARGETID"])
    i = int(np.where(fm["TARGETID"] == int(targetid))[0][0])
    lam, flux, ivar, mask, fwhm = [], [], [], [], []
    g = lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2)
    with fitsio.FITS(coadd) as F:
        for b in "BRZ":
            w = F[f"{b}_WAVELENGTH"].read(); R = F[f"{b}_RESOLUTION"][i:i + 1, :, :][0]
            off = np.arange(R.shape[0]) - R.shape[0] // 2; dl = w[1] - w[0]
            js = np.linspace(20, len(w) - 21, 12).astype(int); sig = []
            for j in js:
                try:
                    sig.append(abs(curve_fit(g, off, R[:, j], p0=[R[:, j].max(), 0, 1])[0][2]))
                except Exception:
                    sig.append(1.0)
            lam.append(w); flux.append(F[f"{b}_FLUX"][i:i + 1, :][0]); ivar.append(F[f"{b}_IVAR"][i:i + 1, :][0])
            mask.append(F[f"{b}_IVAR"][i:i + 1, :][0] <= 0); fwhm.append(2.3548 * dl * np.interp(w, w[js], sig))   # bad = ivar 0, as for SDSS
    lam, flux, ivar, mask, fwhm = (np.concatenate(x) for x in (lam, flux, ivar, mask, fwhm))
    o = np.argsort(lam); return lam[o], flux[o], ivar[o], fwhm[o], mask[o]


def read_sdss(path):
    with fitsio.FITS(path) as F:
        names = [h.get_extname() for h in F]
        d = F[names.index("COADD") if "COADD" in names else 1].read()
    c = {k.lower(): k for k in d.dtype.names}
    lam = 10 ** d[c["loglam"]].astype(np.float64); dlam = np.gradient(lam)
    # bad pixels = ivar == 0 only: on old plates `and_mask` flags most of the red half (e.g. plate 1225: 44 %, plate 581: 100 %)
    ivar = d[c["ivar"]].astype(np.float64)
    return lam, d[c["flux"]].astype(np.float64), ivar, 2.3548 * d[c["wdisp"]] * dlam, ivar <= 0


# ------------------------------------------------------------------ one fit
def fit_one(lam_vac, flux, ivar, fwhm, mask, z, sps_file, label, plot=True):
    good = (ivar > 0) & ~mask & np.isfinite(flux)
    lam_air = util.vac_to_air(lam_vac)
    flux = np.where(good, flux, np.interp(lam_air, lam_air[good], flux[good]))
    var = np.where(good, 1 / np.where(good, ivar, 1), 0.0)
    norm = np.median(flux[good])
    # log-rebin flux; the variance rebinned the same way, divided by the pixels averaged into each new pixel
    velscale = util.log_rebin(lam_air, flux)[2]                   # one velocity scale for every rebinning below
    galaxy, ln_lam, _ = util.log_rebin(lam_air, flux / norm, velscale=velscale)
    var_log = util.log_rebin(lam_air, var / norm ** 2, velscale=velscale)[0]
    bad_log = util.log_rebin(lam_air, (~good).astype(float), velscale=velscale)[0] > 0.3
    lam_gal = np.exp(ln_lam); npix_per = np.clip(lam_gal * velscale / C_KMS / np.median(np.diff(lam_air)), 1, None)
    noise = np.sqrt(np.clip(var_log / npix_per, 1e-8, None)); noise[bad_log] = 1e3
    fwhm_gal = {"lam": lam_gal, "fwhm": np.interp(lam_gal, lam_air, fwhm)}
    sps = lib.sps_lib(sps_file, velscale, fwhm_gal, norm_range=[5070, 5950])
    reg_dim = sps.templates.shape[1:]; stars = sps.templates.reshape(sps.templates.shape[0], -1)
    lam_range_gal = np.array([lam_gal[0], lam_gal[-1]]) / (1 + z)
    gas, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal, tie_balmer=False, limit_doublets=False)
    templates = np.column_stack([stars, gas])
    n_temps = stars.shape[1]; forb = np.array(["[" in n for n in gas_names])
    component = [0] * n_temps + [2 if f else 1 for f in forb]
    gas_component = np.array(component) > 0
    vel = C_KMS * np.log(1 + z); start = [[vel, 150.0], [vel, 100.0], [vel, 100.0]]
    # sky-line residual (5577) and the telluric A band (observed frame, air). The O2 B band (6863-6890 A) is NOT
    # masked: at z = 0.022-0.026 the [SII] doublet sits inside it and a masked template gives absurd fluxes.
    goodpixels = np.where(~bad_log & ~((lam_gal > 5573) & (lam_gal < 5583)) & ~((lam_gal > 7590) & (lam_gal < 7700)))[0]
    bounds_n = [[[vel - 1500, vel + 1500], [10, 500]]] * 3                  # narrow components: sigma <= 500 km/s
    pp = ppxf(templates, galaxy, noise, velscale, start, moments=[2, 2, 2], degree=-1, mdegree=10, lam=lam_gal, lam_temp=sps.lam_temp,
              component=component, gas_component=gas_component, gas_names=gas_names, goodpixels=goodpixels, bounds=bounds_n, quiet=True)
    # broad Balmer trial: a second copy of the Balmer templates as a 4th kinematic component (sigma 500-5000 km/s);
    # adopted, as in FastSpecFit, when the total chi2 drops by > 25 and the broad Halpha flux has S/N > 3
    balmer = ~forb; ib = int(np.where(gas_names == "Halpha")[0][0])
    templates_b = np.column_stack([templates, gas[:, balmer]])
    component_b = component + [3] * int(balmer.sum()); gas_component_b = np.array(component_b) > 0
    gas_names_b = np.concatenate([gas_names, [n + "_broad" for n in gas_names[balmer]]])
    pp_b = ppxf(templates_b, galaxy, noise, velscale, start + [[vel, 1500.0]], moments=[2, 2, 2, 2], degree=-1, mdegree=10, lam=lam_gal,
                lam_temp=sps.lam_temp, component=component_b, gas_component=gas_component_b, gas_names=gas_names_b, goodpixels=goodpixels,
                bounds=bounds_n + [[[vel - 3000, vel + 3000], [500, 5000]]], quiet=True)
    ibb = int(np.where(gas_names_b == "Halpha_broad")[0][0])
    dchi2 = (pp.chi2 - pp_b.chi2) * len(goodpixels)
    snr_broad = pp_b.gas_flux[ibb] / pp_b.gas_flux_error[ibb] if pp_b.gas_flux_error[ibb] > 0 else 0.0
    # a physical broad line: width away from both bounds, comparable to the narrow Halpha (not a few-percent
    # continuum fudge), significant. With Δχ² > 25 and S/N > 3 alone, 31/52 spectra "adopt" a broad component.
    sig_b = float(pp_b.sol[3][1]); ratio_b = float(pp_b.gas_flux[ibb] / max(pp_b.gas_flux[ib], 1e-9))
    adopt_broad = bool(dchi2 > 25 and snr_broad > 5 and 600 < sig_b < 4500 and ratio_b > 0.3)
    out = dict(dchi2_broad=float(dchi2), snr_broad_halpha=float(snr_broad), sigma_broad_trial=sig_b, broad_over_narrow_trial=ratio_b, broad_adopted=adopt_broad)
    if adopt_broad:
        pp, gas_names, gas_component = pp_b, gas_names_b, gas_component_b
        line_wave = np.concatenate([line_wave, line_wave[balmer]])
        dl_b = line_wave[ibb] * (1 + z) * velscale / C_KMS
        out.update(ha_broad_flux=float(pp.gas_flux[ibb] * dl_b * norm), ha_broad_err=float(pp.gas_flux_error[ibb] * dl_b * norm),
                   v_broad=float(pp.sol[3][0]), sigma_broad=float(pp.sol[3][1]))
    out.update(chi2=float(pp.chi2), v_star=float(pp.sol[0][0]), sigma_star=float(pp.sol[0][1]), v_balmer=float(pp.sol[1][0]), sigma_balmer=float(pp.sol[1][1]),
               v_forbidden=float(pp.sol[2][0]), sigma_forbidden=float(pp.sol[2][1]), z_fit=float(np.exp(pp.sol[1][0] / C_KMS) - 1))
    age, met = sps.mean_age_metal(pp.weights[:n_temps].reshape(reg_dim), quiet=True)
    out["age_lw_gyr"], out["metal_lw"] = float(age), float(met)
    for name, fl, er, lw in zip(gas_names, pp.gas_flux, pp.gas_flux_error, line_wave):
        if name not in GAS:
            continue                                  # broad copies are handled above
        key, fac = GAS[name]
        dlam_pix = lw * (1 + z) * velscale / C_KMS                 # pixel size [A] at the observed line
        out[f"{key}_flux"] = float(fl * dlam_pix * norm / fac) if np.isfinite(fl) else np.nan
        out[f"{key}_err"] = float(er * dlam_pix * norm / fac) if np.isfinite(er) else np.nan
    if plot:
        fig = plt.figure(figsize=(14, 8)); plt.subplot(211); pp.plot(); plt.title(f"{label}: pPXF stars + gas, chi2/dof = {pp.chi2:.2f}" + ("  [broad Balmer component adopted]" if adopt_broad else ""))
        for k, (lo, hi, ttl) in enumerate([(4800, 5050, "Hβ [OIII]"), (6500, 6760, "Hα [NII] [SII]")]):
            ax = plt.subplot(2, 2, 3 + k); s_ = (lam_gal / (1 + z) > lo) & (lam_gal / (1 + z) < hi)
            gasfit = pp.gas_bestfit
            ax.plot(lam_gal[s_] / (1 + z), galaxy[s_], "k", lw=0.8, drawstyle="steps-mid", label="data"); ax.plot(lam_gal[s_] / (1 + z), pp.bestfit[s_], "r", lw=1, label="stars + gas")
            ax.plot(lam_gal[s_] / (1 + z), (pp.bestfit - gasfit)[s_], color="tab:orange", lw=1, label="stars"); ax.set_title(ttl); ax.set_xlabel("rest wavelength [Å, air]")
            if k == 0: ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / f"{label}.png", dpi=100); plt.close(fig)
    return out


def bpt_class(x, y):
    if not (np.isfinite(x) and np.isfinite(y)):
        return ""
    kauff = 0.61 / (x - 0.05) + 1.3 if x < 0.05 else -np.inf
    kew = 0.61 / (x - 0.47) + 1.19 if x < 0.47 else -np.inf
    return "SF" if y < kauff else "composite" if y < kew else "AGN"


def derived(row):
    def lr(a, b):
        fa, fb = row.get(f"{a}_flux", np.nan), row.get(f"{b}_flux", np.nan)
        if not (fa > 0 and fb > 0):
            return np.nan, np.nan
        return np.log10(fa / fb), np.hypot(row[f"{a}_err"] / fa, row[f"{b}_err"] / fb) / np.log(10)
    for k in ("ha", "hb", "oiii", "nii", "sii6716", "sii6731", "oi"):
        row[f"{k}_snr"] = row[f"{k}_flux"] / row[f"{k}_err"] if row.get(f"{k}_err", 0) > 0 else np.nan
    row["log_nii_ha"], row["log_nii_ha_err"] = lr("nii", "ha"); row["log_oiii_hb"], row["log_oiii_hb_err"] = lr("oiii", "hb")
    row["sii_flux"] = row.get("sii6716_flux", np.nan) + row.get("sii6731_flux", np.nan)
    row["log_sii_ha"] = np.log10(row["sii_flux"] / row["ha_flux"]) if row["sii_flux"] > 0 and row["ha_flux"] > 0 else np.nan
    row["log_oi_ha"] = np.log10(row["oi_flux"] / row["ha_flux"]) if row.get("oi_flux", 0) > 0 and row["ha_flux"] > 0 else np.nan
    row["balmer_dec"] = row["ha_flux"] / row["hb_flux"] if row.get("hb_flux", 0) > 0 else np.nan
    row["min_snr_bpt"] = np.nanmin([row[f"{k}_snr"] if np.isfinite(row[f"{k}_snr"]) else 0 for k in ("ha", "nii", "hb", "oiii")])
    row["bpt_class"] = bpt_class(row["log_nii_ha"], row["log_oiii_hb"]) if row["min_snr_bpt"] >= 3 else "low S/N"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*"); ap.add_argument("--sps", default="emiles", choices=["emiles", "fsps"]); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_run_ppxf"
    OUT.mkdir(parents=True, exist_ok=True)
    sps_file = SPS / f"spectra_{a.sps}_9.0.npz"
    cache = OUT / f"ppxf_lines_{a.sps}.json"
    done = json.loads(cache.read_text()) if cache.exists() and not a.force else {}
    tg = targets(); rows, skipped = [], []
    for r in tg.itertuples():
        if a.only and r.system not in a.only:
            continue
        if r.label in done:
            rows.append(done[r.label]); continue
        try:
            if r.source == "DESI":
                lam, flux, ivar, fwhm, mask = read_desi(r.file, r.spec_id)
            else:
                fetch_lite(r.file, getattr(r, "run2d", "")); lam, flux, ivar, fwhm, mask = read_sdss(r.file)
            res = fit_one(lam, flux, ivar, fwhm, mask, float(r.z_in), sps_file, r.label)
        except Exception as e:
            skipped.append(f"{r.label}: {e}"); print("  !", r.label, e); continue
        row = derived(dict(label=r.label, system=r.system, role=r.role, source=r.source, spec_id=r.spec_id, z_in=float(r.z_in), sps=a.sps, **res))
        done[r.label] = row; rows.append(row); cache.write_text(json.dumps(done, indent=1, default=float))
        print(f"  {r.label:34s} {r.source} z={row['z_fit']:.5f} sig*={row['sigma_star']:.0f} chi2={row['chi2']:.2f}  N2/Ha={row['log_nii_ha']:+.3f} O3/Hb={row['log_oiii_hb']:+.3f} minS/N={row['min_snr_bpt']:.0f} {row['bpt_class']}")
    t = pd.DataFrame(rows)
    fsf = ROOT / "results/tables/fastspecfit_bpt.csv"
    if fsf.exists() and len(t):
        f = pd.read_csv(fsf).set_index("label")
        for k in ("ha", "hb", "oiii", "nii"):
            t[f"{k}_flux_fsf"] = t.label.map(f[f"{k}_flux"])
        t["log_nii_ha_fsf"], t["log_oiii_hb_fsf"], t["bpt_class_fsf"] = t.label.map(f.log_nii_ha), t.label.map(f.log_oiii_hb), t.label.map(f.bpt_class)
    t.to_csv(ROOT / "results/tables/ppxf_lines.csv", index=False, float_format="%.5g")
    man = dict(run_id=run_id, script="scripts/run_ppxf.py", args=vars(a), ppxf_version=__import__("ppxf").__version__, sps=str(sps_file),
               inputs=["dataset/catalogs/specz_matches.csv", str(REDUX), "dataset/spectra/"], outputs=["results/tables/ppxf_lines.csv", "output/ppxf/*.png"],
               summary=dict(n_fit=len(t), by_source=t.source.value_counts().to_dict() if len(t) else {}, classes=t.bpt_class.value_counts().to_dict() if len(t) else {}, skipped=skipped))
    (ROOT / "results/manifests" / f"{run_id}.json").write_text(json.dumps(man, indent=2, default=str))
    print(json.dumps(man["summary"], indent=1))


if __name__ == "__main__":
    main()
