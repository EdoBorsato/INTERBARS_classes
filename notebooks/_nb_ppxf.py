# Cells of notebooks 02 and 03 (pPXF version); spliced into build_series.py by the builder.
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP2 = r"""import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # one BLAS thread: on Binder / Colab the pod has
    os.environ.setdefault(_v, "1")                                          # one core and multi-threaded BLAS is 30x slower
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, fitsio
import matplotlib.pyplot as plt
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DESI_SPECTRO_REDUX", str(ROOT / "output" / "desi_redux"))
import run_ppxf as RP                      # the project's readers and the production fit, so notebook and pipeline agree
C_KMS = 299792.458
plt.rcParams.update({"figure.dpi": 110})
print("repo:", ROOT, "| pPXF", __import__("ppxf").__version__)"""


def nb02():
    C = [md(r"""# 2. Continuum and emission-line measurements with pPXF

We measure the lines with **pPXF** (Cappellari 2017, 2023), the standard full-spectrum fitting code. It
fits, in one go, a **stellar continuum** (a non-negative combination of simple stellar populations,
broadened by the stars' velocity dispersion) and **Gaussian emission lines**, each convolved with the
instrument's line-spread function. Because it only needs wavelength, flux, noise and resolution arrays,
the same code runs on DESI and SDSS spectra: no format conversion, one set of numbers for the whole sample.

The steps, one per section: read the spectrum (1), put it on a logarithmic wavelength grid (2), build the
stellar and gas templates (3), fit (4), turn the fitted weights into line fluxes (5), see why the stellar
continuum matters for Hβ (6), test for a broad Hα component (7), cross-check against two other codes (8).

`scripts/run_ppxf.py` does exactly this for every spectrum of the sample; we import its readers and, at
the end, its fit function to check that the notebook reproduces the pipeline."""),
        code(SETUP2),
        md(r"""## 2.1 Read a spectrum

`MY_SYSTEMS` comes from notebook 1; the main galaxy's DESI spectrum is used when there is one, else the
SDSS one (`LABEL` can be set to any other spectrum of the list). The readers return the same five arrays
for both surveys: vacuum wavelength [Å], flux and inverse variance [10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹], the
instrumental FWHM at every pixel [Å] (from the DESI resolution matrix, or the SDSS `wdisp`), and a
bad-pixel mask (ivar = 0). `_sdss` labels are SDSS spectra."""),
        code(r"""import json
MY_SYSTEMS = json.loads((ROOT / "notebooks/my_systems.json").read_text()) if (ROOT / "notebooks/my_systems.json").exists() else ["IC3147B"]
tg = RP.targets().set_index("label")
print("spectra of your system(s):"); print(tg[tg.system.isin(MY_SYSTEMS)][["system", "role", "source", "spec_id", "z_in"]].to_string())
cand = tg[(tg.system == MY_SYSTEMS[0]) & (tg.role == "main")].sort_values("source")
LABEL = cand.index[0]                         # the main galaxy's DESI spectrum if there is one, else its SDSS one; or set another label here
r = tg.loc[LABEL]; z = float(r.z_in)
if r.source == "DESI":
    lam_vac, flux, ivar, fwhm, mask = RP.read_desi(r.file, r.spec_id)
else:
    RP.fetch_lite(r.file, getattr(r, "run2d", "")); lam_vac, flux, ivar, fwhm, mask = RP.read_sdss(r.file)
good = (ivar > 0) & ~mask
print(f"{LABEL}: {r.source}, z = {z:.5f}, {good.sum()} good pixels, {lam_vac[0]:.0f}-{lam_vac[-1]:.0f} Å, FWHM {fwhm[good].min():.2f}-{fwhm[good].max():.2f} Å")
fig, ax = plt.subplots(figsize=(12, 3.5)); ax.plot(lam_vac, np.where(good, flux, np.nan), lw=0.5, color="0.3")
ax.set(xlabel="observed vacuum wavelength [Å]", ylabel="flux [1e-17 erg/s/cm²/Å]", title=f"step 1: {LABEL}"); plt.show()"""),
        md(r"""## 2.2 Air wavelengths and the logarithmic grid

Two conventions to get right before any fit. **Air vs vacuum**: the E-MILES stellar library is tabulated
at air wavelengths, DESI and SDSS at vacuum ones; we convert the spectrum (`util.vac_to_air`). Forgetting
it shifts every line by ~80 km/s. **Log grid**: a Doppler shift is a constant *fraction* of the wavelength,
so pPXF works on a grid where each pixel is a constant velocity step, `velscale` km/s (`util.log_rebin`).
The noise is rebinned with the variance and the number of original pixels averaged into each new one."""),
        code(r"""lam_air = util.vac_to_air(lam_vac)
flux_f = np.where(good, flux, np.interp(lam_air, lam_air[good], flux[good]))      # fill bad pixels for the rebinning
var = np.where(good, 1 / np.where(good, ivar, 1), 0.0)
norm = np.median(flux_f[good])                                                   # pPXF likes numbers of order 1
velscale = util.log_rebin(lam_air, flux_f)[2]
galaxy, ln_lam, _ = util.log_rebin(lam_air, flux_f / norm, velscale=velscale)
var_log = util.log_rebin(lam_air, var / norm ** 2, velscale=velscale)[0]
bad_log = util.log_rebin(lam_air, (~good).astype(float), velscale=velscale)[0] > 0.3
lam_gal = np.exp(ln_lam)
npix_per = np.clip(lam_gal * velscale / C_KMS / np.median(np.diff(lam_air)), 1, None)
noise = np.sqrt(np.clip(var_log / npix_per, 1e-8, None)); noise[bad_log] = 1e3
fwhm_gal = {"lam": lam_gal, "fwhm": np.interp(lam_gal, lam_air, fwhm)}
print(f"velscale = {velscale:.1f} km/s per pixel; {len(galaxy)} log pixels; median S/N per pixel = {np.median(galaxy[~bad_log] / noise[~bad_log]):.1f}")"""),
        md(r"""## 2.3 Templates: stars and gas

**Stars**: the E-MILES simple stellar populations (Vazdekis et al. 2016), a grid in age and metallicity.
`sps_lib` loads them, degrades them to the spectrum's FWHM(λ) where the library is sharper, and puts
them on the same velocity grid. **Gas**: one Gaussian per emission line at the instrumental width
(`emission_lines`); pPXF then fits the extra, astrophysical broadening. Doublets with fixed atomic
ratios ([O III], [N II], [O I]) are single templates; the [S II] and [O II] doublets stay free.

Kinematic **components**: 0 = stars, 1 = Balmer lines, 2 = forbidden lines. Lines in the same component
share velocity and width; recombination and collisionally excited lines need not."""),
        code(r"""sps = lib.sps_lib(ROOT / "output/ppxf/sps_models/spectra_emiles_9.0.npz", velscale, fwhm_gal, norm_range=[5070, 5950])
reg_dim = sps.templates.shape[1:]; stars = sps.templates.reshape(sps.templates.shape[0], -1)
print(f"{stars.shape[1]} stellar templates: {reg_dim[0]} ages x {reg_dim[1]} metallicities, {sps.age_grid.min():.2f}-{sps.age_grid.max():.1f} Gyr")
lam_range_gal = np.array([lam_gal[0], lam_gal[-1]]) / (1 + z)
gas, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, lam_range_gal, fwhm_gal, tie_balmer=False, limit_doublets=False)
print("gas templates:", list(gas_names))
templates = np.column_stack([stars, gas])
n_temps = stars.shape[1]; forb = np.array(["[" in n for n in gas_names])
component = [0] * n_temps + [2 if f else 1 for f in forb]; gas_component = np.array(component) > 0

fig, axs = plt.subplots(1, 2, figsize=(13, 3.6))
for (ia, im, lab) in [(0, 2, "youngest"), (reg_dim[0] // 2, 2, "middle age"), (reg_dim[0] - 1, 2, "oldest")]:
    t = sps.templates[:, ia, im]; v = (sps.lam_temp > 5070) & (sps.lam_temp < 5950)          # normalise in the V band, not over 0.1-5 µm
    axs[0].plot(sps.lam_temp, t / np.mean(t[v]), lw=0.6, label=f"{lab}: {sps.age_grid[ia, im]:.2f} Gyr")
axs[0].set(xlim=(3600, 7000), xlabel="rest wavelength [Å, air]", title="three stellar templates (solar metallicity)"); axs[0].legend(fontsize=8)
i = list(gas_names).index("Halpha"); axs[1].plot(sps.lam_temp, gas[:, i], lw=0.8, label="Hα template"); i2 = list(gas_names).index("[NII]6583_d"); axs[1].plot(sps.lam_temp, gas[:, i2], lw=0.8, label="[NII] doublet template (1 : 0.33)")
axs[1].set(xlim=(6520, 6620), xlabel="rest wavelength [Å, air]", title="gas templates = the instrumental LSF"); axs[1].legend(fontsize=8); plt.show()"""),
        md(r"""## 2.4 The fit

One call. Choices worth knowing: `moments=[2, 2, 2]` fits velocity and dispersion for each component;
`degree=-1, mdegree=10` uses a *multiplicative* polynomial only (it absorbs flux-calibration ripples
without touching the line strengths, unlike an additive one); `bounds` keep the narrow widths below
500 km/s; the sky line at 5577 Å and the telluric A band are excluded via `goodpixels`."""),
        code(r"""vel = C_KMS * np.log(1 + z); start = [[vel, 150.0], [vel, 100.0], [vel, 100.0]]
goodpixels = np.where(~bad_log & ~((lam_gal > 5573) & (lam_gal < 5583)) & ~((lam_gal > 7590) & (lam_gal < 7700)))[0]
bounds = [[[vel - 1500, vel + 1500], [10, 500]]] * 3
pp = ppxf(templates, galaxy, noise, velscale, start, moments=[2, 2, 2], degree=-1, mdegree=10, lam=lam_gal, lam_temp=sps.lam_temp,
          component=component, gas_component=gas_component, gas_names=gas_names, goodpixels=goodpixels, bounds=bounds, quiet=True)
print(f"chi2/dof = {pp.chi2:.2f}")
for k, name in enumerate(["stars", "Balmer", "forbidden"]):
    print(f"  {name:10s} V = {pp.sol[k][0]:8.1f} km/s (z = {np.exp(pp.sol[k][0] / C_KMS) - 1:.5f})   sigma = {pp.sol[k][1]:6.1f} km/s")
age, met = sps.mean_age_metal(pp.weights[:n_temps].reshape(reg_dim), quiet=True)
print(f"  light-weighted age {age:.1f} Gyr, [M/H] {met:+.2f}")
plt.figure(figsize=(13, 4)); pp.plot(); plt.title(f"step 4: {LABEL} — black data, red stars + gas, orange gas, green residuals"); plt.show()"""),
        md(r"""## 2.5 From weights to line fluxes

pPXF returns `pp.gas_flux` in the units of the input spectrum *per pixel*: multiply by the pixel size
in Å at the observed line (λ · velscale / c) and by the normalisation to get erg s⁻¹ cm⁻². For the fixed
doublet templates the strong line is the template flux divided by 1.33. Errors are the formal ones."""),
        code(r"""rows = []
for name, fl, er, lw in zip(gas_names, pp.gas_flux, pp.gas_flux_error, line_wave):
    if name in RP.GAS:
        key, fac = RP.GAS[name]; dlam = lw * (1 + z) * velscale / C_KMS
        rows.append(dict(line=name, key=key, flux=fl * dlam * norm / fac, err=er * dlam * norm / fac))
L = pd.DataFrame(rows).set_index("key"); L["snr"] = L.flux / L.err
print(L.round(2).to_string())
n2, o3 = np.log10(L.loc["nii", "flux"] / L.loc["ha", "flux"]), np.log10(L.loc["oiii", "flux"] / L.loc["hb", "flux"])
print(f"\nlog([NII]/Hα) = {n2:+.3f}   log([OIII]/Hβ) = {o3:+.3f}   Hα/Hβ = {L.loc['ha', 'flux'] / L.loc['hb', 'flux']:.2f}")"""),
        md(r"""## 2.6 Why the stellar continuum matters

`pp.bestfit - pp.gas_bestfit` is the stellar model. Under Hβ it dips into the Balmer absorption trough;
a straight line drawn across the line would sit above it and swallow part of the emission. The Hα panel
shows the same, weaker, effect."""),
        code(r"""stars_fit = pp.bestfit - pp.gas_bestfit; rest = lam_gal / (1 + z)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (l0, name) in zip(axs, [(4861.3, "Hβ"), (6562.8, "Hα")]):
    s = (rest > l0 - 60) & (rest < l0 + 60); side = s & (np.abs(rest - l0) > 25)
    p = np.polyfit(rest[side], galaxy[side], 1)
    ax.plot(rest[s], galaxy[s], color="0.3", lw=0.9, drawstyle="steps-mid", label="data"); ax.plot(rest[s], stars_fit[s], color="tab:orange", lw=1.3, label="pPXF stellar model")
    ax.plot(rest[s], np.polyval(p, rest[s]), "--", color="tab:purple", lw=1.2, label="straight line"); ax.plot(rest[s], pp.bestfit[s], "r", lw=0.9, label="stars + gas")
    core = s & (np.abs(rest - l0) < 10); ft = np.trapezoid((galaxy - stars_fit)[core], rest[core]); fn = np.trapezoid((galaxy - np.polyval(p, rest))[core], rest[core])
    ax.set_title(f"{name}: line flux with the straight line {fn / ft - 1:+.0%} relative to the stellar model", fontsize=9); ax.set_xlabel("rest wavelength [Å, air]")
axs[0].legend(fontsize=8); plt.show()"""),
        md(r"""## 2.7 Is there a broad component?

A type-1 AGN adds broad (thousands of km/s) Balmer lines. We add a **fourth component**, a second copy of
the Balmer templates with a width between 500 and 5000 km/s, refit, and keep it only if it is
*physical*: the total χ² drops by more than 25, the broad Hα has S/N > 5, its width is away from both
bounds, and its flux is at least 30 % of the narrow Hα. Without the last three conditions a "broad
component" gets adopted in most spectra as a continuum fudge parked at a bound. Compare IC 3147B with
NGC 1346 (`LABEL = "MCG-01-09-041_comp_sdss"`), a Seyfert 1."""),
        code(r"""balmer = ~forb; ib = int(np.where(gas_names == "Halpha")[0][0])
templates_b = np.column_stack([templates, gas[:, balmer]]); component_b = component + [3] * int(balmer.sum())
gas_names_b = np.concatenate([gas_names, [n + "_broad" for n in gas_names[balmer]]])
pp_b = ppxf(templates_b, galaxy, noise, velscale, start + [[vel, 1500.0]], moments=[2, 2, 2, 2], degree=-1, mdegree=10, lam=lam_gal, lam_temp=sps.lam_temp,
            component=component_b, gas_component=np.array(component_b) > 0, gas_names=gas_names_b, goodpixels=goodpixels,
            bounds=bounds + [[[vel - 3000, vel + 3000], [500, 5000]]], quiet=True)
ibb = int(np.where(gas_names_b == "Halpha_broad")[0][0])
dchi2 = (pp.chi2 - pp_b.chi2) * len(goodpixels); snr_b = pp_b.gas_flux[ibb] / pp_b.gas_flux_error[ibb]; sig_b = pp_b.sol[3][1]; ratio_b = pp_b.gas_flux[ibb] / pp_b.gas_flux[ib]
tests = dict(dchi2_gt_25=dchi2 > 25, snr_gt_5=snr_b > 5, sigma_in_range=600 < sig_b < 4500, broad_over_narrow_gt_0p3=ratio_b > 0.3)
print(f"Δχ² = {dchi2:.0f}   broad Hα S/N = {snr_b:.1f}   σ_broad = {sig_b:.0f} km/s   broad/narrow Hα = {ratio_b:.2f}")
print("tests:", tests, "->", "BROAD COMPONENT ADOPTED" if all(tests.values()) else "narrow-only model kept")
s = (rest > 6480) & (rest < 6650)
fig, ax = plt.subplots(figsize=(8, 3.8)); ax.plot(rest[s], galaxy[s], color="0.3", lw=0.9, drawstyle="steps-mid", label="data")
ax.plot(rest[s], pp.bestfit[s], "r", lw=1, label="narrow only"); ax.plot(rest[s], pp_b.bestfit[s], "b--", lw=1, label="narrow + broad trial")
ax.set(xlabel="rest wavelength [Å, air]", title=f"step 7: {LABEL}"); ax.legend(); plt.show()"""),
        md(r"""## 2.8 Cross-checks

`RP.fit_one` is the production function (same steps, plus the broad test): it should reproduce the
numbers above. For DESI spectra, FastSpecFit (the DESI collaboration's code) was also run; for SDSS
spectra the SDSS pipeline's own Gaussian fits (`SPZLINE`) exist. `results/tables/line_fitter_comparison.csv`
collects them: agreement to a few percent, except Hβ where the SDSS pipeline sits ~15 % low because it
under-corrects the Balmer absorption of 2.6."""),
        code(r"""prod = RP.fit_one(lam_vac, flux, ivar, fwhm, mask, z, ROOT / "output/ppxf/sps_models/spectra_emiles_9.0.npz", LABEL, plot=False)
cmp = pd.DataFrame({"notebook": L.flux, "pipeline (run_ppxf)": [prod[f"{k}_flux"] for k in L.index]}, index=L.index)
lf = pd.read_csv(ROOT / "results/tables/line_fitter_comparison.csv").set_index("label")
if LABEL in lf.index:
    for k in ("hb", "oiii", "ha", "nii"):
        cmp.loc[k, "FastSpecFit"] = lf.loc[LABEL, f"{k}_fsf"]
        if r.source == "SDSS": cmp.loc[k, "SDSS pipeline"] = lf.loc[LABEL, f"{k}_sdsspipe"]
print(cmp.round(1).to_string())"""),
        md(r"""## 2.9 The whole sample against the archival line fluxes

Every spectrum we fitted also has a published measurement: the **DESI DR1 value-added catalogue**
`stellar_mass_emline` (Gaussian fits after continuum subtraction, one row per DESI target, queried
from the NOIRLab Data Lab) and, for SDSS, the **SDSS pipeline** fits stored in each lite file
(`SPZLINE`: `LINEAREA`, `LINEAREA_ERR`). Same photons, different code — the comparison tells us how
much of a line flux is method. Points: one per spectrum of the sample, S/N > 3 on both sides.
Requires internet for the DESI part."""),
        code(r"""import requests, io
import compare_line_fitters as CLF
T = pd.read_csv(ROOT / "results/tables/ppxf_lines.csv").set_index("label")
KEYS = {"hb": ("hbeta", "Hβ"), "oiii": ("oiii5007", "[OIII]5007"), "ha": ("halpha", "Hα"), "nii": ("nii6583", "[NII]6584")}
arch = pd.DataFrame(index=T.index)
# DESI: one query for all targetids
ids = ",".join(T[T.source == "DESI"].spec_id.astype(str))
try:
    q = f"SELECT targetid, halpha_flux, halpha_fluxerr, hbeta_flux, hbeta_fluxerr, nii6583_flux, nii6583_fluxerr, oiii5007_flux, oiii5007_fluxerr FROM desi_dr1.stellar_mass_emline WHERE targetid IN ({ids})"
    rr = requests.post("https://datalab.noirlab.edu/tap/sync", data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q), timeout=180)
    v = pd.read_csv(io.StringIO(rr.text)).drop_duplicates("targetid").set_index("targetid")
    for lab, rw in T[T.source == "DESI"].iterrows():
        tid = int(rw.spec_id)
        if tid in v.index:
            for k, (vk, _) in KEYS.items():
                arch.loc[lab, f"{k}_arch"], arch.loc[lab, f"{k}_arch_err"] = v.loc[tid, f"{vk}_flux"], v.loc[tid, f"{vk}_fluxerr"]
except Exception as e:
    print("DESI VAC query failed (offline?):", e)
# SDSS: the pipeline's SPZLINE table of each lite file
for lab, rw in T[T.source == "SDSS"].iterrows():
    f = Path(tg.loc[lab, "file"])
    if f.exists():
        sp = CLF.spzline(f)
        for k in KEYS:
            arch.loc[lab, f"{k}_arch"], arch.loc[lab, f"{k}_arch_err"] = sp[f"{k}_sdsspipe"], sp[f"{k}_sdsspipe_err"]
A = T.join(arch)
fig, axs = plt.subplots(2, 3, figsize=(15, 9)); axs = axs.ravel()
for ax, (k, (_, name)) in zip(axs[:4], KEYS.items()):
    for src, mk, col in (("DESI", "o", "tab:blue"), ("SDSS", "^", "tab:orange")):
        g = A[(A.source == src) & (A[f"{k}_snr"] > 3) & (A[f"{k}_arch"] / A[f"{k}_arch_err"] > 3)]
        ax.errorbar(g[f"{k}_arch"], g[f"{k}_flux"], xerr=g[f"{k}_arch_err"], yerr=g[f"{k}_err"], fmt=mk, ms=4, capsize=2, color=col, label=f"{src}: {'DR1 VAC' if src == 'DESI' else 'SDSS pipeline'} (n={len(g)})")
        q = (g[f"{k}_flux"] / g[f"{k}_arch"]).values; q = q[np.isfinite(q) & (q > 0)]
        if len(q): ax.text(0.03, 0.92 - 0.07 * (src == "SDSS"), f"{src}: pPXF/archival median {np.median(q):.2f}, scatter {1.4826 * np.median(np.abs(np.log10(q) - np.median(np.log10(q)))):.2f} dex", transform=ax.transAxes, fontsize=8, color=col)
    lim = np.nanmax(A[[f"{k}_flux", f"{k}_arch"]].values) * 1.3; ax.plot([1, lim], [1, lim], "k--", lw=0.8)
    ax.set(xscale="log", yscale="log", xlim=(1, lim), ylim=(1, lim), xlabel=f"{name} archival [1e-17 erg/s/cm²]", ylabel=f"{name} pPXF"); ax.legend(fontsize=7, loc="lower right")
for ax, (a_, b_, lab_) in zip(axs[4:], [("nii", "ha", "log([NII]/Hα)"), ("oiii", "hb", "log([OIII]/Hβ)")]):
    ok = (A[f"{a_}_snr"] > 3) & (A[f"{b_}_snr"] > 3) & (A[f"{a_}_arch"] > 0) & (A[f"{b_}_arch"] > 0)
    x = np.log10(A[f"{a_}_arch"] / A[f"{b_}_arch"]); y = np.log10(A[f"{a_}_flux"] / A[f"{b_}_flux"])
    for src, mk, col in (("DESI", "o", "tab:blue"), ("SDSS", "^", "tab:orange")):
        m = ok & (A.source == src); ax.plot(x[m], y[m], mk, ms=5, color=col, label=src)
        d_ = (y - x)[m].dropna(); ax.text(0.03, 0.92 - 0.07 * (src == "SDSS"), f"{src}: pPXF − archival = {d_.median():+.3f} ± {1.4826 * np.median(np.abs(d_ - d_.median())):.3f} dex", transform=ax.transAxes, fontsize=8, color=col)
    lo, hi = (-1.0, 0.5) if a_ == "nii" else (-1.2, 1.2)               # the BPT range; a few pipeline outliers fall outside
    nout = int(((x[ok] < lo) | (x[ok] > hi) | (y[ok] < lo) | (y[ok] > hi)).sum())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8); ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=lab_ + " archival", ylabel=lab_ + " pPXF", title=f"{nout} outside the frame" if nout else ""); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("step 9: pPXF line fluxes of the whole sample against the archival measurements"); fig.tight_layout(); plt.show()
A[[f"{k}_flux" for k in KEYS] + [f"{k}_arch" for k in KEYS]].to_csv(ROOT / "output/notebooks/archival_flux_comparison.csv")"""),
        md(r"""What to take from it: [O III], Hα and [N II] agree with both archives to a few percent, with 0.01 dex
scatter against the SDSS pipeline — those lines are robust to the method. Hβ is the exception: pPXF is
~15 % above the SDSS pipeline and a few percent off the DESI catalogue, because the codes treat the stellar
absorption under Hβ differently. On the BPT the [N II]/Hα axis is therefore method-independent; the
[O III]/Hβ axis carries a method systematic of a few hundredths of a dex, comparable to the
distance between the Kauffmann and Kewley lines for our composite objects. Classifications near a line
should be quoted with that in mind."""),
        md(r"""## Exercises

1. Run the notebook on the SDSS spectrum of the same galaxy (`IC3147B_sdss`, 3″ fibre): which line
   fluxes change most, which ratios least? Why?
2. Set `mdegree=0` and refit. What happens to the continuum fit and to the Hβ flux?
3. Tie the Balmer lines to their case-B ratios (`tie_balmer=True` in `emission_lines`, then `gas_reddening=0`
   in `ppxf`): pPXF now fits a gas reddening. Compare it with the Hα/Hβ value of 2.5.
4. Loosen the broad-line test to Δχ² > 25 alone. How many of the four criteria does IC 3147B pass? Which
   parameter is at a bound?
5. `pp.weights[:n_temps]` are the light fractions of the stellar templates: plot them on the age-metallicity
   grid. Is the nucleus old or young? Does `sps.mean_age_metal` agree with your eye?""")]
    return C


SETUP3 = r"""import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # one BLAS thread: on Binder / Colab the pod has
    os.environ.setdefault(_v, "1")                                          # one core and multi-threaded BLAS is 30x slower
import os, io, sys
from pathlib import Path
import numpy as np, pandas as pd, requests
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import plot_bpt_ppxf as PB                # the project's BPT plotting (one point per galaxy, pairs joined)
import bpt_comparison as BC               # comparison samples and the stage-vs-population statistics
plt.rcParams.update({"figure.dpi": 110})
STAGE = {0: "pre-interaction", 1: "merger", 2: "post-merger"}
print("repo:", ROOT)"""


def nb03():
    C = [md(r"""# 3. Your galaxy on the BPT diagram

With pPXF line fluxes for every nucleus of the sample (`results/tables/ppxf_lines.csv`: 54 spectra, main
galaxies and companions, DESI and SDSS) each of you takes **one system** — a galaxy, a pair, or a few — and
answers for it: *stars or black hole in the nucleus?* and *does my galaxy look like a typical interacting
galaxy, like a typical barred galaxy, or like neither?*

Tools: the BPT diagram (Baldwin, Phillips & Terlevich 1981) with the **Kauffmann et al. (2003)** and
**Kewley et al. (2001)** demarcations (star-forming / composite / AGN) and the **Schawinski et al. (2007)**
Seyfert–LINER line; and two large SDSS comparison populations with the same lines measured by the MPA-JHU
group (Brinchmann et al. 2004): **spectroscopic pairs** (a companion within 30 kpc and 500 km/s) and
**barred galaxies** (visual bars of Nair & Abraham 2010). Their distributions are drawn as contours
enclosing 20, 50 and 80 % of each population."""),
        code(SETUP3),
        md(r"""## 3.1 Your system

`MY_SYSTEMS` comes from notebook 1 (`notebooks/my_systems.json`). Every spectrum of those systems is
kept: the main galaxy, its companion if there is one, and both the DESI and the SDSS spectrum where both
exist (different fibres: 1.5″ and 3″)."""),
        code(r"""import json
MY_SYSTEMS = json.loads((ROOT / "notebooks/my_systems.json").read_text()) if (ROOT / "notebooks/my_systems.json").exists() else ["IC3147B"]   # set in notebook 1
t = pd.read_csv(ROOT / "results/tables/ppxf_lines.csv")
tg = pd.read_csv(ROOT / "dataset/catalogs/interbars_targets.csv").set_index("system")
sz = pd.read_csv(ROOT / "dataset/catalogs/interbars_specz.csv").set_index("system")
t["stage"] = t.system.map(tg.class_VC).map(STAGE).fillna("unclassified")
mine = t[t.system.isin(MY_SYSTEMS)].copy()
print(f"available systems: {sorted(t.system.unique())}\n")
for s_ in MY_SYSTEMS:
    r = tg.loc[s_]; print(f"{s_}: z = {r.z_hand:.4f}, VC class {r.class_VC:.0f} ({STAGE.get(r.class_VC, '?')}), EB class {r.class_EB:.0f}; companion: {r.comp_name if isinstance(r.comp_name, str) else 'none'}"
          + (f" at {sz.loc[s_, 'comp_sep_kpc']:.0f} kpc, dv {sz.loc[s_, 'dv_comp_kms']:+.0f} km/s" if isinstance(r.comp_name, str) else "") + (f"\n   note: {r.note}" if isinstance(r.note, str) else ""))
cols = ["label", "role", "source", "z_fit", "ha_flux", "ha_err", "hb_flux", "hb_err", "oiii_flux", "oiii_err", "nii_flux", "nii_err", "log_nii_ha", "log_nii_ha_err", "log_oiii_hb", "log_oiii_hb_err", "min_snr_bpt", "bpt_class", "broad_adopted"]
print(); print(mine[cols].round(3).to_string(index=False))"""),
        md(r"""## 3.2 Your nuclei on the BPT, over the pairs and the barred galaxies

Large symbols with labels: your spectra (circle = DESI, triangle = SDSS; a line joins the two nuclei of a
pair; a ring marks a detected broad Hα). Small grey points: the rest of the sample. Blue contours: SDSS
pairs; purple dashed: SDSS barred galaxies."""),
        code(r"""sdss = BC.ratios(BC.load_sdss())
sdss["pair"] = BC.sdss_pairs(sdss, 30.0, 500.0); sdss["in_nair"], sdss["bar"], sdss["napair"], sdss["inter"] = BC.nair_match(sdss)
pops = {"pairs": sdss[sdss.pair], "barred": sdss[sdss.bar]}
grids = {k: BC.density_grid(v.n2.values, v.o3.values) for k, v in pops.items()}
STYLE = {"pairs": ("#1f5fbf", "solid"), "barred": ("#8e2ca3", "dashed")}
ok = t.min_snr_bpt >= 3

def bpt_with_contours(ax, panel="n2"):
    xcol = "log_nii_ha" if panel == "n2" else "log_sii_ha"
    for k, P in pops.items():
        xc, yc, H, lev = BC.contour_levels(P[panel].values, P.o3.values, (-1.3, 0.7) if panel == "n2" else (-1.2, 0.6), (-1.3, 1.3))
        ax.contour(xc, yc, H, levels=lev, colors=[STYLE[k][0]], linewidths=1.3, linestyles=STYLE[k][1]); ax.plot([], [], color=STYLE[k][0], ls=STYLE[k][1], label=f"SDSS {k}: {len(P):,}")
    rest = t[ok & ~t.system.isin(MY_SYSTEMS)]; ax.plot(rest[xcol], rest.log_oiii_hb, "o", ms=3.5, color="0.6", mec="none", label="rest of the sample")
    PB.demarcations(ax, panel)

fig, ax = plt.subplots(figsize=(9, 8)); bpt_with_contours(ax, "n2")
m = mine[mine.min_snr_bpt >= 3]
for s_, g in m.groupby("system"):
    for src in ("DESI", "SDSS"):
        gg = g[g.source == src]
        if len(gg) == 2:                                     # main + companion of this system, same survey: join them
            ax.plot(gg.log_nii_ha, gg.log_oiii_hb, "-", color=PB.COL.get(gg.stage.iloc[0], "k") if False else "0.3", lw=1.2, zorder=4)
for r in m.itertuples():
    col = {"pre-interaction": "#c8102e", "merger": "#e8871e", "post-merger": "#2a9d3f"}.get(r.stage, "0.4")
    ax.errorbar(r.log_nii_ha, r.log_oiii_hb, xerr=r.log_nii_ha_err, yerr=r.log_oiii_hb_err, fmt="o" if r.source == "DESI" else "^", ms=11, mfc=col, mec="k", mew=0.8, ecolor=col, capsize=3, zorder=6)
    if r.broad_adopted: ax.plot(r.log_nii_ha, r.log_oiii_hb, "o", ms=19, mfc="none", mec="k", mew=1.2, zorder=7)
    ax.annotate(f"{r.label}", (r.log_nii_ha, r.log_oiii_hb), xytext=(8, 6), textcoords="offset points", fontsize=8, zorder=8)
ax.plot([], [], "o", ms=9, mfc="w", mec="k", label="your DESI spectrum"); ax.plot([], [], "^", ms=9, mfc="w", mec="k", label="your SDSS spectrum")
ax.legend(fontsize=8, loc="lower right"); ax.set_title(f"{', '.join(MY_SYSTEMS)} on the [N II] BPT"); plt.show()"""),
        md(r"""## 3.3 Which population does your nucleus agree with?

For each of your spectra: the BPT class; the **contour level** at which the point sits in each population
(0.3 means "inside the contour that encloses 30 % of that population", i.e. in its dense core; 0.95 means
in its outskirts); the **density ratio** pairs / barred at the point (how many times more likely a pair
than a barred galaxy is to have these line ratios); and a verdict. A nucleus can be typical of both — the
two populations overlap on the star-forming sequence — so read the ratio, not only the verdict."""),
        code(r"""def contour_level_at(grid, x, y):
    # fraction of the population enclosed by the contour passing through (x, y)
    xe, ye, H = grid; dens = BC.eval_density(grid, np.atleast_1d(x), np.atleast_1d(y))[0]
    srt = np.sort(H.ravel())[::-1]; return float(np.cumsum(srt)[np.searchsorted(-srt, -dens)])

rows = []
for r in m.itertuples():
    lv = {k: contour_level_at(grids[k], r.log_nii_ha, r.log_oiii_hb) for k in pops}
    dens = {k: BC.eval_density(grids[k], np.atleast_1d(r.log_nii_ha), np.atleast_1d(r.log_oiii_hb))[0] for k in pops}
    ratio = dens["pairs"] / max(dens["barred"], 1e-12)
    verdict = ("typical of both" if lv["pairs"] < 0.8 and lv["barred"] < 0.8 else "typical of pairs only" if lv["pairs"] < 0.8 else "typical of barred galaxies only" if lv["barred"] < 0.8 else "outskirts of both")
    verdict += f"; {'pairs' if ratio > 2 else 'barred' if ratio < 0.5 else 'either'} favoured ({ratio:.1f}x)"
    rows.append(dict(label=r.Index if False else r.label, source=r.source, bpt_class=r.bpt_class, level_pairs=lv["pairs"], level_barred=lv["barred"], pairs_over_barred=ratio, verdict=verdict))
verdicts = pd.DataFrame(rows); pd.set_option("display.width", 220); print(verdicts.round(2).to_string(index=False))"""),
        md(r"""## 3.4 Your galaxy's numbers

Everything pPXF measured, in one table per spectrum: line fluxes with errors, the Balmer decrement and the
gas reddening it implies ($E(B-V) = 1.97\,\log_{10}[({\rm H}\alpha/{\rm H}\beta)/2.86]$, Calzetti law),
the dust-corrected Hα flux, the stellar velocity dispersion and light-weighted age, the gas kinematics, and
the broad-line test. The [S II] and [O I] ratios give the two other diagnostic diagrams (Kewley et al. 2006);
compute their classes as an exercise."""),
        code(r"""def calzetti_k(w_A):
    w = w_A / 1e4
    return np.where(w < 0.63, 2.659 * (-2.156 + 1.509 / w - 0.198 / w**2 + 0.011 / w**3) + 4.05, 2.659 * (-1.857 + 1.040 / w) + 4.05)
mine["ebv_gas"] = (1.97 * np.log10(mine.balmer_dec / 2.86)).clip(lower=0)
mine["ha_flux_dustcorr"] = mine.ha_flux * 10 ** (0.4 * mine.ebv_gas * calzetti_k(np.array([6564.6]))[0])
show = ["label", "source", "z_fit", "hb_flux", "hb_err", "oiii_flux", "oiii_err", "ha_flux", "ha_err", "nii_flux", "nii_err", "sii6716_flux", "sii6731_flux", "oi_flux",
        "balmer_dec", "ebv_gas", "ha_flux_dustcorr", "log_nii_ha", "log_oiii_hb", "log_sii_ha", "log_oi_ha", "bpt_class",
        "sigma_star", "age_lw_gyr", "metal_lw", "sigma_balmer", "sigma_forbidden", "broad_adopted", "dchi2_broad", "snr_broad_halpha", "chi2"]
print(mine[show].set_index("label").T.round(3).to_string())"""),
        md(r"""## 3.5 The whole sample, for context

The same diagram with every nucleus of the sample, one spectrum per galaxy (DESI where both exist),
coloured by the VC interaction class, both nuclei of a pair joined; and the per-class statistics against
the two populations (fraction of nuclei inside the 80 % contour, mean log-density, and a two-dimensional
energy-distance permutation test). Where does your system sit in that picture?"""),
        code(r"""pts, pairs = PB.load_sample()
fig, ax = plt.subplots(figsize=(9, 8)); bpt_with_contours(ax, "n2"); ax.lines[-1].remove()      # contours + demarcations, without the grey sample points
PB.draw_panel(ax, pts, pairs, "n2"); PB.sample_legend(ax, loc="lower right"); plt.show()
stats = BC.stage_consistency(pts, {**pops, "all": sdss}, nboot=200, nperm=200, seed=1)
print(stats[["stage", "n", "population", "frac_in80", "mean_logdens", "p_energy", "nonsf_frac_sample", "nonsf_frac_pop"]].round(3).to_string(index=False))"""),
        md(r"""## Exercises

1. Classify your nuclei on the [S II] and [O I] diagrams (Kewley et al. 2006 curves are in
   `scripts/plot_bpt_ppxf.py`). Do the three diagrams agree?
2. If your galaxy has both a DESI and an SDSS spectrum: the fibres are 1.5″ and 3″. Which fluxes change,
   which ratios? What does that say about where the line emission comes from?
3. Move your point by ±1σ along both axes. Does the class change? Does the verdict of 3.3?
4. Look at your galaxy's image (notebook 1) and the classification note: is the VC class consistent with
   what you see? Would you call the companion a companion?
5. Compute EW(Hα) from notebook 2 (gas flux / stellar continuum under the line) and place your nucleus on
   the WHAN diagram (Cid Fernandes et al. 2011: EW < 3 Å = "retired"). Does it change your answer?""")]
    return C


# ======================================================================================= 01 (MY_SYSTEMS is set here)
SETUP1 = r"""import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # one BLAS thread: on Binder / Colab the pod has
    os.environ.setdefault(_v, "1")                                          # one core and multi-threaded BLAS is 30x slower
import os, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, fitsio, requests
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.wcs import WCS
from PIL import Image

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DESI_SPECTRO_REDUX", str(ROOT / "output" / "desi_redux"))
import run_ppxf as RP                      # the project's spectrum inventory and readers (DESI and SDSS)
CACHE = ROOT / "output" / "notebooks"; CACHE.mkdir(parents=True, exist_ok=True)
CFG = ROOT / "notebooks" / "my_systems.json"          # the systems chosen here are read by notebooks 2 and 3
C_KMS = 299792.458
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
STAGE = {0: "pre-interaction", 1: "merger", 2: "post-merger"}
plt.rcParams.update({"figure.dpi": 110})
targets = pd.read_csv(ROOT / "dataset/catalogs/interbars_targets.csv").set_index("system")
specz = pd.read_csv(ROOT / "dataset/catalogs/interbars_specz.csv").set_index("system")
inventory = RP.targets().set_index("label")           # every DESI / SDSS spectrum of the sample

def desi_row(coadd, targetid):
    # the FIBERMAP row and the three arms (wave, flux, ivar, mask, resolution) of one DESI spectrum
    fm = fitsio.read(coadd, "FIBERMAP"); i = int(np.where(fm["TARGETID"] == int(targetid))[0][0]); arms = {}
    with fitsio.FITS(coadd) as F:
        for b in "BRZ":
            arms[b] = dict(wave=F[f"{b}_WAVELENGTH"].read(), flux=F[f"{b}_FLUX"][i:i + 1, :][0], ivar=F[f"{b}_IVAR"][i:i + 1, :][0],
                           mask=F[f"{b}_MASK"][i:i + 1, :][0], res=F[f"{b}_RESOLUTION"][i:i + 1, :, :][0])
    return fm[i], arms

def sdss_file(path):
    # header, COADD table and SPECOBJ row of an SDSS lite file (BOSS-era files carry unnamed HDUs)
    with fitsio.FITS(path) as F:
        names = [h.get_extname() for h in F]
        return F[0].read_header(), F[names.index("COADD") if "COADD" in names else 1].read(), F[names.index("SPECOBJ") if "SPECOBJ" in names else 2].read()[0]

def ls_cutout(ra, dec, size_arcsec, pixscale=0.262, layer="ls-dr10"):
    # Legacy Survey colour JPEG + a WCS built from the request (TAN projection, north up, east left)
    npix = int(round(size_arcsec / pixscale)); f = CACHE / f"ls_{ra:.5f}_{dec:.5f}_{size_arcsec:.0f}_{pixscale}.jpg"
    if not f.exists():
        r = requests.get("https://www.legacysurvey.org/viewer/jpeg-cutout", params=dict(ra=ra, dec=dec, layer=layer, pixscale=pixscale, size=npix), timeout=120)
        r.raise_for_status(); f.write_bytes(r.content)
    im = np.flipud(np.asarray(Image.open(f)))
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(im.shape[1] + 1) / 2, (im.shape[0] + 1) / 2]; w.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    return im, w
print("repo:", ROOT)"""


def nb01():
    C = [md(r"""# 1. Finding the spectra, and what a fibre spectrum is

**interbars**: barred galaxies in interacting pairs, at $z \approx 0.01$–$0.05$. This series asks whether
the gas in their **nuclei** is ionised by young stars or by an active nucleus. The data are optical
fibre spectra from public surveys (DESI, SDSS). Before measuring anything we need to know:

* which spectra exist for each galaxy, and *where exactly* on the galaxy the fibre was placed;
* how big the fibre is on the sky and in kiloparsecs at the galaxy's distance;
* what a spectrum file contains: flux units, uncertainties, the wavelength grid (pixel scale), the
  **spectral resolution** (how blurred a narrow line becomes), and bad-pixel masks.

Each of you works on **one system** (a galaxy, a pair, or a few) through the three notebooks. You choose
it once, in section 1.2 of this notebook; notebooks 2 (line measurements with pPXF) and 3 (BPT
classification) pick it up from there. Environment: the `fastspecfit` conda env
(`source scripts/fastspecfit_env.sh`)."""),
        code(SETUP1),
        md(r"""## 1.1 The sample

One row per system: the main galaxy, its position (resolved through SIMBAD and checked on the images),
the redshift, the interaction stage from the visual classification (`class_VC`: 0 = pre-interaction,
1 = merger, 2 = post-merger), and a named companion where there is one."""),
        code(r"""cols = ["ra", "dec", "z_hand", "class_EB", "class_VC", "comp_name", "comp_z_hand"]
print(targets[cols].round(5).to_string())"""),
        md(r"""## 1.2 Choose your system

Set `MY_SYSTEMS` to one or more names of the table above. The choice is saved to
`notebooks/my_systems.json` and read by the other notebooks. The card below lists what the archives
hold for it: every DESI and SDSS spectrum of the main galaxy and of its companion."""),
        code(r"""MY_SYSTEMS = ["IC3147B"]                       # <-- set once, here; e.g. ["NGC2864"], ["MCG-01-09-041"], ["UGC6142", "Z44-80"]
CFG.write_text(json.dumps(MY_SYSTEMS)); print("saved to", CFG.relative_to(ROOT), "->", MY_SYSTEMS, "\n")
for s_ in MY_SYSTEMS:
    r = targets.loc[s_]
    print(f"{s_}: RA {r.ra:.5f} Dec {r.dec:.5f}, z = {r.z_hand:.4f}, VC class {r.class_VC:.0f} ({STAGE.get(r.class_VC, '?')}), EB class {r.class_EB:.0f}"
          + (f"; companion {r.comp_name} at {specz.loc[s_, 'comp_sep_kpc']:.0f} kpc, dv {specz.loc[s_, 'dv_comp_kms']:+.0f} km/s" if isinstance(r.comp_name, str) else "; no companion"))
    if isinstance(r.note, str): print("   note:", r.note)
    print(inventory[inventory.system == s_][["role", "source", "spec_id", "z_in"]].to_string())
    print()
SYSTEM = MY_SYSTEMS[0]; z = float(targets.loc[SYSTEM, "z_hand"]); ra, dec = float(targets.loc[SYSTEM, "ra"]), float(targets.loc[SYSTEM, "dec"])
kpc_per_arcsec = 1 / COSMO.arcsec_per_kpc_proper(z).value"""),
        md(r"""## 1.3 Searching the archives for spectra

Spectroscopic surveys publish **catalogues** (one row per spectrum: position, redshift, quality) that
can be queried with SQL over the web. Two are relevant here: **SDSS DR17** (2000–2020, 3″ then 2″
fibres) and **DESI DR1** (2021–2022, 1.5″ fibres), both served by the NOIRLab Astro Data Lab. The query
asks for every spectrum within 30″ of each of your galaxies. The full project search (9 surveys, 10′
around every galaxy, `dataset/catalogs/specz_matches.csv`) used exactly these queries."""),
        code(r"""def datalab(sql):
    r = requests.post("https://datalab.noirlab.edu/tap/sync", data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=sql), timeout=120)
    r.raise_for_status(); return pd.read_csv(io.StringIO(r.text))
R = 30 / 3600
for s_ in MY_SYSTEMS:
    ra_, dec_ = float(targets.loc[s_, "ra"]), float(targets.loc[s_, "dec"])
    box = lambda racol, deccol: f"{racol} BETWEEN {ra_ - R / np.cos(np.radians(dec_))} AND {ra_ + R / np.cos(np.radians(dec_))} AND {deccol} BETWEEN {dec_ - R} AND {dec_ + R}"
    try:
        sdss = datalab(f"SELECT specobjid, plate, mjd, fiberid, ra, dec, z, zerr, zwarning, class, survey FROM sdss_dr17.specobj WHERE {box('ra', 'dec')}")
        desi = datalab(f"SELECT targetid, mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zerr, zwarn, spectype, survey, program, healpix, zcat_primary FROM desi_dr1.zpix WHERE {box('mean_fiber_ra', 'mean_fiber_dec')}")
        for name, d in (("SDSS DR17", sdss), ("DESI DR1", desi)):
            d["sep_arcsec"] = 3600 * np.hypot((d.ra - ra_) * np.cos(np.radians(dec_)), d.dec - dec_)
            print(f"== {s_} / {name}: {len(d)} spectra within 30\""); print(d.sort_values("sep_arcsec").round(5).to_string(index=False)); print()
    except Exception as e:
        print("offline? using the project's cached search instead:", e)
        m = pd.read_csv(ROOT / "dataset/catalogs/specz_matches.csv"); print(m[(m.system == s_) & (m.sep_arcsec < 30) & m.survey.isin(["SDSS DR17", "DESI DR1"])][["survey", "obj_id", "z", "zqual", "sep_arcsec"]].to_string(index=False))"""),
        md(r"""Things to notice: several rows can be the *same galaxy* (DESI observes a target in several programs;
`zcat_primary` marks the best one), rows tens of arcseconds away are *other* galaxies (or the companion),
and the quality flags (`zwarning`, `zwarn`) tell whether the pipeline trusted its redshift.

## 1.4 The image, the WCS, and where the fibres sit

A **WCS** (world coordinate system) maps image pixels to sky coordinates. For a small cutout a
tangent-plane projection is enough: a reference pixel `CRPIX` at `CRVAL` (RA, Dec) and a pixel scale
`CDELT` in degrees per pixel (negative in RA: east is left). We build one for a Legacy Survey DR10
colour image (0.262″ native pixels) and draw every fibre of the system on it:

* DESI: 1.5″ diameter at `MEAN_FIBER_RA/DEC` (where the fibre actually was, averaged over exposures);
* SDSS: 3″ diameter at `PLUG_RA/DEC` (from the spectrum file header)."""),
        code(r"""FOV = 60.0
for s_ in MY_SYSTEMS:
    r = targets.loc[s_]; ra_, dec_, z_ = float(r.ra), float(r.dec), float(r.z_hand); kpc = 1 / COSMO.arcsec_per_kpc_proper(z_).value
    im, w = ls_cutout(ra_, dec_, FOV)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(projection=w)); ax.imshow(im, origin="lower")
    def circle(ra_c, dec_c, diam, **kw):
        x, y = w.world_to_pixel_values(ra_c, dec_c); ax.add_patch(plt.Circle((x, y), diam / 2 / 0.262, fill=False, **kw))
    fibre_frac = None
    for lab, sp in inventory[inventory.system == s_].iterrows():
        if sp.source == "DESI":
            fm, _ = desi_row(sp.file, sp.spec_id); circle(fm["MEAN_FIBER_RA"], fm["MEAN_FIBER_DEC"], 1.5, color="cyan", lw=1.6, ls="-" if sp.role == "main" else ":", label=f'DESI 1.5" ({lab})')
            if sp.role == "main" and fm["FLUX_R"] > 0: fibre_frac = fm["FIBERFLUX_R"] / fm["FLUX_R"]
        elif Path(sp.file).exists():
            h = sdss_file(sp.file)[0]; circle(h["PLUG_RA"], h["PLUG_DEC"], 3.0, color="orange", lw=1.6, ls="--" if sp.role == "main" else ":", label=f'SDSS 3" ({lab})')
    x0, y0 = w.world_to_pixel_values(ra_, dec_); ax.plot(x0, y0, "+", color="w", ms=12, mew=1, label="catalogue centre")
    ax.plot([10, 10 + 5 / kpc / 0.262], [10, 10], color="w", lw=2); ax.text(10, 16, "5 kpc", color="w")
    ax.set(xlabel="RA", ylabel="Dec", title=f"{s_}  z={z_:.4f}  {kpc:.3f} kpc/\"  LS DR10 {FOV:.0f}\""); ax.legend(loc="upper right", fontsize=8); plt.show()
    print(f"{s_}: 1 arcsec = {kpc:.3f} kpc -> DESI fibre 1.5\" = {1.5 * kpc:.2f} kpc, SDSS fibre 3\" = {3 * kpc:.2f} kpc" + (f"; DESI fibre / total r-band flux = {fibre_frac:.1%}" if fibre_frac else ""))"""),
        md(r"""The fibre sees only the nucleus: a few percent of the galaxy's light, the central kiloparsec.
Everything we conclude later is about *that region*, not the galaxy as a whole (the bar, the disc and
the tidal features are outside the fibre).

## 1.5 The spectrum file

The spectrum used below is the main galaxy's DESI spectrum when there is one, else its SDSS one
(`SPEC`). Two file layouts:

* **DESI**: a healpix coadd holds hundreds of spectra; `FIBERMAP` identifies the rows, and each
  spectrograph arm (B, R, Z) has its own `WAVELENGTH`, `FLUX` ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$),
  `IVAR` (inverse variance, $1/\sigma^2$), `MASK` (bit flags) and `RESOLUTION` extensions.
* **SDSS**: one "lite" file per spectrum: the `COADD` table (`loglam`, `flux`, `ivar`, `and_mask`,
  `wdisp`, `sky`, `model`), the pipeline's redshift and class in `SPECOBJ`, its line fits in `SPZLINE`."""),
        code(r"""cand = inventory[(inventory.system == SYSTEM) & (inventory.role == "main")].sort_values("source")   # DESI before SDSS
SPEC = cand.index[0]; sp = inventory.loc[SPEC]; print("spectrum used below:", SPEC, f"({sp.source})")
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
if sp.source == "DESI":
    fmrow, arms = desi_row(sp.file, sp.spec_id)
    print(f"targetid {sp.spec_id}: {fmrow['COADD_NUMEXP']} exposures, {fmrow['COADD_EXPTIME']:.0f} s, {fmrow['COADD_NUMTILE']} tile(s)")
    for b, c in zip("BRZ", ("tab:blue", "tab:green", "tab:red")):
        a = arms[b]; ok = a["ivar"] > 0; err = np.where(ok, 1 / np.sqrt(np.where(ok, a["ivar"], 1)), np.nan)
        axs[0].plot(a["wave"], np.where(ok, a["flux"], np.nan), lw=0.5, color=c, label=f"{b} arm {a['wave'][0]:.0f}–{a['wave'][-1]:.0f} Å")
        axs[0].fill_between(a["wave"], a["flux"] - err, a["flux"] + err, color=c, alpha=0.2, lw=0)
        axs[1].plot(a["wave"], err, lw=0.5, color=c); axs[1].plot(a["wave"][a["mask"] > 0], np.zeros((a["mask"] > 0).sum()), "|", color="k", ms=8)
    wave = np.concatenate([arms[b]["wave"] for b in "BRZ"]); flux = np.concatenate([arms[b]["flux"] for b in "BRZ"]); ivar = np.concatenate([arms[b]["ivar"] for b in "BRZ"])
    o = np.argsort(wave); wave, flux, ivar = wave[o], flux[o], ivar[o]; sig_kms_inst = None; ttl = "the three DESI arms"
else:
    hdr, d, so = sdss_file(sp.file); c = {k.lower(): k for k in d.dtype.names}
    wave = 10 ** d[c["loglam"]]; flux = d[c["flux"]]; ivar = d[c["ivar"]]; ok = ivar > 0; err = np.where(ok, 1 / np.sqrt(np.where(ok, ivar, 1)), np.nan)
    print(f"plate {hdr['PLATEID']} mjd {hdr['MJD']} fiber {hdr['FIBERID']}: {hdr.get('NEXP', '?')} exposures, {hdr['EXPTIME']:.0f} s; pipeline z = {float(so['Z']):.5f} ({str(so['CLASS']).strip()})")
    axs[0].plot(wave, np.where(ok, flux, np.nan), lw=0.5, color="tab:orange", label=f"SDSS {wave[0]:.0f}–{wave[-1]:.0f} Å"); axs[0].fill_between(wave, flux - err, flux + err, color="tab:orange", alpha=0.2, lw=0)
    axs[1].plot(wave, err, lw=0.5, color="tab:orange"); axs[1].plot(wave[~ok], np.zeros((~ok).sum()), "|", color="k", ms=8)
    sig_kms_inst = d[c["wdisp"]] * C_KMS * np.log(10) * 1e-4; ttl = "the SDSS spectrum"
axs[0].set(ylabel="flux [1e-17 erg/s/cm²/Å]", title=f"{SPEC}: {ttl} (shaded: ±1σ)"); axs[0].legend()
axs[1].set(xlabel="observed wavelength [Å]", ylabel="1σ error", ylim=(0, None)); axs[1].text(0.01, 0.8, "| = masked pixel (ivar = 0)", transform=axs[1].transAxes, fontsize=8); plt.show()"""),
        md(r"""## 1.6 Pixel scale

DESI's grid is linear, 0.8 Å per pixel: in velocity units, $c\,\Delta\lambda/\lambda$, 67 km/s at the
blue end and 24 km/s at the red end. SDSS's grid is logarithmic: a constant 69 km/s per pixel. A galaxy
line with $\sigma \approx 100$ km/s is sampled by only a few pixels — which is why the resolution, not
just the pixel size, matters."""),
        code(r"""dlam = np.diff(wave); print(f"pixel step: {np.median(dlam):.2f} Å (min {dlam[dlam > 0].min():.2f}, max {dlam.max():.2f}); {np.median(C_KMS * dlam / wave[1:]):.0f} km/s median")
fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(wave[1:], C_KMS * dlam / wave[1:], lw=0.8); ax.set(xlabel="observed wavelength [Å]", ylabel="pixel [km/s]", ylim=(0, 80), title=f"velocity size of one pixel ({sp.source})"); plt.show()"""),
        md(r"""## 1.7 Spectral resolution

A perfectly narrow line is recorded as a blurred profile, the **line-spread function** (LSF). DESI stores
it explicitly: the `RESOLUTION` extension is a band matrix with 11 diagonals per pixel — the LSF centred
on that pixel, in units of pixels; we fit a Gaussian to it at several wavelengths. SDSS stores a
per-pixel `wdisp` (Gaussian σ in pixels of its log grid). Both are converted to the resolving power
$R = \lambda / {\rm FWHM}$ and to $\sigma_{\rm inst}$ in km/s. Whatever your spectrum is, both surveys
are drawn (the other one from a reference spectrum of the sample) so you can compare them."""),
        code(r"""from scipy.optimize import curve_fit
g = lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2)
REF_DESI = inventory.loc["IC3147B"]; REF_SDSS = inventory.loc["IC3147B_comp_sdss"]           # reference spectra, always available
def desi_resolution(arms):
    rows = []
    for b in "BRZ":
        a = arms[b]; nd = a["res"].shape[0]; off = np.arange(nd) - nd // 2
        for j in np.linspace(50, len(a["wave"]) - 50, 6).astype(int):
            prof = a["res"][:, j]
            if prof.sum() < 0.5: continue
            (A, mu, s_), _ = curve_fit(g, off, prof, p0=[prof.max(), 0, 1.5]); dl = a["wave"][1] - a["wave"][0]; lam = a["wave"][j]
            rows.append(dict(arm=b, wave=lam, sigma_pix=abs(s_), sigma_A=abs(s_) * dl, R=lam / (2.3548 * abs(s_) * dl), sigma_kms=C_KMS * abs(s_) * dl / lam))
    return pd.DataFrame(rows)
arms_d = arms if sp.source == "DESI" else desi_row(REF_DESI.file, REF_DESI.spec_id)[1]
d_s = sdss_file(sp.file if sp.source == "SDSS" else REF_SDSS.file)[1]; c = {k.lower(): k for k in d_s.dtype.names}
lam_s = 10 ** d_s[c["loglam"]]; sig_s = d_s[c["wdisp"]] * C_KMS * np.log(10) * 1e-4
res = desi_resolution(arms_d); print(res.round(2).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
a = arms_d["B"]; j = 1500; off = np.arange(11) - 5
axs[0].step(off, a["res"][:, j], where="mid", label=f"DESI LSF at {a['wave'][j]:.0f} Å"); (A, mu, s_), _ = curve_fit(g, off, a["res"][:, j], p0=[0.5, 0, 1.5]); xx = np.linspace(-5, 5, 200)
axs[0].plot(xx, g(xx, A, mu, s_), color="tab:red", label=f"Gaussian σ = {abs(s_):.2f} px = {abs(s_) * 0.8:.2f} Å"); axs[0].set(xlabel="pixel offset", ylabel="LSF"); axs[0].legend()
for b, col in zip("BRZ", ("tab:blue", "tab:green", "tab:red")):
    sel = res.arm == b; axs[1].plot(res.wave[sel], res.R[sel], "o-", color=col, label=f"DESI {b}" + ("" if sp.source == "DESI" else " (reference)"))
axs[1].plot(lam_s, C_KMS / (2.3548 * sig_s), lw=0.7, color="0.4", label="SDSS (from wdisp)" + ("" if sp.source == "SDSS" else " (reference)"))
axs[1].set(xlabel="observed wavelength [Å]", ylabel="R = λ / FWHM", title="resolving power"); axs[1].legend(); plt.show()
print(f"instrumental σ at Hα ({6564.6 * (1 + z):.0f} Å): DESI ≈ {np.interp(6564.6 * (1 + z), res.wave, res.sigma_kms):.0f} km/s, SDSS ≈ {np.interp(6564.6 * (1 + z), lam_s, sig_s):.0f} km/s")"""),
        md(r"""So DESI resolves velocities down to ~25–50 km/s (σ), SDSS ~60–65 km/s. Galaxy emission lines here are
100–150 km/s wide: resolved, but the instrument still contributes measurably, and the fitting code
(notebook 2) must include it — pPXF takes this FWHM(λ) as input.

## 1.8 Two spectra of your system

If your galaxy has both a DESI and an SDSS spectrum, they are overlaid (same units; different fibre,
coverage and resolution). Otherwise the main galaxy and its companion, or, failing that, the reference
pair of the sample."""),
        code(r"""mine = inventory[inventory.system == SYSTEM]
def read_any(lab):
    sp_ = inventory.loc[lab]
    if sp_.source == "DESI":
        _, ar = desi_row(sp_.file, sp_.spec_id); w_ = np.concatenate([ar[b]["wave"] for b in "BRZ"]); f_ = np.concatenate([ar[b]["flux"] for b in "BRZ"]); i_ = np.concatenate([ar[b]["ivar"] for b in "BRZ"]); o_ = np.argsort(w_)
        return w_[o_], np.where(i_[o_] > 0, f_[o_], np.nan), f"{lab}: DESI 1.5\" fibre"
    d_ = sdss_file(sp_.file)[1]; c_ = {k.lower(): k for k in d_.dtype.names}
    return 10 ** d_[c_["loglam"]], np.where(d_[c_["ivar"]] > 0, d_[c_["flux"]], np.nan), f"{lab}: SDSS 3\" fibre"
mains = mine[mine.role == "main"]; comps = mine[mine.role == "comp"]
if mains.source.nunique() == 2: pick = list(mains.index)
elif len(comps): pick = [mains.index[0], comps.index[0]]
else: pick = ["IC3147B", "IC3147B_comp_sdss"]
fig, ax = plt.subplots(figsize=(12, 3.8))
for lab, col in zip(pick, ("tab:blue", "tab:orange")):
    w_, f_, l_ = read_any(lab); ax.plot(w_, f_, lw=0.5, color=col, label=l_)
ax.set(xlabel="observed wavelength [Å]", ylabel="flux [1e-17 erg/s/cm²/Å]"); ax.legend(); plt.show()"""),
        md(r"""## Exercises

1. How far is each fibre from the catalogue centre of your galaxy? Is that within the fibre radius?
   (`MEAN_FIBER_RA/DEC` and `PLUG_RA/DEC` against `ra`, `dec` of the sample table.)
2. From the fibre-to-total flux ratio and the image, estimate what fraction of the *bar* the fibre covers.
3. The velocity size of a pixel and the LSF σ in pixels: at which wavelength does your spectrum sample the
   LSF most coarsely? Would you trust a line width of 30 km/s there?
4. DESI: look up the coadd `MASK` bit definitions (desispec `specmask`); SDSS: the `and_mask` bits
   (`sdss.org/dr17/algorithms/bitmasks`). Which bits are set in your spectrum, and would you exclude those pixels?
5. Query SDSS or DESI (section 1.3) 2′ around your galaxy instead of 30″: what else has a spectrum there,
   and at which redshifts?""")]
    return C
