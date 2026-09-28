"""Build (and optionally execute) the student notebook series on nuclear spectroscopy of the
interbars sample:

  01_data_and_spectra.ipynb        finding the spectra, the image and the fibre, pixel scale,
                                   spectral resolution, WCS, what the fibre samples
  02_continuum_and_lines.ipynb     pPXF as the fitting engine, step by step: log grid, stellar and gas
                                   templates, the fit, weights -> fluxes, Balmer absorption, broad-line test,
                                   cross-checks (FastSpecFit, SDSS pipeline)
  03_bpt_classification.ipynb      the pPXF table, BPT / [S II] diagrams with pairs joined, dust, WHAN,
                                   comparison contours (SDSS pairs, barred) and stage-vs-population tests

Run:  source scripts/fastspecfit_env.sh && python notebooks/build_series.py [--execute]
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
HERE = Path(__file__).resolve().parent

SETUP = r"""import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):    # one BLAS thread: on Binder / Colab the pod has
    os.environ.setdefault(_v, "1")                                          # one core and multi-threaded BLAS is 30x slower
import os, io, json, subprocess
from pathlib import Path
import numpy as np, pandas as pd, fitsio, requests
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DESI = Path(os.environ.get("INTERBARS_DESI", ROOT / "output" / "desi_redux"))
for k, v in {"DESI_SPECTRO_REDUX": DESI, "SPECPROD": "iron", "DUST_DIR": DESI / "dust" / "v0_1",
             "FTEMPLATES_DIR": DESI / "ftemplates", "FPHOTO_DIR": DESI / "legacysurvey" / "dr9"}.items():
    os.environ.setdefault(k, str(v))
OUT = ROOT / "output" / "fastspecfit"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "output" / "notebooks"; CACHE.mkdir(parents=True, exist_ok=True)
C_KMS = 299792.458
COSMO = FlatLambdaCDM(H0=70, Om0=0.3)
plt.rcParams.update({"figure.dpi": 110})

def desi_targets():
    # every DESI DR1 spectrum within 5" of a sample galaxy or companion (from the survey search of the project)
    m = pd.read_csv(ROOT / "dataset/catalogs/specz_matches.csv", dtype={"obj_id": str})
    d = m[(m.survey == "DESI DR1") & (m.sep_arcsec <= 5) & m.usable].copy()
    for k in ("survey", "program", "healpix", "primary"):
        d[k] = d.extra.str.extract(rf"{k}=(\w+)")[0]
    d = d[d.primary == "t"].sort_values("sep_arcsec").drop_duplicates(["system", "role"])
    d["label"] = d.system + np.where(d.role == "comp", "_comp", "")
    d["coadd"] = [DESI / "iron" / "healpix" / r.survey / r.program / str(int(r.healpix) // 100) / r.healpix /
                  f"coadd-{r.survey}-{r.program}-{r.healpix}.fits" for r in d.itertuples()]
    d["redrock"] = [Path(str(c).replace("coadd-", "redrock-")) for c in d.coadd]
    return d.set_index("label")

def read_desi(coadd, targetid):
    # one spectrum out of a healpix coadd: FIBERMAP row -> the three arms (wave, flux, ivar, mask, resolution)
    fm = fitsio.read(coadd, "FIBERMAP")
    i = int(np.where(fm["TARGETID"] == targetid)[0][0])
    arms = {}
    with fitsio.FITS(coadd) as F:
        for b in "BRZ":
            arms[b] = dict(wave=F[f"{b}_WAVELENGTH"].read(), flux=F[f"{b}_FLUX"][i:i + 1, :][0],
                           ivar=F[f"{b}_IVAR"][i:i + 1, :][0], mask=F[f"{b}_MASK"][i:i + 1, :][0],
                           res=F[f"{b}_RESOLUTION"][i:i + 1, :, :][0])
    return fm[i], arms

def joined(arms):
    w = np.concatenate([a["wave"] for a in arms.values()]); f = np.concatenate([a["flux"] for a in arms.values()])
    iv = np.concatenate([a["ivar"] for a in arms.values()]); o = np.argsort(w)
    return w[o], f[o], iv[o]

def ls_cutout(ra, dec, size_arcsec, pixscale=0.262, layer="ls-dr10"):
    # Legacy Survey colour JPEG + a WCS built from the request (TAN projection, north up, east left)
    from PIL import Image
    from astropy.wcs import WCS
    npix = int(round(size_arcsec / pixscale))
    f = CACHE / f"ls_{ra:.5f}_{dec:.5f}_{size_arcsec:.0f}_{pixscale}.jpg"
    if not f.exists():
        r = requests.get("https://www.legacysurvey.org/viewer/jpeg-cutout", params=dict(ra=ra, dec=dec, layer=layer, pixscale=pixscale, size=npix), timeout=120)
        r.raise_for_status(); f.write_bytes(r.content)
    im = np.flipud(np.asarray(Image.open(f)))
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(im.shape[1] + 1) / 2, (im.shape[0] + 1) / 2]; w.wcs.cdelt = [-pixscale / 3600, pixscale / 3600]
    return im, w

targets = pd.read_csv(ROOT / "dataset/catalogs/interbars_targets.csv").set_index("system")
STAGE = {0: "pre-interaction", 1: "merger", 2: "post-merger"}
print("repo:", ROOT)"""


# ======================================================================================= 01
def nb01():
    C = [md(r"""# 1. Finding the spectra, and what a fibre spectrum is

**interbars**: barred galaxies in interacting pairs, at $z \approx 0.01$–$0.05$. This series asks whether
the gas in their **nuclei** is ionised by young stars or by an active nucleus. The data are optical
fibre spectra from public surveys (DESI, SDSS). Before measuring anything we need to know:

* which spectra exist for each galaxy, and *where exactly* on the galaxy the fibre was placed;
* how big the fibre is on the sky and in kiloparsecs at the galaxy's distance;
* what a spectrum file contains: flux units, uncertainties, the wavelength grid (pixel scale), the
  **spectral resolution** (how blurred a narrow line becomes), and bad-pixel masks.

All of this is in this notebook; notebook 2 measures the lines with pPXF, notebook 3 classifies the galaxies.
Environment: the `fastspecfit` conda env (`source scripts/fastspecfit_env.sh`)."""),
        code(SETUP),
        md(r"""## 1.1 The sample

One row per system: the main galaxy, its position (resolved through SIMBAD and checked on the
images), the redshift, the interaction stage from the visual classification (`class_VC`: 0 = pre-interaction,
1 = merger, 2 = post-merger), and a named companion where there is one."""),
        code(r"""cols = ["ra", "dec", "z_hand", "class_EB", "class_VC", "comp_name", "comp_z_hand", "note"]
print(targets[cols].round(5).to_string())"""),
        md(r"""## 1.2 Searching the archives for spectra

Spectroscopic surveys publish **catalogues** (one row per spectrum: position, redshift, quality) that
can be queried with SQL over the web. Two are relevant here: **SDSS DR17** (2000–2020, 3″ then 2″
fibres) and **DESI DR1** (2021–2022, 1.5″ fibres), both served by the NOIRLab Astro Data Lab. The query
below asks for every spectrum within 30″ of the galaxy. The full project search (9 surveys, 10′ around
every galaxy, `dataset/catalogs/specz_matches.csv`) used exactly these queries."""),
        code(r"""SYSTEM = "IC3147B"
t = targets.loc[SYSTEM]; ra, dec, z = float(t.ra), float(t.dec), float(t.z_hand)
R = 30 / 3600
def datalab(sql):
    r = requests.post("https://datalab.noirlab.edu/tap/sync", data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=sql), timeout=120)
    r.raise_for_status(); return pd.read_csv(io.StringIO(r.text))
box = lambda racol, deccol: f"{racol} BETWEEN {ra - R / np.cos(np.radians(dec))} AND {ra + R / np.cos(np.radians(dec))} AND {deccol} BETWEEN {dec - R} AND {dec + R}"
try:
    sdss = datalab(f"SELECT specobjid, plate, mjd, fiberid, ra, dec, z, zerr, zwarning, class, survey FROM sdss_dr17.specobj WHERE {box('ra', 'dec')}")
    desi = datalab(f"SELECT targetid, mean_fiber_ra AS ra, mean_fiber_dec AS dec, z, zerr, zwarn, spectype, survey, program, healpix, zcat_primary FROM desi_dr1.zpix WHERE {box('mean_fiber_ra', 'mean_fiber_dec')}")
    for name, d in (("SDSS DR17", sdss), ("DESI DR1", desi)):
        d["sep_arcsec"] = 3600 * np.hypot((d.ra - ra) * np.cos(np.radians(dec)), d.dec - dec)
        print(f"== {name}: {len(d)} spectra within 30\""); print(d.sort_values("sep_arcsec").round(5).to_string(index=False))
except Exception as e:
    print("offline? using the project's cached search instead:", e)
    m = pd.read_csv(ROOT / "dataset/catalogs/specz_matches.csv"); print(m[(m.system == SYSTEM) & (m.sep_arcsec < 30) & m.survey.isin(["SDSS DR17", "DESI DR1"])][["survey", "obj_id", "z", "zqual", "sep_arcsec"]].to_string(index=False))"""),
        md(r"""Things to notice: several rows can be the *same galaxy* (DESI observes a target in several programs;
`zcat_primary` marks the best one), rows tens of arcseconds away are *other* galaxies (or the companion),
and the quality flags (`zwarning`, `zwarn`) tell whether the pipeline trusted its redshift.

## 1.3 The image, the WCS, and where the fibre sits

A **WCS** (world coordinate system) maps image pixels to sky coordinates. For a small cutout a
tangent-plane projection is enough: a reference pixel `CRPIX` at `CRVAL` (RA, Dec) and a pixel scale
`CDELT` in degrees per pixel (negative in RA: east is left). We build one for a Legacy Survey DR10
colour image (0.262″ native pixels) and draw the fibres on it:

* DESI: 1.5″ diameter at `MEAN_FIBER_RA/DEC` (where the fibre actually was, averaged over exposures);
* SDSS: 3″ diameter at `PLUG_RA/DEC` (from the spectrum file header)."""),
        code(r"""tg = desi_targets(); r = tg.loc[SYSTEM]
fmrow, arms = read_desi(r.coadd, int(r.obj_id))
kpc_per_arcsec = 1 / COSMO.arcsec_per_kpc_proper(z).value
FOV = 60.0
im, w = ls_cutout(ra, dec, FOV)
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(projection=w))
ax.imshow(im, origin="lower")
def circle(ra_c, dec_c, diam_arcsec, **kw):
    x, y = w.world_to_pixel_values(ra_c, dec_c); ax.add_patch(plt.Circle((x, y), diam_arcsec / 2 / 0.262, fill=False, **kw))
circle(fmrow["MEAN_FIBER_RA"], fmrow["MEAN_FIBER_DEC"], 1.5, color="cyan", lw=1.5, label='DESI fibre 1.5"')
comp_spec = sorted((ROOT / "dataset/spectra" / SYSTEM).glob("*/spec-[0-9]*.fits"))
for f in comp_spec:
    h = fitsio.read_header(f, 0); circle(h["PLUG_RA"], h["PLUG_DEC"], 3.0, color="orange", lw=1.5, ls="--", label=f'SDSS fibre 3" ({f.parent.name})')
x0, y0 = w.world_to_pixel_values(ra, dec); ax.plot(x0, y0, "+", color="w", ms=12, mew=1, label="catalogue centre")
ax.plot([10, 10 + 5 / kpc_per_arcsec / 0.262], [10, 10], color="w", lw=2); ax.text(10, 16, "5 kpc", color="w")
ax.set(xlabel="RA", ylabel="Dec", title=f"{SYSTEM}  z={z:.4f}  {kpc_per_arcsec:.3f} kpc/\"  LS DR10 {FOV:.0f}\""); ax.legend(loc="upper right", fontsize=8); plt.show()
print(f"1 arcsec = {kpc_per_arcsec:.3f} kpc  ->  DESI fibre 1.5\" = {1.5 * kpc_per_arcsec:.2f} kpc, SDSS fibre 3\" = {3 * kpc_per_arcsec:.2f} kpc")
print(f"fibre / total r-band flux (Legacy Survey photometry in the fibermap): {fmrow['FIBERFLUX_R'] / fmrow['FLUX_R']:.1%}")"""),
        md(r"""The fibre sees only the nucleus: a few percent of the galaxy's light, the central kiloparsec.
Everything we conclude later is about *that region*, not the galaxy as a whole (the bar, the disc and
the tidal features are outside the fibre).

## 1.4 The spectrum file

A DESI healpix coadd holds hundreds of spectra; `FIBERMAP` identifies the rows, and each spectrograph
arm has its own `WAVELENGTH`, `FLUX` ($10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$), `IVAR` (inverse
variance, $1/\sigma^2$), `MASK` (bit flags of bad pixels) and `RESOLUTION` extensions."""),
        code(r"""print(f"targetid {int(r.obj_id)}: {fmrow['COADD_NUMEXP']} exposures, {fmrow['COADD_EXPTIME']:.0f} s total, {fmrow['COADD_NUMTILE']} tile(s)")
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
for b, c in zip("BRZ", ("tab:blue", "tab:green", "tab:red")):
    a = arms[b]; ok = a["ivar"] > 0; err = np.where(ok, 1 / np.sqrt(np.where(ok, a["ivar"], 1)), np.nan)
    axs[0].plot(a["wave"], np.where(ok, a["flux"], np.nan), lw=0.5, color=c, label=f"{b} arm {a['wave'][0]:.0f}–{a['wave'][-1]:.0f} Å")
    axs[0].fill_between(a["wave"], a["flux"] - err, a["flux"] + err, color=c, alpha=0.2, lw=0)
    axs[1].plot(a["wave"], err, lw=0.5, color=c); axs[1].plot(a["wave"][a["mask"] > 0], np.zeros((a["mask"] > 0).sum()), "|", color="k", ms=8)
axs[0].set(ylabel="flux [1e-17 erg/s/cm²/Å]", title=f"{SYSTEM}: the three DESI arms (shaded: ±1σ)"); axs[0].legend()
axs[1].set(xlabel="observed wavelength [Å]", ylabel="1σ error", ylim=(0, None)); axs[1].text(0.01, 0.8, "| = masked pixel", transform=axs[1].transAxes, fontsize=8); plt.show()"""),
        md(r"""## 1.5 Pixel scale

The wavelength grid is linear, 0.8 Å per pixel. In velocity units that is $c\,\Delta\lambda/\lambda$:
67 km/s at the blue end, 24 km/s at the red end. A galaxy line with $\sigma \approx 100$ km/s is
therefore sampled by only a few pixels in the blue — which is why we need the resolution, not just the
pixel size."""),
        code(r"""wj, fj, ivj = joined(arms)
dlam = np.diff(wj); print(f"pixel step: {np.median(dlam):.2f} Å (min {dlam[dlam > 0].min():.2f}, max {dlam.max():.2f})")
fig, ax = plt.subplots(figsize=(8, 3)); ax.plot(wj[1:], C_KMS * dlam / wj[1:], lw=0.8); ax.set(xlabel="observed wavelength [Å]", ylabel="pixel [km/s]", ylim=(0, 80), title="velocity size of one pixel"); plt.show()"""),
        md(r"""## 1.6 Spectral resolution

A perfectly narrow line is recorded as a blurred profile, the **line-spread function** (LSF). DESI stores
it explicitly: the `RESOLUTION` extension is a band matrix with 11 diagonals per pixel — the LSF centred
on that pixel, in units of pixels. We fit a Gaussian to it at several wavelengths and convert the width
to the resolving power $R = \lambda / {\rm FWHM}$ and to $\sigma_{\rm inst}$ in km/s. SDSS stores instead a
per-pixel `wdisp` (Gaussian σ in pixels of its log-wavelength grid, 69 km/s per pixel)."""),
        code(r"""from scipy.optimize import curve_fit
g = lambda x, A, mu, s: A * np.exp(-0.5 * ((x - mu) / s) ** 2)
rows = []
for b in "BRZ":
    a = arms[b]; nd = a["res"].shape[0]; off = np.arange(nd) - nd // 2
    for j in np.linspace(50, len(a["wave"]) - 50, 6).astype(int):
        prof = a["res"][:, j]
        if prof.sum() < 0.5: continue
        (A, mu, s), _ = curve_fit(g, off, prof, p0=[prof.max(), 0, 1.5])
        dl = a["wave"][1] - a["wave"][0]; lam = a["wave"][j]
        rows.append(dict(arm=b, wave=lam, sigma_pix=abs(s), sigma_A=abs(s) * dl, R=lam / (2.3548 * abs(s) * dl), sigma_kms=C_KMS * abs(s) * dl / lam))
res = pd.DataFrame(rows); print(res.round(2).to_string(index=False))
# SDSS, for comparison: a lite file of the companion
f = comp_spec[0]; d = fitsio.read(f, "COADD"); lam = 10 ** d["loglam"]; sig_kms_sdss = d["wdisp"] * C_KMS * np.log(10) * 1e-4
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
a = arms["B"]; j = 1500; off = np.arange(11) - 5
axs[0].step(off, a["res"][:, j], where="mid", label=f"DESI LSF at {a['wave'][j]:.0f} Å"); (A, mu, s), _ = curve_fit(g, off, a["res"][:, j], p0=[0.5, 0, 1.5]); xx = np.linspace(-5, 5, 200)
axs[0].plot(xx, g(xx, A, mu, s), color="tab:red", label=f"Gaussian σ = {abs(s):.2f} px = {abs(s) * 0.8:.2f} Å"); axs[0].set(xlabel="pixel offset", ylabel="LSF"); axs[0].legend()
for b, c in zip("BRZ", ("tab:blue", "tab:green", "tab:red")):
    sel = res.arm == b; axs[1].plot(res.wave[sel], res.R[sel], "o-", color=c, label=f"DESI {b}")
axs[1].plot(lam, lam / (2.3548 * d["wdisp"] * lam * np.log(10) * 1e-4), lw=0.7, color="0.4", label="SDSS (from wdisp)")
axs[1].set(xlabel="observed wavelength [Å]", ylabel="R = λ / FWHM", title="resolving power"); axs[1].legend(); plt.show()
print(f"instrumental σ at Hα ({6564.6 * (1 + z):.0f} Å): DESI ≈ {np.interp(6564.6 * (1 + z), res.wave, res.sigma_kms):.0f} km/s, SDSS ≈ {np.interp(6564.6 * (1 + z), lam, sig_kms_sdss):.0f} km/s")"""),
        md(r"""So DESI resolves velocities down to ~35–50 km/s (σ), SDSS ~65 km/s. Galaxy emission lines here are
100–150 km/s wide: resolved, but the instrument still contributes measurably, and the fitting code
(notebook 2) must include it — pPXF takes this FWHM(λ) as input.

## 1.7 Two galaxies, two instruments

The companion of IC 3147B, IC 3147, has an SDSS spectrum. Same physical units, different wavelength
coverage, resolution and fibre size."""),
        code(r"""fig, ax = plt.subplots(figsize=(12, 3.8))
ax.plot(wj, np.where(ivj > 0, fj, np.nan), lw=0.5, color="tab:blue", label=f"{SYSTEM}: DESI, 1.5\" fibre, {fmrow['COADD_EXPTIME']:.0f} s")
h = fitsio.read_header(f, 0); ax.plot(lam, d["flux"], lw=0.5, color="tab:orange", label=f"{f.parent.name}: SDSS plate {h['PLATEID']} fibre {h['FIBERID']}, 3\" fibre, {h['EXPTIME']:.0f} s")
ax.set(xlabel="observed wavelength [Å]", ylabel="flux [1e-17 erg/s/cm²/Å]"); ax.legend(); plt.show()"""),
        md(r"""## Exercises

1. Repeat 1.2–1.4 for another system of the table (try one with a companion in DESI:
   `2dFGRSTGN321Z156`). How far is the fibre from the catalogue centre? Is that within the fibre radius?
2. From the fibre-to-total flux ratio and the image, estimate what fraction of the *bar* the fibre covers.
3. The velocity size of a pixel and the LSF σ in pixels: at which wavelength does DESI sample the LSF
   most coarsely? Would you trust a line width of 30 km/s there?
4. Look up the DESI coadd `MASK` bit definitions (desispec `specmask`). Which bits are set in this spectrum?
5. Query SDSS for a galaxy of the table that has *no* DESI spectrum (e.g. `Mrk1302`). Which fibre size?""")]
    return C


# ======================================================================================= 02 / 03: pPXF versions
from _nb_ppxf import nb01, nb02, nb03  # noqa: E402   (nb01 here overrides the older one above)


def build(execute):
    for fname, cells in (("01_data_and_spectra.ipynb", nb01()), ("02_continuum_and_lines.ipynb", nb02()),
                         ("03_bpt_classification.ipynb", nb03())):
        nb = nbf.v4.new_notebook(); nb["cells"] = cells
        nb.metadata["kernelspec"] = dict(name="python3", display_name="Python (fastspecfit)", language="python")
        out = HERE / fname; nbf.write(nb, out); print("wrote", out)
        if execute:
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--ExecutePreprocessor.timeout=1200",
                            str(out), "--output", out.name], check=True)


if __name__ == "__main__":
    build("--execute" in sys.argv)
