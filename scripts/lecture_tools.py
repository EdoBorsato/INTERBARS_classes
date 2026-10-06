"""Helpers for the two lecture notebooks (notebooks/L1_spectra_and_ppxf.ipynb, notebooks/L2_from_archive_to_bpt.ipynb).

Only the plumbing lives here (reading, line lists, the BPT frame, the pPXF input preparation); everything the
students are meant to *see* — the LOSVD convolution, the mock-galaxy model, the pPXF call — is written out in
the notebooks themselves.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ppxf.ppxf_util as util
import ppxf.sps_util as lib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ppxf as RP  # noqa: E402

C_KMS = 299792.458
SPS_FILE = ROOT / "output/ppxf/sps_models/spectra_emiles_9.0.npz"
# LEDA3091410 and 2MASXJ08232861+0924041 are the same pair entered twice in the target list (each the other's
# companion, same two SDSS fibres): for the course it is one system, 2MASXJ08232861+0924041 + its companion LEDA3091410.
COURSE_EXCLUDE = {"LEDA3091410"}

# rest-frame AIR wavelengths [Å]
EMISSION = [("[O II]", 3727.4), ("[Ne III]", 3868.8), ("Hδ", 4101.7), ("Hγ", 4340.5), ("Hβ", 4861.3), ("[O III]", 4958.9),
            ("[O III]", 5006.8), ("He I", 5875.6), ("[O I]", 6300.3), ("[N II]", 6548.1), ("Hα", 6562.8), ("[N II]", 6583.5),
            ("[S II]", 6716.4), ("[S II]", 6730.8), ("[S III]", 9068.6)]
ABSORPTION = [("Ca II K", 3933.7), ("Ca II H", 3968.5), ("Hδ", 4101.7), ("G band", 4304.4), ("Hγ", 4340.5), ("Hβ", 4861.3),
              ("Mg b", 5175.4), ("Fe", 5270.0), ("Na D", 5892.9), ("Hα", 6562.8), ("TiO", 7100.0), ("Ca II triplet", 8542.1)]


def annotate(ax, kinds=("emission", "absorption"), y_em=0.97, y_ab=0.03, z=0.0, fontsize=7, skip=()):
    """Mark the main features inside the axis' x range (rest frame unless z is given)."""
    lo, hi = ax.get_xlim()
    if "emission" in kinds:
        last = -1e9
        for name, w in EMISSION:
            x = w * (1 + z)
            if lo < x < hi and name not in skip:
                ax.axvline(x, color="tab:red", lw=0.6, alpha=0.35, zorder=0)
                if x - last > (hi - lo) * 0.012:
                    ax.text(x, y_em, name, transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="top", fontsize=fontsize, color="tab:red")
                last = x
    if "absorption" in kinds:
        for name, w in ABSORPTION:
            x = w * (1 + z)
            if lo < x < hi and name not in skip:
                ax.axvline(x, color="tab:blue", lw=0.6, ls=":", alpha=0.5, zorder=0)
                ax.text(x, y_ab, name, transform=ax.get_xaxis_transform(), rotation=90, ha="center", va="bottom", fontsize=fontsize, color="tab:blue")


# ------------------------------------------------------------------------------------------------ spectra
def inventory():
    t = RP.targets()
    return t[~t.system.isin(COURSE_EXCLUDE)].set_index("label")


def read(label, inv=None, download=True):
    """(lam_vac, flux, ivar, fwhm_A, mask, z, source) of one spectrum of the sample; SDSS files are fetched from the SAS if missing."""
    inv = inventory() if inv is None else inv
    r = inv.loc[label]
    if r.source == "DESI":
        out = RP.read_desi(r.file, r.spec_id)
    else:
        if download:
            RP.fetch_lite(r.file, getattr(r, "run2d", ""))
        out = RP.read_sdss(r.file)
    return (*out, float(r.z_in), r.source)


def prepare(lam_vac, flux, ivar, fwhm, mask, z, sps_file=SPS_FILE):
    """The pPXF inputs, exactly as in scripts/run_ppxf.py: air wavelengths, log grid, noise, stellar + gas templates."""
    good = (ivar > 0) & ~mask & np.isfinite(flux)
    lam_air = util.vac_to_air(lam_vac)
    flux_f = np.where(good, flux, np.interp(lam_air, lam_air[good], flux[good]))
    var = np.where(good, 1 / np.where(good, ivar, 1), 0.0)
    norm = np.median(flux_f[good])
    velscale = util.log_rebin(lam_air, flux_f)[2]
    galaxy, ln_lam, _ = util.log_rebin(lam_air, flux_f / norm, velscale=velscale)
    var_log = util.log_rebin(lam_air, var / norm ** 2, velscale=velscale)[0]
    bad_log = util.log_rebin(lam_air, (~good).astype(float), velscale=velscale)[0] > 0.3
    lam_gal = np.exp(ln_lam)
    npix_per = np.clip(lam_gal * velscale / C_KMS / np.median(np.diff(lam_air)), 1, None)
    noise = np.sqrt(np.clip(var_log / npix_per, 1e-8, None)); noise[bad_log] = 1e3
    fwhm_gal = {"lam": lam_gal, "fwhm": np.interp(lam_gal, lam_air, fwhm)}
    sps = lib.sps_lib(sps_file, velscale, fwhm_gal, norm_range=[5070, 5950])
    reg_dim = sps.templates.shape[1:]; stars = sps.templates.reshape(sps.templates.shape[0], -1)
    gas, gas_names, line_wave = util.emission_lines(sps.ln_lam_temp, np.array([lam_gal[0], lam_gal[-1]]) / (1 + z), fwhm_gal,
                                                    tie_balmer=False, limit_doublets=False)
    forb = np.array(["[" in n for n in gas_names])
    vel = C_KMS * np.log(1 + z)
    goodpixels = np.where(~bad_log & ~((lam_gal > 5573) & (lam_gal < 5583)) & ~((lam_gal > 7590) & (lam_gal < 7700)))[0]
    return dict(galaxy=galaxy, noise=noise, lam_gal=lam_gal, velscale=velscale, norm=norm, z=z, sps=sps, reg_dim=reg_dim, fwhm_gal=fwhm_gal,
                n_temps=stars.shape[1], templates=np.column_stack([stars, gas]), gas_names=gas_names, line_wave=line_wave,
                component=[0] * stars.shape[1] + [2 if f else 1 for f in forb], goodpixels=goodpixels,
                start=[[vel, 150.0], [vel, 100.0], [vel, 100.0]], bounds=[[[vel - 1500, vel + 1500], [10, 500]]] * 3)


def line_fluxes(pp, P):
    """pPXF gas weights -> line fluxes [1e-17 erg/s/cm²] with formal errors (strong line of a fixed doublet = template / 1.33)."""
    rows = []
    for name, fl, er, lw in zip(P["gas_names"], pp.gas_flux, pp.gas_flux_error, P["line_wave"]):
        if name in RP.GAS:
            key, fac = RP.GAS[name]; dlam = lw * (1 + P["z"]) * P["velscale"] / C_KMS
            rows.append(dict(line=name, key=key, flux=fl * dlam * P["norm"] / fac, err=er * dlam * P["norm"] / fac))
    L = pd.DataFrame(rows).set_index("key"); L["snr"] = L.flux / L.err
    return L


# ------------------------------------------------------------------------------------------------ BPT
def kauffmann03(x): return 0.61 / (x - 0.05) + 1.3
def kewley01(x): return 0.61 / (x - 0.47) + 1.19
def schawinski07(x): return 1.05 * x + 0.45


def bpt_frame(ax, labels=True, xlim=(-1.6, 0.7), ylim=(-1.3, 1.3)):
    """Empty [N II] BPT with the Kauffmann (2003), Kewley (2001) and Schawinski (2007) lines."""
    x1 = np.linspace(-2, 0.04, 200); x2 = np.linspace(-2, 0.46, 200); x3 = np.linspace(-0.18, 1, 50)
    ax.plot(x1, kauffmann03(x1), "k--", lw=1); ax.plot(x2, kewley01(x2), "k-", lw=1)
    x3 = x3[schawinski07(x3) > kewley01(np.minimum(x3, 0.46))]; ax.plot(x3, schawinski07(x3), "k:", lw=1)
    if labels:
        for (x, y, s) in [(-1.25, -0.9, "star-forming"), (-0.15, -0.95, "composite"), (-0.65, 1.05, "Seyfert"), (0.3, -0.35, "LINER")]:
            ax.text(x, y, s, fontsize=9, color="0.35", style="italic")
    ax.set(xlim=xlim, ylim=ylim, xlabel=r"log [N II] 6583 / H$\alpha$", ylabel=r"log [O III] 5007 / H$\beta$")


bpt_class = RP.bpt_class


# ------------------------------------------------------------------------------------------------ mock galaxies (lecture 1, part C)
ZOOMS = [(3800, 4400, "4000 Å break, Ca H & K, Hδ"), (4820, 5030, "Hβ, [O III]"), (6500, 6760, "Hα, [N II], [S II]")]


def show_mock(m, title="", data=None, xlim=(3650, 9400)):
    """A mock galaxy from make_galaxy(): the whole spectrum, three zooms and, if it has gas, its place on the BPT.
    `data` = optional (rest wavelength, flux) of a real spectrum drawn underneath."""
    import matplotlib.pyplot as plt
    has_gas = np.isfinite(m.get("bpt", (np.nan, np.nan))[0])
    fig = plt.figure(figsize=(14, 7.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], width_ratios=[1.4, 1, 1.2, 1])
    ax = fig.add_subplot(gs[0, :]); s = (m["lam"] > xlim[0]) & (m["lam"] < xlim[1])
    if data is not None:
        sd = (data[0] > xlim[0]) & (data[0] < xlim[1]); ax.plot(data[0][sd], data[1][sd], color="0.6", lw=0.6, label="real spectrum")
    ax.plot(m["lam"][s], m["flux"][s], color="k", lw=0.5, label="your galaxy" + (f" (S/N = {m['snr']:g})" if m.get("snr") else ""))
    ax.plot(m["lam"][s], m["stars"][s], color="tab:orange", lw=0.9, label="its stars")
    if has_gas: ax.plot(m["lam"][s], m["gas"][s], color="tab:red", lw=0.7, alpha=0.8, label="its gas")
    top = np.nanpercentile(m["flux"][s], 99.9); ax.set(xlim=xlim, ylim=(0, max(1.6, min(top * 1.1, 8))), ylabel="flux (1 = mean at 5070–5950 Å)")
    annotate(ax, fontsize=7); ax.legend(fontsize=8, loc="upper right", ncol=2); ax.set_title(title)
    for k, (lo, hi, ttl) in enumerate(ZOOMS):
        a = fig.add_subplot(gs[1, k]); s = (m["lam"] > lo) & (m["lam"] < hi)
        if data is not None:
            sd = (data[0] > lo) & (data[0] < hi); a.plot(data[0][sd], data[1][sd], color="0.6", lw=0.8, drawstyle="steps-mid")
        a.plot(m["lam"][s], m["flux"][s], color="k", lw=0.8, drawstyle="steps-mid"); a.plot(m["lam"][s], m["stars"][s], color="tab:orange", lw=1.1)
        a.set_xlim(lo, hi); a.set_title(ttl, fontsize=9); a.set_xlabel("rest wavelength [Å]"); annotate(a, fontsize=6)
    a = fig.add_subplot(gs[1, 3]); bpt_frame(a, labels=True)
    if has_gas:
        a.plot(*m["bpt"], "*", ms=18, color="gold", mec="k", zorder=5)
    else:
        a.text(-1.5, 1.05, "no gas: not on the BPT", fontsize=9)
    a.set_title("where its gas sits on the BPT", fontsize=9)
    fig.tight_layout(); plt.show()


def light_map(ax, w, ages, metals, title=""):
    """Light fractions on the age-metallicity grid."""
    im = ax.imshow(w.T, origin="lower", aspect="auto", cmap="magma_r", vmin=0, extent=[-0.5, len(ages) - 0.5, -0.5, len(metals) - 0.5])
    ax.set_xticks(range(0, len(ages), 3)); ax.set_xticklabels([f"{a:.2g}" for a in ages[::3]]); ax.set_yticks(range(len(metals))); ax.set_yticklabels([f"{m:+.1f}" for m in metals])
    ax.set(xlabel="age [Gyr]", ylabel="[M/H]", title=title); return im
