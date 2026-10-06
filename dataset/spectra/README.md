# dataset/spectra/

1D optical spectra of the sample galaxies and their companions, copied from the SSD archive
(`/media/eborsato/WorkSSD/archive/INTERBARS/sample/spectra`, downloaded 2026-05 by `dataset/catalogs/archive/redshift_check_spectra.py`) on 2026-09-28
by `scripts/ingest_archive_spectra.py`. Layout `<system>/<galaxy>/spec-*.fits`; `.fits` gitignored,
`checksums.csv` (sha256, survey, id, z) tracked.

- `spec-PPPP-MMMMM-FFFF.fits`: SDSS DR16 "lite" coadds (`COADD`: loglam/flux/ivar, 1e-17 erg/s/cm²/Å;
  `SPECOBJ`: pipeline z, class), 3" fibres (SDSS) / 2" (BOSS), R ~ 1500–2500, 3600–10400 Å.
- `spec-desi-<targetid>.fits`: single-target extraction from the DESI DR1 healpix coadd (`B/R/Z_*` arms,
  same units), 1.5" fibres, R ~ 2000–5000, 3600–9800 Å. Redshift not in the file: see
  `dataset/catalogs/specz_matches.csv` (DESI DR1 row of the same targetid).

Skipped from the archive: 2MASXJ09551395+1417576/PGC028602 (unreadable: spec-5325-55980-0350.fits), IC3378/IC3379 (unreadable: spec-5852-56034-0264.fits), IC775/PGC039595, NGC 5614/IC2810B, NGC 5614/NGC 5614 (wrong galaxies, see docs/SPECZ_SEARCH_PLAN.md §2).
Plots: `output/specz_gallery/spectra/<system>__<galaxy>.png` (all), examples in `results/figures/spectra/`.

35 spectra: 24 SDSS, 11 DESI.
