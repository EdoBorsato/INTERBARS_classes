# interbars — nuclear spectroscopy of interacting barred galaxies (student bundle)

Three notebooks: `notebooks/01_data_and_spectra.ipynb` → `02_continuum_and_lines.ipynb` → `03_bpt_classification.ipynb`
(read `notebooks/README.md`). Each student chooses a system once, in notebook 1, section 1.2.

## Run in the browser, nothing to install

* **Binder** (free, no account; a session lasts while you work and is discarded after ~10 min of inactivity, so
  download your notebook when done): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdoBorsato/INTERBARS_classes/HEAD?urlpath=lab/tree/notebooks/01_data_and_spectra.ipynb)
* **Google Colab** (Google account; your copy is saved in your Drive): open a notebook and run its first cell:
  [notebook 1](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/01_data_and_spectra.ipynb) · [notebook 2](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/02_continuum_and_lines.ipynb) · [notebook 3](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/03_bpt_classification.ipynb).
  Notebooks 2 and 3 read the system you chose in notebook 1 from `notebooks/my_systems.json`; on Colab each
  notebook has its own fresh copy of the bundle, so set `MY_SYSTEMS` in every notebook (the value in the file
  is only the default).

## Or on your own computer (once)

    conda env create -f environment-students.yml      # or:  python -m venv venv && venv/bin/pip install -r requirements-students.txt
    conda activate interbars-students
    jupyter lab notebooks/

Python ≥ 3.10; everything is pip-installable (pPXF, fitsio, astropy, scipy, pandas, matplotlib, Pillow, requests, jupyter).
No DESI software is needed: the DESI spectra are shipped as coadd files trimmed to the sample, the SDSS spectra as
the original lite files, the pPXF E-MILES templates and the two SDSS comparison catalogues are included.
Internet is only needed for the archive queries of notebook 1 (section 1.3), the Legacy Survey image cutouts
(cached under `output/notebooks/`), and the DESI catalogue cross-check in notebook 2 (section 2.9): those cells
print a message and continue when offline.

Bundle built 2026-09-28 from the interbars repository; source of the notebooks: `notebooks/build_series.py`.
