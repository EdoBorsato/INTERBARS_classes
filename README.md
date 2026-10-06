# interbars — nuclear spectroscopy of interacting barred galaxies (student bundle)

Two lectures:

* **Lecture 1** — `notebooks/L1_spectra_and_ppxf.ipynb`: the ingredients of pPXF (stellar and gas templates) and a
  first fit of a real nucleus; appendices on the features of a spectrum, kinematics and dust.
* **Lecture 2** — `notebooks/L2_from_archive_to_bpt.ipynb`: build mock galaxies by hand, then download the spectra of
  your galaxies, fit them with pPXF,
  and place them on the BPT diagram of barred and of interacting galaxies. If your system is MCG-01-09-041 (a Seyfert 1),
  continue in `notebooks/L2b_broad_lines.ipynb`: broad lines and the black-hole mass.

Optional background, not needed for the lectures: the longer notebook series in `reference/`
(`01_data_and_spectra` → `02_continuum_and_lines` → `03_bpt_classification`, and a line fit by hand).

**Slides**: `slides/lecture1_slides.pdf` (lecture 1: active nuclei, the sample, how pPXF works).

**The sample gallery**: `gallery.html` (download it and open it in a browser) shows every system of the sample, its
companion and the spectrum of its nucleus. Use it to choose your galaxy for lecture 2.

## Run in the browser, nothing to install

* **Binder** (free, no account; a session lasts while you work and is discarded after ~10 min of inactivity, so
  download your notebook when done): [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/EdoBorsato/INTERBARS_classes/HEAD?urlpath=lab/tree/notebooks/L1_spectra_and_ppxf.ipynb)
* **Google Colab** (Google account; your copy is saved in your Drive): open a notebook and run its first cell:
  [lecture 1](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/L1_spectra_and_ppxf.ipynb) · [lecture 2](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/L2_from_archive_to_bpt.ipynb) · [lecture 2b](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/notebooks/L2b_broad_lines.ipynb);
  background series: [notebook 1](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/reference/01_data_and_spectra.ipynb) · [notebook 2](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/reference/02_continuum_and_lines.ipynb) · [notebook 3](https://colab.research.google.com/github/EdoBorsato/INTERBARS_classes/blob/HEAD/reference/03_bpt_classification.ipynb).

## Or on your own computer (once)

    conda env create -f environment-students.yml      # or:  python -m venv venv && venv/bin/pip install -r requirements-students.txt
    conda activate interbars-students
    jupyter lab notebooks/

Python ≥ 3.10; everything is pip-installable (pPXF, fitsio, astropy, scipy, pandas, matplotlib, Pillow, requests, jupyter).
No DESI software is needed: the DESI spectra are shipped as coadd files trimmed to the sample, the SDSS spectra as
the original lite files, the pPXF E-MILES templates and the two SDSS comparison catalogues are included.
The lectures run without internet once the folder is open: every spectrum, the galaxy images and the comparison
catalogues are included. Lecture 2 shows how to query the archives and download a spectrum live; if the network is
slow, set `ONLINE = False` in its first cell (it also switches itself off after the first failed request).
In the background series, the archive queries (notebook 1) and the DESI catalogue cross-check (notebook 2, 2.9) need
internet and print a message when offline.

Bundle built 2026-10-06 from the interbars repository.
