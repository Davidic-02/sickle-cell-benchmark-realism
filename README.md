# Benchmark curation and crop geometry in sickle cell morphology classification

Code and results for the paper *"Benchmark curation and crop geometry affect reported accuracy of sickle cell morphology classification"* (David Olusegun Adekoya, Federal University of Technology Akure).

Published accuracies on the **erythrocytesIDB** benchmark exceed 96%, but they are measured on cell images that the dataset curators selected individually and balanced across classes. This study compares those curated cells with cells extracted from annotated full-field smears **in the same dataset**, holding preprocessing, features, models and evaluation identical.

## Main findings

| | Balanced accuracy |
|---|---|
| Curated benchmark cells (IDB1) | **91.3%** (95% interval 89.8–93.2) |
| Cells extracted from smears (IDB2) | **80.1%** (76.0–84.3) |
| Difference | **11.2 points** (6.4–15.5) |
| Same comparison, ResNet18 | **14.2 points** (93.2% vs 79.0%) |

The difference survives controls for class balance, sample size, extraction quality, image resolution, blur and contrast.

Two further results:

- **Crop geometry matters on its own.** Resizing a non-square cell crop to a square input distorts cell shape and costs **7.1 points** (3.1–9.7) on identical cells, cutting elongated-cell recall from 85.2% to 68.5%.
- **Two negative results.** Grouping cross-validation folds by source smear changes nothing (−0.4 to −0.5 points), and surrounding smear context does not help — wide context costs up to 10.0 points.

## Reproducing

The dataset is **not** included (see Licence). Download it first:

```bash
# erythrocytesIDB v2 — Zenodo DOI 10.5281/zenodo.18299474
curl -L -o eidb.zip "https://zenodo.org/api/records/18299474/files/Version%202,%20erythrocytesIDB%202017.zip/content"
md5 eidb.zip    # expect 9d480fe1763f6ddb2b8bb4983cdb51e7
mkdir -p data/raw && unzip -q eidb.zip -x "__MACOSX/*" -d data/raw
mv "data/raw/Version 2, erythrocytesIDB" data/raw/eidb
```

Then create the environment and run the analyses in order:

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy scipy pandas scikit-learn scikit-image opencv-python-headless matplotlib

./.venv/bin/python src/extract_cells.py        # cells from masks, with parent smear id
./.venv/bin/python src/run_splits.py           # cell-level vs smear-grouped splits
./.venv/bin/python src/curated_vs_natural.py   # curated vs smear-extracted arms
./.venv/bin/python src/extract_reframed.py     # square frames matched to curated framing
./.venv/bin/python src/gap_intervals.py        # headline gap with intervals
./.venv/bin/python src/aspect_test.py          # crop geometry
./.venv/bin/python src/context_pilot.py        # surrounding context
./.venv/bin/python src/make_figures.py         # figures, plotted from results/
```

The ResNet18 arm needs Python 3.10+ (current PyTorch drops 3.9):

```bash
python3.11 -m venv .venv311
./.venv311/bin/pip install numpy scikit-learn scikit-image opencv-python-headless torch torchvision
./.venv311/bin/python src/cnn_gap.py
```

## Layout

| Path | Contents |
|---|---|
| `src/` | Extraction, analysis and figure scripts |
| `results/` | Raw outputs every number and figure is drawn from |
| `paper/` | Manuscript (Markdown and Word), figures, submission notes |
| `RESULTS.md` | All 15 experiments in the order they were run, including the negative ones |

## Licence

Code in this repository is MIT licensed.

The **erythrocytesIDB dataset is not redistributed here**. It is licensed CC BY-NC-ND 4.0, which does not permit distribution of derived works, so the extracted cell crops are excluded by `.gitignore`. Download the dataset from Zenodo under its own terms.

## Citing

Cite the dataset and its describing publication alongside this work:

- M. Gonzalez-Hidalgo, F. A. Guerrero-Pena, S. Herold-Garcia, A. Jaume-i-Capo and P. D. Marrero-Fernandez, "Red blood cell cluster separation from digital images for use in sickle cell disease," *IEEE Journal of Biomedical and Health Informatics*, vol. 19, no. 4, pp. 1514–1525, 2015, doi: 10.1109/JBHI.2014.2356402.
