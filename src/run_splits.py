"""Measure how much a random cell-level split inflates performance.

Same data, same features, same model, same number of folds -- the only thing
that changes is whether cells from one smear may appear in both train and test.

  cell-level  : StratifiedKFold        (what the field does)
  smear-level : StratifiedGroupKFold   (what it should do)
"""
import csv
import pathlib

import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZE = (64, 64)
N_SPLITS = 5
SEEDS = range(10)


def load():
    cache = ROOT / "data" / "features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        print(f"loaded cached features {z['X'].shape}", flush=True)
        return z["X"], z["y"], z["g"]

    X, y, g = [], [], []
    with open(ROOT / "data" / "cells_manifest.csv") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        if i % 200 == 0:
            print(f"  hog {i}/{len(rows)}", flush=True)
        if True:
            img = imread(ROOT / "data" / r["path"])
            if img.ndim == 3:
                img = rgb2gray(img)
            img = resize(img, SIZE, anti_aliasing=True)
            X.append(hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), block_norm="L2-Hys"))
            y.append(r["label"])
            g.append(r["smear_id"])
    X, y, g = np.asarray(X), np.asarray(y), np.asarray(g)
    np.savez_compressed(cache, X=X, y=y, g=g)
    print(f"cached features {X.shape}", flush=True)
    return X, y, g


def evaluate(X, y, groups, grouped, seed):
    if grouped:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        folds = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        folds = cv.split(X, y)

    accs, f1s = [], []
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=3000))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        accs.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def summarise(name, vals):
    a = np.array([v[0] for v in vals]) * 100
    f = np.array([v[1] for v in vals]) * 100
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"{name:<14} acc {a.mean():5.2f}%  (95% CI {lo:5.2f}-{hi:5.2f})   macro-F1 {f.mean():5.2f}%")
    return a


def main():
    X, y, groups = load()
    print(f"cells {len(y)}  smears {len(set(groups))}  features {X.shape[1]}")
    for cls in sorted(set(y)):
        print(f"  {cls:<10} {(y == cls).sum()}")
    print()

    cell, smear = [], []
    for s in SEEDS:
        cell.append(evaluate(X, y, groups, False, s))
        smear.append(evaluate(X, y, groups, True, s))
        print(f"  seed {s}: cell {cell[-1][0]*100:.2f}%  smear {smear[-1][0]*100:.2f}%", flush=True)
    print()

    a_cell = summarise("cell-level", cell)
    a_smear = summarise("smear-level", smear)
    print(f"\ninflation from ignoring smear identity: {a_cell.mean() - a_smear.mean():+.2f} points")


if __name__ == "__main__":
    main()
