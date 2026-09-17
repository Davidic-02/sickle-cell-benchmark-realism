"""Curated benchmark cells vs cells extracted naturally from smears.

Same preprocessing, same features, same models, same protocol.
Three arms so curation is not confounded with class balance or sample size:
  A curated    : IDB1's hand-picked 625 cells (balanced by construction)
  B natural    : our 2263 mask-extracted cells (natural class mix)
  C natural-bal: B subsampled to balanced + size-matched to A
"""
import csv, pathlib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIZE = (64, 64)
CLASSES = ("circular", "elongated", "other")


def feats(path):
    img = imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    img = resize(img, SIZE, anti_aliasing=True)
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def load_curated():
    cache = ROOT / "data" / "features_curated.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True); return z["X"], z["y"]
    base = ROOT / "data" / "raw" / "eidb" / "erythrocytesIDB1" / "individual cells"
    X, y = [], []
    for c in CLASSES:
        for p in sorted((base / c).glob("*.jpg")):
            X.append(feats(p)); y.append(c)
    X, y = np.asarray(X), np.asarray(y)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def mk(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0)
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", class_weight="balanced"))


def ev(X, y, model, seeds=range(5)):
    acc, bal, f1 = [], [], []
    for s in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(X, y):
            m = mk(model); m.fit(X[tr], y[tr]); p = m.predict(X[te])
            acc.append(accuracy_score(y[te], p))
            bal.append(balanced_accuracy_score(y[te], p))
            f1.append(f1_score(y[te], p, average="macro"))
    return np.mean(acc) * 100, np.mean(bal) * 100, np.mean(f1) * 100


def main():
    Xc, yc = load_curated()
    z = np.load(ROOT / "data" / "features.npz", allow_pickle=True)
    Xn, yn = z["X"], z["y"]

    print(f"A curated    : {len(yc)} cells  " + str({c: int((yc == c).sum()) for c in CLASSES}))
    print(f"B natural    : {len(yn)} cells  " + str({c: int((yn == c).sum()) for c in CLASSES}))

    rng = np.random.default_rng(0)
    per = min((yn == c).sum() for c in CLASSES)
    idx = np.concatenate([rng.choice(np.where(yn == c)[0], per, replace=False) for c in CLASSES])
    Xb, yb = Xn[idx], yn[idx]
    print(f"C natural-bal: {len(yb)} cells  " + str({c: int((yb == c).sum()) for c in CLASSES}))
    print()

    for model in ("rf", "rbf-svm"):
        print(f"--- {model} ---")
        for name, X, y in (("A curated", Xc, yc), ("B natural", Xn, yn), ("C natural-bal", Xb, yb)):
            a, b, f = ev(X, y, model)
            print(f"  {name:<14} acc {a:6.2f}%   bal-acc {b:6.2f}%   macro-F1 {f:6.2f}%")
        print()


if __name__ == "__main__":
    main()
