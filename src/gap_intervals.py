"""Headline gap with intervals, plus a native-resolution control.

Reframing matched *relative* framing, but crops still differ in native pixel size
before the 64x64 resize, so they discard different amounts of detail. Arm C
downsamples natural crops to the curated median native side first.

Paired by draw: each of 20 draws subsamples every arm with the same seed.
"""
import csv, pathlib, numpy as np, cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CL = ("circular", "elongated", "other"); S = 64; PER = 100; DRAWS = 20

def side(im): return float(np.sqrt(im.shape[0] * im.shape[1]))
def hog64(gray):
    g = cv2.resize(gray, (S, S), interpolation=cv2.INTER_AREA) / 255.
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")

cur = [(p, c) for c in CL for p in sorted((ROOT / "data/raw/eidb/erythrocytesIDB1/individual cells" / c).glob("*.jpg"))]
nat = [(ROOT / "data" / r["path"], r["label"]) for r in csv.DictReader(open(ROOT / "data/cells_reframed_manifest.csv"))
       if r["touches_border"] == "0" and r["frame_clipped"] == "0"]

Gc = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p, _ in cur]
Gn = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p, _ in nat]
sc, sn = np.median([side(g) for g in Gc]), np.median([side(g) for g in Gn])
print(f"native crop side (px): curated median {sc:.0f}  [{np.percentile([side(g) for g in Gc],5):.0f}-{np.percentile([side(g) for g in Gc],95):.0f}]"
      f"   natural median {sn:.0f}  [{np.percentile([side(g) for g in Gn],5):.0f}-{np.percentile([side(g) for g in Gn],95):.0f}]")
f = sc / sn
print(f"resolution-matching factor for natural crops: x{f:.3f}\n")

Xc = np.array([hog64(g) for g in Gc]); yc = np.array([c for _, c in cur])
Xn = np.array([hog64(g) for g in Gn]); yn = np.array([c for _, c in nat])
Xm = np.array([hog64(cv2.resize(g, (max(8, int(g.shape[1] * f)), max(8, int(g.shape[0] * f))),
                                interpolation=cv2.INTER_AREA)) for g in Gn])

def ev(X, y, seed):
    r = np.random.default_rng(seed)
    i = np.concatenate([r.choice(np.where(y == c)[0], PER, replace=False) for c in CL])
    v = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[i], y[i]):
        m = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")).fit(X[i][tr], y[i][tr])
        v.append(balanced_accuracy_score(y[i][te], m.predict(X[i][te])))
    return np.mean(v) * 100

A = np.array([ev(Xc, yc, s) for s in range(DRAWS)])
B = np.array([ev(Xn, yn, s) for s in range(DRAWS)])
C = np.array([ev(Xm, yn, s) for s in range(DRAWS)])
ci = lambda v: f"{v.mean():6.2f}  [95% {np.percentile(v,2.5):6.2f}, {np.percentile(v,97.5):6.2f}]"
print(f"RBF-SVM balanced accuracy, {PER}/class, {DRAWS} paired draws x 5-fold CV")
print(f"  curated                               {ci(A)}")
print(f"  natural (reframed, clean)             {ci(B)}")
print(f"  natural (reframed, clean, res-matched){ci(C)}")
print(f"\n  GAP curated - natural                 {ci(A - B)}")
print(f"  GAP curated - natural res-matched     {ci(A - C)}")
