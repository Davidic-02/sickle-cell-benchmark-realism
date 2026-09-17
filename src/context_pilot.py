"""Context pilot: does surrounding smear help classify a cell?

Each IDB2 cell is cut at several context levels (cell bbox as a fraction of frame
area: 0.85 tight ... 0.07 wide). Widening the frame also shrinks the cell in the
64x64 input, so every level has a MASKED twin: identical frame and scale, but all
pixels outside the (dilated) cell mask replaced by background. unmasked - masked
isolates information carried by the neighbourhood rather than by scale.

The same cells are used at every level (cells whose widest frame would leave the
image are dropped). Paired draws, class-balanced, 5-fold CV, RBF-SVM on HOG.
"""
import csv, pathlib
import cv2, numpy as np
from skimage.feature import hog
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/eidb/erythrocytesIDB2"
CL = ("circular", "elongated", "other")
FILLS = (0.85, 0.50, 0.35, 0.20, 0.12, 0.07)
S, DRAWS = 64, 20

def H(g):
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")

U = [[] for _ in FILLS]; M = [[] for _ in FILLS]; y = []; groups = []
for d in sorted(RAW.iterdir()):
    if not d.is_dir() or not (d / "source.jpg").exists():
        continue
    gray = cv2.cvtColor(cv2.imread(str(d / "source.jpg")), cv2.COLOR_BGR2GRAY); Hh, Ww = gray.shape
    for c in CL:
        mp = d / f"mask-{c}.jpg"
        if not mp.exists():
            continue
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mh, mw = m.shape; sx, sy = Ww / mw, Hh / mh
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, yy, w, h, a = st[i]
            if a < max(20, 1e-4 * mw * mh) or x <= 1 or yy <= 1 or x + w >= mw - 1 or yy + h >= mh - 1:
                continue
            bw, bh = w * sx, h * sy
            if min(bw, bh) < 20 or max(bw, bh) / max(min(bw, bh), 1) > 4:
                continue
            cx, cy = (x + w / 2) * sx, (yy + h / 2) * sy
            big = np.sqrt(bw * bh / min(FILLS))
            if cx - big / 2 < 0 or cy - big / 2 < 0 or cx + big / 2 > Ww or cy + big / 2 > Hh:
                continue
            # cell mask at source resolution, local window only
            X0, Y0 = int(cx - big / 2), int(cy - big / 2); X1, Y1 = int(cx + big / 2), int(cy + big / 2)
            comp = (lab == i).astype(np.uint8)
            cm = cv2.resize(comp, (Ww, Hh), interpolation=cv2.INTER_NEAREST)[Y0:Y1, X0:X1]
            k = max(3, int(0.06 * max(bw, bh))) | 1
            cm = cv2.dilate(cm, np.ones((k, k), np.uint8))
            win = gray[Y0:Y1, X0:X1]
            bg = np.median(win[cm == 0]) if (cm == 0).any() else np.median(win)
            masked_win = np.where(cm > 0, win, bg).astype(np.uint8)
            for j, f in enumerate(FILLS):
                side = np.sqrt(bw * bh / f)
                a0, b0 = int(cx - side / 2) - X0, int(cy - side / 2) - Y0
                a1, b1 = int(cx + side / 2) - X0, int(cy + side / 2) - Y0
                a0, b0 = max(0, a0), max(0, b0)
                U[j].append(H(win[b0:b1, a0:a1])); M[j].append(H(masked_win[b0:b1, a0:a1]))
            y.append(c); groups.append(d.name)

y = np.array(y); U = [np.array(u) for u in U]; M = [np.array(v) for v in M]
per = min(100, min(int((y == c).sum()) for c in CL))
print(f"cells used at every level: {len(y)}  {dict((c, int((y == c).sum())) for c in CL)}  -> {per}/class", flush=True)

def ev(X, seed):
    r = np.random.default_rng(seed)
    i = np.concatenate([r.choice(np.where(y == c)[0], per, replace=False) for c in CL])
    v = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[i], y[i]):
        mdl = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")).fit(X[i][tr], y[i][tr])
        v.append(balanced_accuracy_score(y[i][te], mdl.predict(X[i][te])))
    return np.mean(v) * 100

rows = []
ci = lambda v: (v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5))
print(f"\n{'fill':>5} {'unmasked':>22} {'masked':>22} {'context effect (U-M)':>26}")
for j, f in enumerate(FILLS):
    u = np.array([ev(U[j], s) for s in range(DRAWS)]); mm = np.array([ev(M[j], s) for s in range(DRAWS)])
    a, b, dlt = ci(u), ci(mm), ci(u - mm)
    print(f"{f:5.2f} {a[0]:6.2f} [{a[1]:5.1f},{a[2]:5.1f}] {b[0]:6.2f} [{b[1]:5.1f},{b[2]:5.1f}] {dlt[0]:+7.2f} [{dlt[1]:+5.1f},{dlt[2]:+5.1f}]", flush=True)
    rows.append(dict(fill=f, unmasked=a[0], unmasked_lo=a[1], unmasked_hi=a[2], masked=b[0], masked_lo=b[1],
                     masked_hi=b[2], effect=dlt[0], effect_lo=dlt[1], effect_hi=dlt[2]))
with open(ROOT / "results/context_pilot.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print("\nwrote results/context_pilot.csv")
