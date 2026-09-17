"""Is the 'framing' effect really aspect-ratio distortion?

Original tight crops were non-square bboxes resized to 64x64, which squashes
elongated cells. Curated crops are square (80x80). Same cells, three crops:
  A tight bbox, non-square -> 64x64         (distorted; original pipeline)
  B tight bbox letterboxed to square        (same tightness, undistorted)
  C square frame, cell fills 35% of area    (reframed pipeline)
"""
import pathlib
import cv2, numpy as np
from skimage.feature import hog
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/eidb/erythrocytesIDB2"
CL = ("circular", "elongated", "other"); S, DRAWS = 64, 20

def H(g):
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    return hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")

A, B, C, y, asp = [], [], [], [], []
for d in sorted(RAW.iterdir()):
    if not d.is_dir() or not (d / "source.jpg").exists():
        continue
    g = cv2.cvtColor(cv2.imread(str(d / "source.jpg")), cv2.COLOR_BGR2GRAY); Hh, Ww = g.shape
    for c in CL:
        mp = d / f"mask-{c}.jpg"
        if not mp.exists():
            continue
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mh, mw = m.shape; sx, sy = Ww / mw, Hh / mh
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, yy, w, h, a = st[i]
            if a < max(20, 1e-4 * mw * mh) or x <= 1 or yy <= 1 or x + w >= mw - 1 or yy + h >= mh - 1:
                continue
            bw, bh = w * sx, h * sy
            if min(bw, bh) < 20 or max(bw, bh) / max(min(bw, bh), 1) > 4:
                continue
            cx, cy = (x + w / 2) * sx, (yy + h / 2) * sy
            side = np.sqrt(bw * bh / 0.35)
            if cx - side / 2 < 0 or cy - side / 2 < 0 or cx + side / 2 > Ww or cy + side / 2 > Hh:
                continue
            px, py = 0.02 * bw, 0.02 * bh
            ta = g[int(cy - bh / 2 - py):int(cy + bh / 2 + py), int(cx - bw / 2 - px):int(cx + bw / 2 + px)]
            L = max(ta.shape); bgv = int(np.median(ta))
            sq = np.full((L, L), bgv, np.uint8)
            oy, ox = (L - ta.shape[0]) // 2, (L - ta.shape[1]) // 2
            sq[oy:oy + ta.shape[0], ox:ox + ta.shape[1]] = ta
            fr = g[int(cy - side / 2):int(cy + side / 2), int(cx - side / 2):int(cx + side / 2)]
            A.append(H(ta)); B.append(H(sq)); C.append(H(fr)); y.append(c); asp.append(max(bw, bh) / min(bw, bh))

A, B, C, y, asp = map(np.array, (A, B, C, y, asp))
per = min(100, min(int((y == c).sum()) for c in CL))
print(f"cells {len(y)} {dict((c, int((y == c).sum())) for c in CL)} -> {per}/class")
print("median bbox aspect ratio: " + "  ".join(f"{c} {np.median(asp[y == c]):.2f}" for c in CL) + "\n")

def ev(X, seed):
    r = np.random.default_rng(seed)
    i = np.concatenate([r.choice(np.where(y == c)[0], per, replace=False) for c in CL])
    b, rc = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X[i], y[i]):
        p = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")).fit(X[i][tr], y[i][tr]).predict(X[i][te])
        b.append(balanced_accuracy_score(y[i][te], p)); rc.append(recall_score(y[i][te], p, labels=CL, average=None))
    return np.mean(b) * 100, np.mean(rc, 0) * 100

res = {}
for name, X in (("A tight, non-square (distorted)", A), ("B tight, letterboxed square", B), ("C square frame, fill 0.35", C)):
    out = [ev(X, s) for s in range(DRAWS)]
    v = np.array([o[0] for o in out]); rr = np.mean([o[1] for o in out], 0); res[name[0]] = v
    print(f"{name:<34} {v.mean():6.2f} [95% {np.percentile(v, 2.5):5.1f}, {np.percentile(v, 97.5):5.1f}]   recall c/e/o {rr[0]:.1f}/{rr[1]:.1f}/{rr[2]:.1f}")
ci = lambda v: f"{v.mean():+6.2f} [95% {np.percentile(v, 2.5):+5.1f}, {np.percentile(v, 97.5):+5.1f}]"
print(f"\nundistorting (B - A):           {ci(res['B'] - res['A'])}")
print(f"adding context after (C - B):   {ci(res['C'] - res['B'])}")
