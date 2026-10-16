"""Does confidence-based referral recover curated-level accuracy on smear cells?

Hypothesis tested: curation behaves like an informal selective classifier, so a
model that refers its least confident cells to a human should reach curated-level
accuracy at some coverage. Natural arm = reframed IDB2 cells at their natural class
proportions, smear-grouped folds. Confidence = margin between the best and
runner-up SVM decision values. Result: balanced accuracy does not recover.
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score

CL = ("circular", "elongated", "other")
z = np.load("data/features_reframed.npz", allow_pickle=True); X, y, g = z["X"], z["y"], z["g"]
zc = np.load("data/features_curated.npz", allow_pickle=True); Xc, yc = zc["X"], zc["y"]
SEEDS = range(5)
COVS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)

def oof(X, y, groups, seed):
    S = np.zeros((len(y), 3))
    if groups is not None:
        folds = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        folds = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")).fit(X[tr], y[tr])
        S[te] = m.decision_function(X[te])
    return S

def curve(X, y, groups):
    rows = {c: [] for c in COVS}
    for s in SEEDS:
        S = oof(X, y, groups, s)
        pred = np.array(CL)[S.argmax(1)]
        top2 = np.sort(S, axis=1)[:, -2:]
        conf = top2[:, 1] - top2[:, 0]
        order = np.argsort(-conf)
        for c in COVS:
            keep = order[: int(round(c * len(y)))]
            rows[c].append((accuracy_score(y[keep], pred[keep]),
                            balanced_accuracy_score(y[keep], pred[keep]),
                            [float((y[keep] == k).sum() / (y == k).sum()) for k in CL]))
    return {c: (np.mean([r[0] for r in rows[c]]) * 100,
                np.mean([r[1] for r in rows[c]]) * 100,
                np.mean([r[2] for r in rows[c]], axis=0) * 100) for c in COVS}

nat = curve(X, y, g)
cur = curve(Xc, yc, None)
print(f"natural smear cells n={len(y)} {dict((k,int((y==k).sum())) for k in CL)}  |  curated n={len(yc)}")
print("confidence = SVM margin (best minus runner-up); smear-grouped CV for natural arm; 5 seeds\n")
print(f"{'coverage':>8} | {'NATURAL acc':>11} {'bal-acc':>8} | retained % circ/elong/other | {'CURATED acc':>11} {'bal-acc':>8}")
for c in COVS:
    a, b, r = nat[c]; ca, cb, _ = cur[c]
    print(f"{c:8.0%} | {a:10.2f}% {b:7.2f}% | {r[0]:5.1f} / {r[1]:5.1f} / {r[2]:5.1f}      | {ca:10.2f}% {cb:7.2f}%")
tgt = cur[1.0][1]
print(f"\ncurated full-coverage balanced accuracy = {tgt:.2f}%")
hit = [c for c in COVS if nat[c][1] >= tgt]
print("natural coverage at which balanced accuracy reaches that:", f"{max(hit):.0%}" if hit else "never reached in the range tested")
