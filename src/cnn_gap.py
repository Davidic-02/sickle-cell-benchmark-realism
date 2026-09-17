"""CNN arm: does the curated-vs-natural gap survive learned features?

ResNet18 (ImageNet weights, fine-tuned). Designed to match the controlled HOG arms:
  * grayscale replicated to 3 channels -- curated/natural hue differs (173 vs 286 deg)
  * natural = reframed IDB2 cells (curated-matched framing), no border-touching, no clipped frame
  * class-balanced, size-matched (100/class), 3 draws x 5 folds
"""
import csv, pathlib, numpy as np, torch, torch.nn as nn, cv2
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CL = ("circular", "elongated", "other")
PER, SIZE, EPOCHS, DRAWS = 100, 96, 12, 10
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]

def img(p):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return (np.stack([g, g, g]) - MEAN) / STD

def curated():
    b = ROOT / "data/raw/eidb/erythrocytesIDB1/individual cells"
    P, y = [], []
    for i, c in enumerate(CL):
        for p in sorted((b / c).glob("*.jpg")):
            P.append(p); y.append(i)
    return P, np.array(y)

def natural():
    P, y = [], []
    for r in csv.DictReader(open(ROOT / "data/cells_reframed_manifest.csv")):
        if r["touches_border"] == "0" and r["frame_clipped"] == "0":
            P.append(ROOT / "data" / r["path"]); y.append(CL.index(r["label"]))
    return P, np.array(y)

def draw(P, y, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], PER, replace=False) for k in range(3)])
    return np.stack([img(P[i]) for i in idx]).astype(np.float32), y[idx]

def aug(x):
    x = torch.rot90(x, np.random.randint(4), (2, 3))
    return torch.flip(x, (3,)) if np.random.rand() < .5 else x

def fit_predict(X, y, tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = resnet18(weights=ResNet18_Weights.DEFAULT); m.fc = nn.Linear(m.fc.in_features, 3); m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr]).long()
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            nn.functional.cross_entropy(m(aug(Xt[b]).to(dev)), yt[b].to(dev)).backward()
            opt.step()
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            out.append(m(torch.from_numpy(X[te[i:i + 128]]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def run(name, P, y):
    bals, recs = [], []
    for d in range(DRAWS):
        X, yy = draw(P, y, d)
        for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=d).split(X, yy)):
            p = fit_predict(X, yy, tr, te, d * 10 + f)
            bals.append(balanced_accuracy_score(yy[te], p))
            recs.append(recall_score(yy[te], p, average=None, labels=[0, 1, 2]))
        print(f"  {name} draw {d}: running bal-acc {np.mean(bals) * 100:.2f}%", flush=True)
    return np.array(bals) * 100, np.mean(recs, 0) * 100

if __name__ == "__main__":
    print("device", dev, flush=True)
    a, ra = run("curated", *curated())
    b, rb = run("natural", *natural())
    lines = [f"CNN ResNet18 grayscale, {PER}/class, {DRAWS} draws x 5 folds, {EPOCHS} epochs",
             f"curated  bal-acc {a.mean():6.2f}% (fold range {a.min():.1f}-{a.max():.1f})  recall c/e/o {ra[0]:.1f}/{ra[1]:.1f}/{ra[2]:.1f}",
             f"natural  bal-acc {b.mean():6.2f}% (fold range {b.min():.1f}-{b.max():.1f})  recall c/e/o {rb[0]:.1f}/{rb[1]:.1f}/{rb[2]:.1f}",
             f"GAP {a.mean() - b.mean():+.2f} pts   (HOG+RBF-SVM reference: +10.47)"]
    print("\n".join(lines))
    (ROOT / "results/cnn_gap.txt").write_text("\n".join(lines) + "\n")
