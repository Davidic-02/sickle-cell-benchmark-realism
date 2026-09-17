"""Re-extract IDB2 cells with curated-matched framing, saved as images.

Curated IDB1 cells fill a median 35.2% of their frame; tight mask crops fill 85%.
Framing alone moved balanced accuracy ~9 points, so the natural arm is re-cut
to the curated convention. Each cell also records whether it touches the image
border (truncated cell) or whether its frame had to be clipped.

Writes data/cells_reframed/<class>/<smear>__<n>.png and data/cells_reframed_manifest.csv
"""
import csv, pathlib
import cv2, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eidb" / "erythrocytesIDB2"
OUT = ROOT / "data" / "cells_reframed"
CLASSES = ("circular", "elongated", "other")
FILL = 0.352            # curated median cell-bbox / frame area
MIN_SIDE, MAX_ASPECT = 20, 4.0

rows = []
for c in CLASSES:
    (OUT / c).mkdir(parents=True, exist_ok=True)
for d in sorted(RAW.iterdir()):
    if not d.is_dir() or not (d / "source.jpg").exists():
        continue
    src = cv2.imread(str(d / "source.jpg")); H, W = src.shape[:2]
    for c in CLASSES:
        mp = d / f"mask-{c}.jpg"
        if not mp.exists():
            continue
        m = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        mh, mw = m.shape; sx, sy = W / mw, H / mh
        n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
        k = 0
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < max(20, 1e-4 * mw * mh):
                continue
            bw, bh = w * sx, h * sy
            if min(bw, bh) < MIN_SIDE or max(bw, bh) / max(min(bw, bh), 1) > MAX_ASPECT:
                continue
            border = int(x <= 1 or y <= 1 or x + w >= mw - 1 or y + h >= mh - 1)
            cx, cy = (x + w / 2) * sx, (y + h / 2) * sy
            side = np.sqrt(bw * bh / FILL)
            x0, y0 = int(cx - side / 2), int(cy - side / 2)
            x1, y1 = int(cx + side / 2), int(cy + side / 2)
            clipped = int(x0 < 0 or y0 < 0 or x1 > W or y1 > H)
            crop = src[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
            if crop.size == 0:
                continue
            name = f"{d.name}__{k:03d}.png"
            cv2.imwrite(str(OUT / c / name), crop)
            rows.append(dict(path=f"cells_reframed/{c}/{name}", label=c, smear_id=d.name,
                             touches_border=border, frame_clipped=clipped))
            k += 1

with open(ROOT / "data" / "cells_reframed_manifest.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(f"reframed cells: {len(rows)}")
for c in CLASSES:
    r = [x for x in rows if x["label"] == c]
    print(f"  {c:<10} {len(r):5d}  border {sum(x['touches_border'] for x in r):4d}  clipped {sum(x['frame_clipped'] for x in r):4d}")
