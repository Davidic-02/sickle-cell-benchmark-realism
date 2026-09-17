"""Cut individual cells out of the mask-annotated smears (IDB2 + IDB3).

erythrocytesIDB ships 625 loose cells with no link to their parent smear.
IDB2/IDB3 instead give source.jpg plus per-class masks, so cells cut from
them carry a known smear id -- which is what makes a grouped split possible.

Writes data/cells/<class>/<smear_id>__<n>.png and data/cells_manifest.csv
"""
import csv
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "eidb"
OUT = ROOT / "data" / "cells"
CLASSES = ("circular", "elongated", "other")

MIN_AREA_FRAC = 1e-4   # of mask area -- masks ship at different scales per subset
MIN_AREA_FLOOR = 20    # px, in mask space
PAD_FRAC = 0.02        # context around the bbox, relative to cell size


def smear_dirs():
    for subset in ("erythrocytesIDB2", "erythrocytesIDB3"):
        for d in sorted((RAW / subset).iterdir()):
            if d.is_dir():
                yield subset, d


def load_mask(path):
    """Masks are JPEG, so they are not cleanly binary -- threshold at mid-grey."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 127).astype(np.uint8)


def main():
    for c in CLASSES:
        (OUT / c).mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for subset, d in smear_dirs():
        smear_id = d.name
        src_path = d / "source.jpg"
        if not src_path.exists():
            skipped.append((smear_id, "no source.jpg"))
            continue
        src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if src is None:
            skipped.append((smear_id, "unreadable source.jpg"))
            continue
        H, W = src.shape[:2]

        for cls in CLASSES:
            mpath = d / f"mask-{cls}.jpg"
            if not mpath.exists():
                skipped.append((smear_id, f"no mask-{cls}.jpg"))
                continue
            mask = load_mask(mpath)
            if mask is None:
                skipped.append((smear_id, f"mask-{cls} unreadable"))
                continue

            # IDB2 masks ship at quarter scale; work in mask space, scale boxes up.
            mh, mw = mask.shape[:2]
            sx, sy = W / mw, H / mh

            min_area = max(MIN_AREA_FLOOR, MIN_AREA_FRAC * mw * mh)
            n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            kept = 0
            for i in range(1, n_lab):                      # 0 is background
                x, y, w, h, area = stats[i]
                if area < min_area:
                    continue
                px, py = PAD_FRAC * w, PAD_FRAC * h
                x0 = max(0, int(round((x - px) * sx)))
                y0 = max(0, int(round((y - py) * sy)))
                x1 = min(W, int(round((x + w + px) * sx)))
                y1 = min(H, int(round((y + h + py) * sy)))
                crop = src[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                name = f"{smear_id}__{kept:03d}.png"
                cv2.imwrite(str(OUT / cls / name), crop)
                rows.append(
                    dict(path=f"cells/{cls}/{name}", label=cls, smear_id=smear_id,
                         subset=subset, area=int(area * sx * sy),
                         w=int(x1 - x0), h=int(y1 - y0))
                )
                kept += 1

    with open(ROOT / "data" / "cells_manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["path", "label", "smear_id", "subset", "area", "w", "h"])
        wr.writeheader()
        wr.writerows(rows)

    smears = sorted({r["smear_id"] for r in rows})
    print(f"cells extracted : {len(rows)}")
    print(f"source smears   : {len(smears)}")
    for cls in CLASSES:
        n = sum(r["label"] == cls for r in rows)
        print(f"  {cls:<10}: {n}")
    if skipped:
        print("\nskipped:")
        for s, why in skipped:
            print(f"  {s}: {why}")
    if not rows:
        sys.exit("no cells extracted")


if __name__ == "__main__":
    main()
