"""Paper figures, plotted only from files in results/ so they regenerate with the numbers.

Palette: reference categorical slots 1-2 (blue curated / orange natural), validated
all-pairs in light mode. Text stays in ink colours; bars start at zero.
"""
import csv, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R, OUT = ROOT / "results", ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})
N = r"([\d.]+)"
SN = r"([+-][\d.]+)"

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=SURF, dpi=300)
    plt.close(fig)

def bar(ax, x, h, color, lo=None, hi=None, label=None):
    ax.bar(x, h, width=0.36, color=color, edgecolor=SURF, linewidth=2, label=label, zorder=2)
    if lo is not None:
        ax.errorbar(x, h, yerr=[[h - lo], [hi - h]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    ax.text(x, (hi if hi is not None else h) + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=8, color=INK)

# ---- Fig 1: headline gap ---------------------------------------------------
g = open(R / "gap_intervals.txt").read()
hc = [float(v) for v in re.search(r"\n\s+curated\s+" + N + r"\s+\[95%\s+" + N + r",\s+" + N + r"\]", g).groups()]
hn = [float(v) for v in re.search(r"natural \(reframed, clean\)\s+" + N + r"\s+\[95%\s+" + N + r",\s+" + N + r"\]", g).groups()]
c = open(R / "cnn_gap_10draws.log").read()
cc = [float(v) for v in re.search(r"curated\s+bal-acc\s+" + N + r"% \(fold range " + N + "-" + N + r"\)\s+recall c/e/o " + N + "/" + N + "/" + N, c).groups()]
cn = [float(v) for v in re.search(r"natural\s+bal-acc\s+" + N + r"% \(fold range " + N + "-" + N + r"\)\s+recall c/e/o " + N + "/" + N + "/" + N, c).groups()]

fig, ax = plt.subplots(figsize=(4.4, 3.2))
bar(ax, -0.19, hc[0], S1, hc[1], hc[2], "Curated benchmark cells (IDB1)")
bar(ax, 0.19, hn[0], S2, hn[1], hn[2], "Cells extracted from smears (IDB2)")
bar(ax, 0.81, cc[0], S1)
bar(ax, 1.19, cn[0], S2)
ax.set_xticks([0, 1], ["HOG + RBF-SVM\n(95% interval, 20 draws)", "ResNet18\n(mean of 10 draws)"])
ax.set_ylim(0, 108); ax.set_ylabel("Balanced accuracy (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=8)
save(fig, "fig1_headline_gap")

# ---- Fig 2: aspect distortion ----------------------------------------------
a = open(R / "aspect_test.txt").read()
arms = re.findall(r"^([ABC]) .*?\s" + N + r" \[95%\s+" + N + r",\s+" + N + r"\]", a, re.M)
ba = re.search(r"undistorting \(B - A\):\s+" + SN + r" \[95%\s+" + SN + r",\s+" + SN + r"\]", a).groups()
cb = re.search(r"adding context after \(C - B\):\s+" + SN + r" \[95%\s+" + SN + r",\s+" + SN + r"\]", a).groups()
labels = ["Tight crop,\nstretched to square", "Tight crop,\npadded to square", "Square frame,\nmore context"]
fig, ax = plt.subplots(figsize=(4.4, 3.2))
for i, (_, m, lo, hi) in enumerate(arms):
    m, lo, hi = float(m), float(lo), float(hi)
    ax.bar(i, m, width=0.5, color=S1, edgecolor=SURF, linewidth=2, zorder=2)
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    ax.text(i, hi + 1.5, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
def bracket(x0, x1, yb, d):
    ax.plot([x0, x0, x1, x1], [yb - 1.5, yb, yb, yb - 1.5], color=INK2, linewidth=1)
    ax.text((x0 + x1) / 2, yb + 1.2, f"{float(d[0]):+.1f} pts  [{float(d[1]):+.1f}, {float(d[2]):+.1f}]",
            ha="center", va="bottom", fontsize=8, color=INK2)
bracket(0.05, 0.95, 90, ba)
bracket(1.05, 1.95, 101, cb)
ax.set_xticks(range(3), labels); ax.set_ylim(0, 110); ax.set_ylabel("Balanced accuracy (%)")
save(fig, "fig3_aspect_distortion")

# ---- Fig 3: context pilot ----------------------------------------------------
rows = list(csv.DictReader(open(R / "context_pilot.csv")))
x = np.arange(len(rows)); fills = [float(r["fill"]) for r in rows]
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for key, col, name in (("unmasked", S1, "Surroundings visible"), ("masked", S2, "Surroundings replaced by background")):
    m = np.array([float(r[key]) for r in rows]); lo = np.array([float(r[key + "_lo"]) for r in rows]); hi = np.array([float(r[key + "_hi"]) for r in rows])
    ax.fill_between(x, lo, hi, color=col, alpha=0.14, linewidth=0, zorder=1)
    ax.plot(x, m, color=col, linewidth=2, marker="o", markersize=5.5, markeredgecolor=SURF, markeredgewidth=1.5, label=name, zorder=3)
    ax.text(x[-1] + 0.12, m[-1], name.split(" ")[1] if key == "unmasked" else "background", va="center", fontsize=8, color=INK2)
ax.set_xticks(x, [f"{f:.2f}" for f in fills]); ax.set_xlim(-0.3, len(rows) - 0.3)
ax.set_xlabel("Cell bounding box / frame area   (tight  →  wide context)")
ax.set_ylabel("Balanced accuracy (%)"); ax.set_ylim(50, 92)
ax.legend(loc="lower left", fontsize=8)
save(fig, "fig4_context_pilot")

# ---- Fig 4: per-class recall -------------------------------------------------
pc = open(R / "reframed_transfer_perclass.txt").read()
hog_rows = re.findall(r"^(circular|elongated|other)\s+" + N + r"%\s+" + N + r"%", pc, re.M)
CL = ["circular", "elongated", "other"]
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
panels = [("HOG + RBF-SVM", [float(r[1]) for r in hog_rows], [float(r[2]) for r in hog_rows]),
          ("ResNet18", cc[3:6], cn[3:6])]
for ax, (title, cur, nat) in zip(axes, panels):
    for i in range(3):
        bar(ax, i - 0.19, cur[i], S1, label="Curated (IDB1)" if i == 0 else None)
        bar(ax, i + 0.19, nat[i], S2, label="Extracted from smears (IDB2)" if i == 0 else None)
    ax.set_xticks(range(3), CL); ax.set_title(title, fontsize=9, color=INK); ax.set_ylim(0, 108)
axes[0].set_ylabel("Recall (%)")
axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.12), ncol=2, fontsize=8)
save(fig, "fig2_per_class_recall")

print("figures:", sorted(p.name for p in OUT.glob("*.png")))
print("fig1 values", hc, hn, cc[0], cn[0])
