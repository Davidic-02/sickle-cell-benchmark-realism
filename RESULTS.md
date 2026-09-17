# erythrocytesIDB: why curated-benchmark accuracy overstates smear-level performance

Working log. Numbers are RBF-SVM on 64x64 grayscale HOG, balanced accuracy,
class-balanced and size-matched arms, 5 subsample draws x 5-fold CV, unless noted.

## Data
- erythrocytesIDB v2 (Zenodo 10.5281/zenodo.18299474), md5 9d480fe1763f6ddb2b8bb4983cdb51e7, 1,299 images, no metadata files.
- **Curated arm:** IDB1 "individual cells" — 625 hand-picked cells (202 circular / 211 elongated / 212 other). No link to parent smear.
- **Natural arm:** cells cut from IDB2 smears using per-class masks. IDB2 masks ship at 1/4 source scale.
- IDB3 (undocumented on Zenodo) is 267x200 px; excluded from main comparisons as a resolution confound.
- Incomplete folders: IDB2-37 (no mask-circular), IDB3-08 (no mask.jpg, still usable), IDB3-11 (no source.jpg, unusable).
- `labeled.jpg` = source image with annotator letters (E/C/...) stamped on cells; masks follow those letters. Not an independent label reference.

## Findings, in the order they were tested

| # | Question | Result | Verdict |
|---|---|---|---|
| 1 | Does cell-level (vs smear-level) splitting inflate accuracy? | Inflation −0.4 to −0.5 pts across ridge / RF / RBF-SVM | **Rejected.** Single-lab dataset; little between-smear variation to exploit |
| 2 | Curated vs natural, balance- and size-matched | 92.7% vs 68.9%, gap **+23.7** | Gap is real |
| 3 | Is it extraction junk? | 19/2,263 crops (0.8%) suspect; excluding them changes nothing | Ruled out |
| 4 | Is it IDB3 low resolution? | IDB2-only, clean: gap +22.2 / +23.5 | Ruled out |
| 5 | Is it image quality (blur / contrast / colour)? | Degrading curated below natural sharpness: −0.3 pts. HOG is grayscale, so hue cannot act | Ruled out *for HOG* (contrast half weak: HOG block-norm is contrast-invariant) |
| 6 | Do subsets transfer? (tight crops) | curated→natural 47.0%, natural→curated 35.7% (chance 33.3%) | Non-exchangeable as cropped |
| 7 | **Crop framing** (curated cells fill 35.2% of frame, tight crops 85.3%) | Reframing natural: 69.0% → **77.7%**; explains **37%** of gap | Real, but **mechanism re-attributed in #15** (aspect distortion, not context) |
| 8 | Transfer after reframing | curated→natural **72.2%**, natural→curated **61.7%** | Framing drove most of the non-exchangeability |
| 9 | Per-class gap after reframing | +13.1 / +15.4 / +16.5 (was +11.4 / **+32.8** / +26.9) | Earlier "curation drops borderline sickle cells" reading **withdrawn** — that was framing |

Residual after undistorted square framing: **~11–15 pts, roughly uniform across classes.**

| 10 | Border-truncated / clipped cells | Excluding both: gap 12.5 → 10.5 | Small contributor |
| 11 | Native resolution (curated 80 px vs natural 303 px) | Res-matched gap 11.4 vs 11.2 | Ruled out |
| 12 | **Headline with intervals** (20 paired draws) | curated 91.3 [89.8, 93.2] vs natural 80.1 [76.0, 84.3]; **gap 11.2 [6.4, 15.5]** | Interval excludes zero |
| 13 | **CNN (ResNet18, grayscale, reframed, clean), 10 draws x 5 folds** | curated 93.2% vs natural 79.0%; **gap +14.2** (3-draw pilot: +12.6) | **Gap survives learned features** |
| 14 | **Context pilot**: same 1,305 cells at 6 framing levels, each with a background-masked twin | Context effect (unmasked − masked): +0.2 / +1.0 / +1.9 at fill 0.85–0.35 (CIs include 0); −2.6 / **−5.7** / **−10.0** at fill 0.20–0.07 | **Context does not help; wide context harms** |
| 15 | **Aspect distortion**: same 1,764 cells, tight non-square→64x64 vs tight letterboxed square vs square frame | 72.4% → 79.5% → 79.6%; undistorting **+7.1 [+3.1, +9.7]**, then context +0.2 [−4.7, +3.7]; elongated recall 68.5% → 85.2% | **#7's effect is aspect-ratio distortion** |

## Claims this supports
- Resizing non-square cell crops to a square input distorts shape and costs ~7 points of balanced accuracy (elongated recall −17 pts). Amount of surrounding context gives no benefit; wide context harms.
- Reported erythrocytesIDB accuracy does not describe performance on cells as they occur on smears; a ~15-point gap remains after controlling balance, size, resolution, extraction quality, blur and framing.

## Claims this does NOT support (yet)
- That the residual is caused by *curation*. Curated cells come from IDB1 smears, natural from IDB2; IDB1 ships no masks, so the two cannot be compared on the same smears.
- CNN natural fold range 65–93%: high variance at 100/class.

## Open
- Same-smear comparison impossible (IDB1 has no masks): the residual ~11 pts is "curated IDB1 vs natural IDB2", not proven to be curation.
- More CNN draws; per-class CNN confusion.

## Literature anchors (verified via Crossref / full text)
- González-Hidalgo et al. 2015, IEEE JBHI 19(4):1514–1525 — dataset describing publication (Zenodo `describes` link).
- Delgado-Font et al. 2020, Med Biol Eng Comput 58(6):1265–1284 — F-measure 0.97 normal / 0.95 elongated (dataset authors).
- Alzubaidi et al. 2020, Electronics 9(3):427 — 99.54% on erythrocytesIDB (99.98% with SVM), as quoted in Jennifer et al.
- Jennifer et al. 2023, Heliyon 9(11):e22203 — 629 cells augmented to 3,334/class; up to 100% (ResNet-50), ~96% test (VGG+RF).
- Dataset licence CC BY-NC-ND 4.0: derived crops cannot be redistributed — release code only.
- Documented 629 curated cells (203/212/214); archive contains 625 (202/211/212).
