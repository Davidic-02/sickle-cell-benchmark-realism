# From Curated Cells to Blood Smears: Assessing Benchmark Realism in Sickle Cell Morphology Classification

**David Adekoya**

Federal University of Technology, Akure, Nigeria

---

## Abstract

**Background.** Machine-learning models for classifying red blood cell morphology in sickle cell disease (SCD) report accuracies above 96% on the public erythrocytesIDB benchmark. These evaluations use a curated set of individually selected, class-balanced cell images. Whether such performance describes cells as they occur on a peripheral blood smear has not been examined.

**Methods.** We compared the benchmark's 625 curated cell images (erythrocytesIDB1) with 1,876 cells extracted from annotated full-field smear images in the same dataset (erythrocytesIDB2), using identical preprocessing, features, models and evaluation. Comparisons were class-balanced and size-matched, with paired subsampling over repeated stratified cross-validation. We tested candidate explanations for any difference in turn: class balance, sample size, extraction quality, image resolution, blur and contrast, crop geometry, border truncation and surrounding context. Results were replicated with a fine-tuned ResNet18.

**Results.** Balanced accuracy was 91.3% (95% interval 89.8–93.2) on curated cells and 80.1% (76.0–84.3) on smear-extracted cells, a gap of 11.2 percentage points (6.4–15.5), using HOG features with an RBF support vector machine. With ResNet18 the gap was 14.2 points (93.2% vs 79.0%). Resizing non-square cell crops to a square input distorted cell shape and alone reduced balanced accuracy by 7.1 points (3.1–9.7), cutting elongated-cell recall from 85.2% to 68.5%. Additional surrounding context gave no benefit, and wide context reduced accuracy by up to 10.0 points (3.0–16.4). Grouping cross-validation folds by source smear did not change performance.

**Conclusions.** Performance on erythrocytesIDB's curated cells does not transfer to cells extracted from smears in the same dataset, even after controlling for balance, size, resolution, image quality and crop geometry. Crop geometry is a large, easily overlooked source of variation in reported results. Because curated and smear-extracted cells come from different source images, the remaining gap cannot be attributed to curation alone. We recommend that SCD morphology benchmarks include smear-level cell populations and report crop construction explicitly.

---

## 1. Introduction

Sickle cell disease causes red blood cells to deform into elongated, sickle-like shapes, and quantifying these shapes in peripheral blood smears supports diagnosis and monitoring. Automated morphology classification is attractive where specialist microscopy review is scarce.

erythrocytesIDB [1, 2] is the principal public benchmark for this task. It contains Giemsa-stained smear images from patients with SCD at the General Hospital "Dr. Juan Bruno Zayas Alfonso", Santiago de Cuba, organised in three subsets: erythrocytesIDB1, with 196 full-field images and a set of individual cell images classified as circular, elongated or other; and erythrocytesIDB2 and erythrocytesIDB3, with 50 and 30 full-field images accompanied by per-class cell masks.

Published work on the individual-cell set reports high performance. The dataset authors reported F-measures of 0.97 for normal and 0.95 for elongated cells using shape descriptors [3]. Deep learning approaches report accuracies above 98% [4], up to 98.58% test accuracy for VGG16 and 100% precision and recall for ResNet-50 on a smaller subset [5], and 96.80% with CNN features and an SVM under stratified 5-fold cross-validation [6].

These results describe a specific population: cells individually selected by the dataset curators, centred in their frame and balanced across classes. A deployed system would instead encounter every cell on a smear, including borderline shapes and cells in crowded fields, in the natural class proportions. We asked how much of the reported performance survives that change of population, and which properties of the evaluation drive the difference.

## 2. Methods

### 2.1 Data

We used erythrocytesIDB Version 2 (Zenodo, DOI 10.5281/zenodo.18299474; archive MD5 `9d480fe1763f6ddb2b8bb4983cdb51e7`). The archive contains 1,299 images and no metadata files. The curated arm comprised the erythrocytesIDB1 individual cell images: 625 cells (202 circular, 211 elongated, 212 other). The dataset documentation lists 629 cells (203/212/214); four were absent from the archive.

The smear arm was derived from erythrocytesIDB2, whose full-field images are 3136 × 2352 px with per-class masks supplied at one-quarter resolution. erythrocytesIDB3 was excluded from the main comparisons because its source images are 267 × 200 px, which would confound resolution with population. One erythrocytesIDB2 folder lacked a circular-class mask and contributed only elongated and other cells.

### 2.2 Cell extraction

Cells were identified as connected components of each per-class mask, thresholded at mid-grey because masks are stored as JPEG. Components below 10⁻⁴ of the mask area (minimum 20 px), with a source-resolution side below 20 px, or with a bounding-box aspect ratio above 4 were discarded as specks or fragments (19 of 2,263 initially extracted cells, 0.8%). Cells touching the image border were excluded.

Each cell was cropped as a **square** frame centred on its bounding box, sized so the bounding box occupied 35.2% of the frame area, the median measured on curated crops. Cells whose frame would extend beyond the image were excluded. The final smear arm comprised 1,767–1,876 cells depending on the analysis.

### 2.3 Features and models

For the classical pipeline, images were converted to greyscale, resized to 64 × 64 px, and described with HOG features [7] (9 orientations, 8 × 8 px cells, 2 × 2 blocks, L2-Hys normalisation; 1,764 dimensions). Classification used an RBF support vector machine (C = 10, class-balanced weights) after standardisation, implemented in scikit-learn [8]. Random forest and ridge classifiers were used in secondary analyses.

For the deep learning pipeline, ResNet18 [9] initialised with ImageNet weights was fine-tuned on greyscale images replicated to three channels and resized to 96 × 96 px. Greyscale input was used because curated and smear images differ markedly in hue (Section 3.3). Training used AdamW (learning rate 3 × 10⁻⁴, weight decay 10⁻⁴), batch size 32, 12 epochs, with random 90° rotations and horizontal flips applied within training folds only.

### 2.4 Evaluation

Every comparison was **class-balanced and size-matched**: each arm was subsampled to the same number of cells per class (100 unless stated). Performance was estimated with stratified 5-fold cross-validation, and the primary metric was balanced accuracy. For the classical pipeline, subsampling was repeated over 20 draws paired across arms by random seed; we report the mean and a 95% percentile interval across draws, including for paired differences. ResNet18 was evaluated over 10 draws; we report the mean and the range across folds.

### 2.5 Analyses

Candidate explanations for any curated–smear difference were tested in sequence:

1. **Smear-grouped splitting.** Cell-level versus smear-grouped folds within the smear arm.
2. **Class balance and sample size.** Balanced, size-matched comparison.
3. **Extraction quality and resolution.** Exclusion of suspect crops; smear arm restricted to high-resolution erythrocytesIDB2 cells; natural crops downsampled to the curated native size (80 px) before feature extraction.
4. **Image quality.** Curated images blurred and contrast-matched to smear-image statistics.
5. **Cross-arm transfer.** Training on one arm and testing on the other.
6. **Crop geometry.** On identical cells: a tight non-square bounding box resized to square (distorting); the same box padded to square with background (undistorted); and a square frame with context.
7. **Border truncation.** Exclusion of border-touching cells and clipped frames.
8. **Surrounding context.** Six framing levels (bounding box occupying 85% down to 7% of frame area), each paired with a masked twin in which pixels outside the dilated cell mask were replaced by the frame's median background, isolating context from scale.

## 3. Results

### 3.1 Smear-grouped splitting does not change performance

Within the smear arm, grouping folds by source smear did not reduce accuracy relative to cell-level folds. The difference was −0.43 points for ridge, −0.45 for random forest and −0.49 for RBF-SVM; smear-grouped performance was marginally higher in every case. Leakage through shared smears does not explain the results that follow.

### 3.2 Smear-extracted cells are classified substantially worse

In the balanced, size-matched comparison with undistorted square crops, balanced accuracy was **91.3% (89.8–93.2) on curated cells and 80.1% (76.0–84.3) on smear-extracted cells**, a paired gap of **11.2 points (6.4–15.5)** (Figure 1). Downsampling smear crops to the curated native size left the gap unchanged at 11.4 points (6.2–15.5). Excluding border-touching cells and clipped frames reduced it from 12.5 to 10.5 points at 100 cells per class. ResNet18 reproduced the effect: **93.2% on curated and 79.0% on smear-extracted cells, a gap of 14.2 points** (fold ranges 83.3–98.3% and 65.0–93.3%).

![Figure 1](figures/fig1_headline_gap.png)

**Figure 1.** Balanced accuracy on curated benchmark cells versus cells extracted from smears in the same dataset. Both arms class-balanced and size-matched (100 cells per class). HOG + RBF-SVM error bars show 95% intervals across 20 paired draws; ResNet18 bars show the mean of 10 draws.

The shortfall was spread across classes (Figure 2). Recall on smear-extracted cells was 13.1, 15.4 and 16.5 points lower than on curated cells for circular, elongated and other cells respectively (HOG + RBF-SVM), and 14.5, 11.7 and 16.4 points lower with ResNet18.

![Figure 2](figures/fig2_per_class_recall.png)

**Figure 2.** Per-class recall on curated and smear-extracted cells, for each model. HOG + RBF-SVM at 108 cells per class; ResNet18 at 100 cells per class, mean of 10 draws.

### 3.3 Image quality does not explain the gap

Curated and smear images differed visibly: median hue 173° vs 286°, saturation 0.129 vs 0.210, greyscale contrast (standard deviation) 0.084 vs 0.041, and sharpness (Laplacian variance) 3.7 × 10⁻⁴ vs 2.7 × 10⁻⁴. This is despite the dataset documentation describing a single acquisition protocol across subsets. Hue cannot act on the greyscale pipelines. Blurring curated images to below smear-image sharpness and matching their contrast changed curated balanced accuracy by only −0.3 points (92.7% to 92.4%). Because HOG block normalisation is largely contrast-invariant, this test is informative for blur and weak for contrast.

### 3.4 Crop geometry is a large source of variation

On 1,764 identical cells, resizing a tight non-square crop to a square input produced **72.4% (68.3–76.0)** balanced accuracy; padding the same crop to square gave **79.5% (75.8–82.0)**; and a square frame with additional context gave **79.6% (75.3–82.9)** (Figure 3). Removing the distortion improved performance by **7.1 points (3.1–9.7)**, whereas adding context thereafter changed it by 0.2 points (−4.7 to 3.7). The effect fell mainly on elongated cells, the class whose shape the distortion alters most (median bounding-box aspect ratio 1.62, against 1.07 for circular cells). Elongated recall rose from 68.5% to 85.2% when crops were not distorted.

![Figure 3](figures/fig3_aspect_distortion.png)

**Figure 3.** Effect of crop geometry on identical cells. Stretching non-square crops to a square input distorts shape; padding to square removes the distortion; further context adds nothing. Error bars show 95% intervals across 20 paired draws.

Crop geometry also governed cross-arm transfer. With distorted crops, training on curated cells and testing on smear cells gave 47.0% balanced accuracy, and the reverse gave 35.7%, near chance (33.3%). With undistorted square crops these rose to 72.2% and 61.7%.

### 3.5 Surrounding context does not help

Across six framing levels on 1,305 cells (69 per class), showing the real surroundings rather than a background-masked frame had no detectable benefit at moderate framing: +0.2 (−2.5 to 2.5), +1.0 (−1.5 to 4.5) and +1.9 (−1.3 to 6.6) points at 85%, 50% and 35% fill. Wider context was harmful: −2.6 (−7.5 to 1.4), **−5.7 (−9.2 to −1.8)** and **−10.0 (−16.4 to −3.0)** points at 20%, 12% and 7% fill (Figure 4). Masked frames stayed between 74% and 78% throughout, indicating that the decline reflects neighbouring content rather than the smaller apparent cell size.

![Figure 4](figures/fig4_context_pilot.png)

**Figure 4.** Classification with increasing surrounding context. Each framing level is paired with a version in which everything outside the cell is replaced by background. Shaded bands show 95% intervals across 20 paired draws.

## 4. Discussion

On the erythrocytesIDB benchmark, curated-cell performance substantially overstates performance on cells extracted from annotated smears in the same dataset. The gap of 11–14 balanced-accuracy points survived controls for class balance, sample size, extraction quality, resolution, blur and crop geometry, and appeared with both hand-crafted features and a fine-tuned convolutional network.

Our absolute curated-arm accuracies (91–93%) are below published results (96–100%) [4–6]. This study measures a *relative* difference under a deliberately simple, fixed pipeline without augmentation-based dataset expansion. It does not attempt to reproduce state-of-the-art performance, and its conclusions concern how performance changes between populations under identical conditions.

The crop-geometry result has practical consequences beyond this dataset. Resizing a non-square bounding box to a fixed square input is a common default, yet on identical cells it cost seven points of balanced accuracy and seventeen points of elongated-cell recall. That is the class of greatest clinical interest in SCD. Reported differences between methods of a similar magnitude may reflect crop construction rather than modelling. We recommend that morphology studies state how crops are formed.

Two expected explanations did not hold. Grouping folds by source smear did not change performance, unlike settings where shared acquisition conditions between training and test data inflate results [10]; with a single laboratory and protocol there was little between-smear variation to exploit. Surrounding context also did not help cell classification, and wide context degraded it.

### Limitations

The central limitation is that **curated and smear-extracted cells come from different source images** (erythrocytesIDB1 and erythrocytesIDB2). erythrocytesIDB1 provides no masks, so cells cannot be extracted naturally from the images from which curated cells were drawn. The residual gap therefore reflects the combined effect of curation and of any systematic difference between subsets; the documented colour differences suggest such differences exist. It should not be attributed to the curators' selection alone.

Further limitations: labels come from a single specialist, with no independent reference. Masks are stored as lossy JPEG at reduced resolution. The "other" class is small (69–108 usable smear cells), widening intervals. The contrast component of the image-quality test is weak for HOG features. ResNet18 was evaluated over fewer draws than the classical pipeline. All data originate from one hospital. The dataset licence (CC BY-NC-ND 4.0) does not permit redistribution of derived cell crops, so we release code rather than extracted images.

## 5. Conclusions

Accuracy reported on curated erythrocytesIDB cells does not describe performance on cells extracted from smears in the same dataset, and crop geometry alone can shift results by several points. SCD morphology benchmarks should include smear-level cell populations in their natural proportions, and studies should report crop construction explicitly.

## Data and code availability

erythrocytesIDB is available from Zenodo (DOI 10.5281/zenodo.18299474) under CC BY-NC-ND 4.0. All extraction, analysis and figure code is available at [repository URL]; running it on the downloaded archive regenerates every reported number and figure.

## References

1. González-Hidalgo M, Guerrero-Peña FA, Herold-García S, Jaume-i-Capó A, Marrero-Fernández PD. Red blood cell cluster separation from digital images for use in sickle cell disease. *IEEE J Biomed Health Inform.* 2015;19(4):1514–1525. doi:10.1109/JBHI.2014.2356402
2. Marrero Fernández PD, Coello Said G, Delgado Font WE, et al. erythrocytesIDB (Version 2, October 2017) [dataset]. Zenodo. doi:10.5281/zenodo.18299474
3. Delgado-Font W, Escobedo-Nicot M, González-Hidalgo M, Herold-García S, Jaume-i-Capó A, Mir A. Diagnosis support of sickle cell anemia by classifying red blood cell shape in peripheral blood images. *Med Biol Eng Comput.* 2020;58(6):1265–1284. doi:10.1007/s11517-019-02085-9
4. Alzubaidi L, Fadhel MA, Al-Shamma O, Zhang J, Duan Y. Deep learning models for classification of red blood cells in microscopy images to aid in sickle cell anemia diagnosis. *Electronics.* 2020;9(3):427. doi:10.3390/electronics9030427
5. Jennifer SS, Shamim MH, Reza AW, Siddique N. Sickle cell disease classification using deep learning. *Heliyon.* 2023;9(11):e22203. doi:10.1016/j.heliyon.2023.e22203
6. Cardoso VJA, Moreira R, Mari JF, Moreira LFR. Improving sickle cell disease classification: a fusion of conventional classifiers, segmented images, and convolutional neural networks. In: *Encontro Nacional de Inteligência Artificial e Computacional (ENIAC)*; 2023. arXiv:2412.17975
7. Dalal N, Triggs B. Histograms of oriented gradients for human detection. In: *Proc IEEE CVPR.* 2005;1:886–893. doi:10.1109/CVPR.2005.177
8. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res.* 2011;12:2825–2830.
9. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: *Proc IEEE CVPR.* 2016:770–778. doi:10.1109/CVPR.2016.90
10. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns.* 2023;4(9):100804. doi:10.1016/j.patter.2023.100804
