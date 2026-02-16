# Repository Summary

## What This Repo Does

A Kaggle competition solution for **Beyond Visible Spectrum AI for Agriculture 2026** -- classifying wheat patches into 3 classes (**Healthy / Rust / Other**) using multimodal UAV remote sensing imagery (RGB, Multispectral, Hyperspectral).

The approach is **not** deep learning. It is a hand-crafted feature engineering + LightGBM pipeline:

1. Load aligned multimodal images per sample (HS 125-band, MS 5-band, RGB 3-channel).
2. Extract ~600 engineered features spanning spectral, spatial, and cross-modal domains.
3. Select top features via mRMR, filter low-variance, scale, then train a LightGBM classifier.
4. Optionally tune hyperparameters with Optuna.
5. Ensemble multiple seed-varied models and produce a submission CSV.

---

## Architecture and Data Flow

```
data/train/                        data/val/
  HS/  MS/  RGB/                     HS/  MS/  RGB/
   |    |    |                        |    |    |
   v    v    v                        v    v    v
+----------------+               +----------------+
|  data.load_train|               |  data.load_test |
|  parse labels  |               |                 |
|  drop blanks   |               |                 |
+-------+--------+               +-------+--------+
        |                                |
  samples + labels                  test_samples
  (list of dicts:                   (list of dicts:
   hs, ms, rgb)                      hs, ms, rgb)
        |                                |
        v                                v
+------------------+             +------------------+
| features.extract |             | features.extract |
|   per sample:    |             |   per sample:    |
|                  |             |                  |
| MS ---+          |             | (same pipeline)  |
|  GLCM/LBP       |             +--------+---------+
|  spatial unif.   |                      |
|  spectral idx    |                      |
|                  |                X_test (N_test, ~600)
| RGB --+          |
|  color stats     |
|  HSV stats       |
|  texture         |
|                  |
| HS ---+          |
|  spectral shape  |
|  derivatives     |
|  spatial hetero. |
|  pixel distrib.  |
|  narrow-band idx |
|  red-edge onset  |
|  GLCM/LBP       |
|  consistency     |
|                  |
| Cross-modal --+  |
|  MS/HS/RGB cmp  |
+--------+---------+
         |
   X_train (N_train, ~600)
         |
         v
+------------------+           +------------------+
| [Optuna HPO]     |           |                  |
| (optional)       +---------->| evaluate()       |
| tunes LGB params |           | Stratified K-Fold|
| + n_features     |           |                  |
| + var_threshold   |           | per fold:        |
| + class weights   |           |  mRMR select     |
+------------------+           |  VarianceThresh   |
                               |  StandardScaler   |
                               |  LightGBM + early |
                               |    stopping       |
                               +--------+----------+
                                        |
                                  best params +
                                  best iterations
                                        |
                                        v
                               +------------------+
                               | train_final()    |
                               |                  |
                               | full-data mRMR   |
                               | VarianceThresh   |
                               | StandardScaler   |
                               |                  |
                               | fit 3 LightGBM   |
                               | (seeds 42,123,456)|
                               | capped at median |
                               |   CV iterations  |
                               +--------+---------+
                                        |
                              models + scaler + selector
                                        |
                                        v
                               +------------------+
                               | predict()        |
                               |                  |
                               | apply selector   |
                               | apply var filter  |
                               | apply scaler     |
                               | avg probabilities |
                               |   across 3 models|
                               | argmax -> labels  |
                               +--------+---------+
                                        |
                                        v
                               outputs/submission.csv
                               (Id, Category)
```

### Data shapes at each stage

| Stage | Train | Test |
|---|---|---|
| Raw HS | (32x32x125) float32 | same |
| Raw MS | (64x64x5) float32 | same |
| Raw RGB | (64x64x3) float32 | same |
| After feature extraction | (N, ~600) float32 | (N_test, ~600) |
| After mRMR selection | (N, 40-80) | (N_test, 40-80) |
| After variance threshold | (N, k) where k <= n_features | same |
| After scaling | (N, k) standardized | same |
| Model output | 3-class probabilities | same |

---

## Source Modules (`src/`)

### `config.py`

Central configuration dataclass (`CFG`). Holds all paths, class names, LightGBM defaults, Optuna trial counts, CV folds, ensemble seeds, and feature selection thresholds. Everything is overridable from CLI args in `main.py`.

### `data.py`

Data loading. Reads HS (`.tif`, 125 bands), MS (`.tif`, 5 bands), and RGB (`.png`) per sample. Parses class labels from filenames (`Health_hyper_*`, `Rust_hyper_*`, etc.). Drops blank/degenerate samples (mean < 1.0). Returns list-of-dicts with raw arrays + integer labels.

### `features.py`

The core of the pipeline -- ~570 lines of hand-crafted feature extraction. For each sample it computes:

| Domain | Features | Key Idea |
|---|---|---|
| **MS texture** | GLCM (contrast, homogeneity, energy, correlation, entropy, dissimilarity) + LBP histograms on NDVI, NIR, RedEdge at 64x64 | Spatial structure at higher resolution |
| **MS spatial uniformity** | Quadrant and 4x4 block NDVI mean/std spread | Rust lesions create patchy NDVI |
| **MS spectral indices** | NDVI, NDRE, GNDVI, CI-RE, CI-Green, SAVI -- each with full distributional stats | Standard vegetation health indicators |
| **RGB** | Per-channel stats, HSV stats, GLCM/LBP on grayscale, gradient magnitude | Coarse color and texture |
| **HS spectral shape** | Region area (VIS/RE/NIR), region fractions, median spectrum fractions, mean-median divergence, continuum removal at red absorption | Scale-invariant spectral signatures |
| **HS derivatives** | 1st/2nd order spectral derivatives, derivative values at diagnostic red-edge bands | Captures slope of spectral transitions |
| **HS spatial heterogeneity** | Quadrant spectral CV, quadrant NDVI spread, bright/dark pixel spectral angle | Rust patches vs uniform healthy tissue |
| **HS pixel distributions** | Entropy, skew, kurtosis, IQR at 6 diagnostic bands; NIR bimodality ratio | Shape of within-patch reflectance distribution |
| **HS pixel-level indices** | NDVI, NDRE, Red/NIR ratio, REIP (red-edge inflection point) -- distributional stats | Pixel-level disease indicators |
| **HS narrow-band ratios** | PRI, ARI, CRI, 45 pairwise normalized-difference ratios across 10 key bands | Disease-specific biochemical indices |
| **HS red-edge onset** | Mean, min, range, NIR-normalized values, per-pixel skew/kurtosis (bands 44-60) | 30% H-vs-R separation region per Mahlein et al. 2012 |
| **HS red absorption depth** | Depth relative to green+NIR continuum, distributional stats | Chlorophyll damage indicator |
| **HS GLCM/LBP** | On red-edge and NIR band averages | Spatial texture in critical spectral regions |
| **HS consistency** | Band-level CV, pixel-level CV, NIR CV, red-edge CV | Rust is spectrally consistent, Health is variable |
| **Cross-modal** | MS-vs-HS NDVI difference/product, per-band MS/HS spatial std ratio, RGB/HS contrast ratio | Resolution and modality agreement |

### `optimize.py`

Training, evaluation, hyperparameter search, and inference.

- **`mrmr_select`** -- Minimum Redundancy Maximum Relevance feature selection (Peng et al., IEEE TPAMI 2005). Greedy forward selection using MI for relevance and Pearson |r| for redundancy. Picks the most informative, least correlated subset from ~600 features.
- **`evaluate`** -- Stratified K-fold CV with per-fold mRMR selection, variance thresholding, standard scaling, and LightGBM with early stopping. Reports per-fold and mean train/val macro-F1.
- **`run_optimization`** -- Optuna TPE search over LightGBM hyperparameters, feature count, variance threshold, and class weights. Penalizes train-val gap > 0.10 to avoid overfitting.
- **`train_final`** -- Fits an ensemble of 3 LightGBM models (different random seeds) on full training data. Caps `n_estimators` at the median CV best iteration to prevent overtraining.
- **`predict`** -- Applies the saved mRMR selection, variance filter, and scaler to test data, averages predicted probabilities across the ensemble, and produces a submission DataFrame.

---

## How `main.py` Works

```
CLI args --> load config --> load data --> extract features --> [optuna HPO] --> evaluate --> train final --> predict --> save CSV
```

Step-by-step:

1. **Parse CLI args** -- `--run-optuna` (toggle HPO), `--trials`, `--folds`, `--features`.
2. **Load data** -- `load_train` and `load_test` read all modalities, drop blanks, assign labels.
3. **Extract features** -- `extract_batch` runs the full feature pipeline on every sample, producing `(N, ~600)` NumPy arrays.
4. **Optional Optuna HPO** -- If `--run-optuna`, searches LightGBM params + feature selection config. Updates `cfg` with best trial results. Otherwise uses defaults from `config.py`.
5. **Evaluate** -- Stratified 5-fold CV with per-fold mRMR feature selection. Prints per-fold F1, overall classification report, and confusion matrix.
6. **Train final** -- mRMR on full training set, variance filter, scale, fit 3-seed LightGBM ensemble capped at median CV iteration count.
7. **Predict + save** -- Apply pipeline to test features, average ensemble probabilities, write `outputs/submission.csv`.

---

## Optuna Hyperparameter Search

When `--run-optuna` is passed, Optuna's TPE sampler searches the following parameters jointly:

| Parameter | Search Space | Purpose |
|---|---|---|
| `n_features` | 40 -- 80 (step 10) | Number of features selected by mRMR |
| `var_threshold` | 1e-10 -- 1e-4 (log) | Variance filter cutoff post-mRMR |
| `n_estimators` | 300 -- 1200 | Boosting rounds (early-stopped per fold) |
| `max_depth` | 3 -- 6 | Tree depth |
| `learning_rate` | 0.01 -- 0.1 (log) | Shrinkage rate |
| `subsample` | 0.5 -- 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.3 -- 0.6 | Column sampling ratio |
| `min_child_samples` | 30 -- 100 | Minimum leaf samples |
| `reg_alpha` | 1.0 -- 50.0 (log) | L1 regularization |
| `reg_lambda` | 1.0 -- 50.0 (log) | L2 regularization |
| `num_leaves` | 8 -- 20 | Max leaves per tree |
| `min_split_gain` | 0.05 -- 0.5 | Minimum loss reduction to split |
| `path_smooth` | 0.5 -- 10.0 | Smoothing for leaf predictions |
| `extra_trees` | True / False | Extremely randomized tree splits |
| `health_weight` | 1.0 -- 2.5 | Class weight for Health (minority handling) |

The objective maximizes mean macro-F1 across stratified K-fold CV, with a linear penalty kicking in when the train-val gap exceeds 0.10. This steers the search away from overfitting configurations. Each trial runs full per-fold mRMR feature selection so the feature count and model params are co-optimized.

---

## Key Design Decisions

- **Feature engineering over deep learning** -- Small dataset (~hundreds of samples) with high-dimensional spectral data favors classical ML with domain-informed features.
- **Per-fold feature selection** -- mRMR runs inside each CV fold to prevent information leakage and give honest generalization estimates.
- **Overfitting control** -- Optuna objective penalizes train-val gap; early stopping in CV; iteration capping in final training; strong L1/L2 regularization defaults.
- **Multi-seed ensemble** -- 3 LightGBM models with different seeds for prediction stability.
