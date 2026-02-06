# Wheat Disease Classification - Multimodal Remote Sensing

Competition: [Beyond Visible Spectrum AI for Agriculture 2026](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026)

## Problem

Classify wheat into 3 classes (Healthy/Rust/Other) using multimodal remote sensing:

- RGB: 3 channels (visible)
- Multispectral (MS): 5 channels (visible + NIR)
- Hyperspectral (HS): 101 channels (400-1000nm)

## Implementation Approach

### Architecture Design

**Core Principle:** Leverage pretrained knowledge where available, use lightweight custom models for spectral data, apply heavy regularization.

```
RGB (3ch)  → ResNet34 [FROZEN pretrained] + trainable head → 256-D
MS (5ch)   → SpectralAttention → ResNet18 [scratch] → 512-D
HS (101ch) → PCA(20) → SpectralAttention → ResNet18 [scratch] → 512-D
            ↓
       Fusion (concat or attention)
            ↓
       Classifier → 3 classes
```

Architecture

```
RGB (3ch)  → ResNet34 [frozen pretrained] → 256-D
MS (5ch)   → SpectralAttention → ResNet18 → 512-D
HS (101ch) → PCA(20) → SpectralAttention → ResNet18 → 512-D
            ↓
       Fusion (concat/attention)
            ↓
       Classifier → 3 classes
```

**Design Rationale:**

1. **Frozen RGB encoder**: Pretrained ImageNet features prevent overfitting on 180 RGB samples
2. **PCA (101→20)**: Removes spectral redundancy, keeps 95%+ variance, reduces params
3. **SpectralAttention**: Learns which bands matter for disease detection (~200 params)
4. **ResNet18 for MS/HS**: Balances spatial pattern learning with parameter efficiency
5. **Heavy regularization**: Dropout (0.5), weight decay (0.05), mixup (0.2), label smoothing (0.1)

**Normalization:** RGB uses ImageNet stats (frozen model requirement), MS/HS use computed dataset stats.
data/
├── train/
│   ├── RGB/
│   ├── MS/
│   └── HS/
└── test/
    ├── RGB/
    ├── MS/
    └── HS/

```

### 2. Compute Statistics & Fit PCA

```bash
python -m src.stats
```

This computes normalization statistics and fits PCA model on training data.

### 3. Train Model

```bash
**Setup:**
```bash
kaggle competitions download -c beyond-visible-spectrum-ai-for-agriculture-2026
unzip *.zip -d data/
```

**Train:**

```bash
python -m src.stats              # Fit PCA + compute normalization stats
python -m src.train              # Train model
python -m src.optimize --n_trials 50  # Hyperparameter search
```

Or use `run.ipynb` for interactive training

```python
HS_raw (H,W,101) → flatten → PCA.transform → reshape → (H,W,20)
```

## Results

Training monitors:

- **val_f1:** Primary metric (macro F1-score)
- **val_acc:** Classification accuracy
- **val_loss:** Cross-entropy loss

Early stopping based on val_f1 with patience=15 epochs.
Files

- `src/config.py` - Hyperparameters
- `src/models.py` - SpectralAttention + MultiModalClassifier
- `src/train.py` - Training loop with mixup
- `src/utils.py` - Data loading, PCA transform, augmentation
- `src/stats.py` - PCA fitting + stats computation
- `src/optimize.py` - Optuna hyperparameter search
- `run.ipynb` - End-to-end training notebook
