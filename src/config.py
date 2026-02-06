from dataclasses import dataclass


@dataclass
class CFG:
    ROOT: str = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2026/Kaggle_Prepared"
    TRAIN_DIR: str = "train"
    VAL_DIR: str = "val"
    OUT_DIR: str = "./outputs"
    
    USE_RGB: bool = True
    USE_MS: bool = True
    USE_HS: bool = True
    
    # Architecture
    RGB_BACKBONE: str = "resnet34"
    MS_BACKBONE: str = "resnet10t"
    HS_BACKBONE: str = "resnet10t"
    RGB_PRETRAINED: bool = True
    RGB_FREEZE_ENCODER: bool = True
    MS_FREEZE_ENCODER: bool = True
    HS_FREEZE_ENCODER: bool = True
    
    # PCA for Hyperspectral
    PCA_COMPONENTS: int = 20
    PCA_PATH: str = "./outputs/pca_hs.pkl"
    
    # Hyperparameters (tunable via Optuna)
    IMG_SIZE: int = 64
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LR: float = 0.0001
    WD: float = 0.02
    LABEL_SMOOTHING: float = 0.1
    DROPOUT: float = 0.6
    MIXUP_ALPHA: float = 0.2
    
    NUM_WORKERS: int = 4
    SEED: int = 3557
    
    # Hyperspectral preprocessing
    HS_DROP_FIRST: int = 10
    HS_DROP_LAST: int = 14
    
    # Normalization (RGB uses ImageNet for frozen pretrained model)
    RGB_MEAN: tuple = (0.485, 0.456, 0.406)
    RGB_STD: tuple = (0.229, 0.224, 0.225)
    
    # MS/HS stats computed from training data - update after running stats.py
    MS_MEAN: tuple = (0.382, 0.404, 0.378, 0.397, 0.393)
    MS_STD: tuple = (0.211, 0.208, 0.211, 0.207, 0.208)
    
    # HS PCA stats (20 components) - update after running stats.py with PCA
    HS_MEAN: tuple = (-0.000003, -0.000006, -0.000005, -0.000001, 0.000004,
                      -0.000003, 0.000002, 0.000001, 0.000000, -0.000001,
                      -0.000002, 0.000001, 0.000000, 0.000001, -0.000000,
                      0.000000, -0.000000, 0.000000, 0.000000, -0.000000)
    HS_STD: tuple = (2.051, 0.892, 0.218, 0.181, 0.112,
                     0.081, 0.055, 0.046, 0.035, 0.029,
                     0.024, 0.021, 0.018, 0.016, 0.014,
                     0.013, 0.012, 0.011, 0.010, 0.009)
    
    # Logging
    WANDB_ENABLED: bool = True
    WANDB_PROJECT_NAME: str = "wheat-disease-multimodal"
    WANDB_RUN_NAME: str = "default"


LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}