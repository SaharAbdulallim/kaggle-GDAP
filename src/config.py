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
    MS_BACKBONE: str = "resnet18"
    HS_BACKBONE: str = "resnet18"
    RGB_PRETRAINED_WEIGHTS: str = "imagenet"
    RGB_FREEZE_ENCODER: bool = True
    
    # PCA for Hyperspectral
    PCA_COMPONENTS: int = 20
    PCA_PATH: str = "./outputs/pca_hs.pkl"
    
    # Hyperparameters (tunable via Optuna)
    IMG_SIZE: int = 64
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LR: float = 2e-4
    WD: float = 0.05
    LABEL_SMOOTHING: float = 0.1
    DROPOUT: float = 0.5
    MIXUP_ALPHA: float = 0.2
    FUSION_TYPE: str = "concat"  # options: "concat", "attention"
    
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
    MS_STD: tuple = (0.209, 0.206, 0.209, 0.205, 0.206)
    
    # HS PCA stats (20 components) - update after running stats.py with PCA
    HS_MEAN: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    HS_STD: tuple = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    
    # Logging
    WANDB_ENABLED: bool = True
    WANDB_PROJECT_NAME: str = "wheat-disease-multimodal"
    WANDB_RUN_NAME: str = "default"


LABELS = ["Health", "Rust", "Other"]
LBL2ID = {k: i for i, k in enumerate(LABELS)}
ID2LBL = {i: k for k, i in LBL2ID.items()}