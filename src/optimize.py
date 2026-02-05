import os

import optuna
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from src.config import CFG
from src.train import MultiModalClassifier
from src.utils import WheatDataModule, seed_everything


def objective(trial: optuna.Trial) -> float:
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"
    cfg.EPOCHS = 30
    
    cfg.IMG_SIZE = trial.suggest_categorical("img_size", [64, 128, 224])
    cfg.LR = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    cfg.WD = trial.suggest_float("wd", 1e-3, 1e-1, log=True)
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    scheduler_type = trial.suggest_categorical("scheduler", ["onecycle", "cosine", "plateau"])
    aug_strength = trial.suggest_categorical("aug_strength", ["light", "medium", "strong"])
    
    cfg.RGB_BACKBONE = "efficientnet_b0"
    cfg.MS_BACKBONE = "efficientnet_b0"
    cfg.HS_BACKBONE = "resnet18"
    
    # cfg.RGB_BACKBONE = trial.suggest_categorical("rgb_backbone", ["efficientnet_b0", "resnet34", "mobilenetv3_large_100"])
    # cfg.MS_BACKBONE = trial.suggest_categorical("ms_backbone", ["efficientnet_b0", "resnet18", "resnet34"])
    # cfg.HS_BACKBONE = trial.suggest_categorical("hs_backbone", ["resnet18", "resnet34", "densenet121"])
    
    try:
        dm = WheatDataModule(cfg, aug_strength=aug_strength)
        dm.setup()
        
        model = MultiModalClassifier(
            cfg, 
            hs_channels=dm.hs_ch, 
            num_classes=3,
            label_smoothing=label_smoothing,
            dropout=dropout,
            scheduler_type=scheduler_type
        )
        
        trainer = pl.Trainer(
            max_epochs=cfg.EPOCHS,
            accelerator='auto',
            devices=1,
            callbacks=[
                EarlyStopping(monitor='val_f1', patience=10, mode='max')
            ],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        
        trainer.fit(model, dm)
        
        return trainer.callback_metrics['val_f1'].item()
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0


def run_optimization(n_trials: int = 50):
    seed_everything(3557)
    os.makedirs("./outputs", exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='wheat_multimodal',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val_f1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    optuna.visualization.plot_optimization_history(study).write_html("./outputs/optimization_history.html")
    optuna.visualization.plot_param_importances(study).write_html("./outputs/param_importances.html")
    
    best_config = {
        'val_f1': float(study.best_value),
        'trial': study.best_trial.number,
        'hyperparameters': study.best_params
    }
    
    with open("./outputs/best_params.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    return study.best_params


if __name__ == "__main__":
    best_params = run_optimization(n_trials=50)
    print("Saved best params to outputs/best_params.yaml")
