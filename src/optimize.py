import os

import optuna
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from src.config import CFG
from src.train import MultiModalClassifier
from src.utils import WheatDataModule, seed_everything


def objective(trial: optuna.Trial) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"
    cfg.EPOCHS = 30
    
    cfg.IMG_SIZE = 224
    cfg.LR = 0.00023125569665524058
    cfg.WD = 0.044560031865346926
    cfg.BATCH_SIZE = 64
    cfg.LABEL_SMOOTHING = 0.13396786789053855
    cfg.DROPOUT = 0.44921894226882936
    cfg.SCHEDULER_TYPE = "onecycle"
    

    aug_strength = "medium"
    
    cfg.RGB_BACKBONE = trial.suggest_categorical("rgb_backbone", [
        "efficientnet_b0", "resnet18", "resnet34"
    ])
    cfg.MS_BACKBONE = trial.suggest_categorical("ms_backbone", [
        "efficientnet_b0", "resnet18", "resnet34"
    ])
    cfg.HS_BACKBONE = trial.suggest_categorical("hs_backbone", [
        "resnet18", "resnet34"
    ])
    cfg.FUSION_TYPE = trial.suggest_categorical("fusion_type", [
        "concat", "attention", "bilinear", "transformer"
    ])
    
    dm = WheatDataModule(cfg, aug_strength=aug_strength)
    dm.setup()
    
    model = MultiModalClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)
    
    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[
            EarlyStopping(monitor='val_f1', patience=15, mode='max')
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, dm)
    
    val_f1 = trainer.callback_metrics.get('val_f1', torch.tensor(0.0)).item()
    
    del model
    del trainer
    del dm
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    return val_f1


def run_optimization(n_trials: int = 40):
    seed_everything(3557)
    os.makedirs("./outputs", exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='wheat_fusion_comparison',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(
        objective, 
        n_trials=n_trials,
        gc_after_trial=True
    )
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val_f1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    optuna.visualization.plot_optimization_history(study).write_html("./outputs/optimization_history_v2.html")
    optuna.visualization.plot_param_importances(study).write_html("./outputs/param_importances_v2.html")
    
    best_config = {
        'val_f1': float(study.best_value),
        'trial': study.best_trial.number,
        'hyperparameters': study.best_params
    }
    
    with open("./outputs/best_params_v2.yaml", "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    return study.best_params


if __name__ == "__main__":
    best_params = run_optimization(n_trials=30)
    print("Saved best fusion type to outputs/best_params_v2.yaml")
