import os

import optuna
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping

from src.config import CFG
from src.train import WheatClassifier
from src.utils import WheatDataModule, seed_everything

torch.set_float32_matmul_precision('medium')


def objective(trial: optuna.Trial) -> float:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"
    cfg.EPOCHS = 30
    cfg.WANDB_ENABLED = False
    
    cfg.DROPOUT = trial.suggest_float("dropout", 0.3, 0.6, step=0.1)
    cfg.WD = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    cfg.LABEL_SMOOTHING = trial.suggest_float("label_smoothing", 0.0, 0.15, step=0.05)
    cfg.MIXUP_ALPHA = trial.suggest_float("mixup_alpha", 0.0, 0.3, step=0.1)
    cfg.FUSION_TYPE = trial.suggest_categorical("fusion_type", ["concat", "attention"])
    cfg.BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64])
    cfg.LR = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    
    dm = WheatDataModule(cfg)
    dm.setup()
    
    model = WheatClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)
    
    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[EarlyStopping(monitor='val_f1', patience=10, mode='max')],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        precision='16-mixed',
        deterministic=True
    )
    
    trainer.fit(model, dm)
    
    val_f1 = trainer.callback_metrics.get('val_f1', torch.tensor(0.0)).item()
    
    del model, trainer, dm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return val_f1


def run_optimization(n_trials: int = 50):
    seed_everything(4433)
    os.makedirs("./outputs", exist_ok=True)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='wheat-multimodal-optimization',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    
    print(f"\nBest trial: {study.best_trial.number} | val_f1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=50)
    args = parser.parse_args()
    
    study = run_optimization(args.n_trials)
    
    best_config = {
        'val_f1': float(study.best_value),
        'trial': study.best_trial.number,
        'hyperparameters': study.best_params
    }
    
    with open("./outputs/best_params.yaml", "w", encoding="utf-8") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print("Saved to outputs/best_params.yaml")