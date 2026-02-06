import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score

from src.config import CFG, ID2LBL
from src.models import MultiModalClassifier
from src.utils import WheatDataModule, seed_everything

torch.set_float32_matmul_precision('medium')


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = len(y)
    index = torch.randperm(batch_size, device=y.device)
    
    mixed_x = {}
    for key in x:
        mixed_x[key] = lam * x[key] + (1 - lam) * x[key][index]
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class WheatClassifier(pl.LightningModule):
    def __init__(self, cfg: CFG, hs_channels: int, num_classes: int = 3):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.model = MultiModalClassifier(cfg, hs_channels, num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if self.cfg.MIXUP_ALPHA > 0 and self.training:
            x_mixed, y_a, y_b, lam = mixup_data(x, y, self.cfg.MIXUP_ALPHA)
            logits = self(x_mixed)
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)
        
        self.train_acc(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
    
    def predict_step(self, batch, batch_idx):
        x, ids = batch
        logits = self(x)
        return {'ids': ids, 'preds': logits.argmax(1)}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.LR,
            weight_decay=self.cfg.WD
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1',
                'interval': 'epoch'
            }
        }


def main():
    cfg = CFG()
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    dm = WheatDataModule(cfg)
    dm.setup()
    
    model = WheatClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)
    print(f"Train: {len(dm.train_ds)} | Val: {len(dm.val_ds)} | Test: {len(dm.test_ds)} | HS: {dm.hs_ch} channels")
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.OUT_DIR,
        filename='best-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    
    early_stop_cb = EarlyStopping(monitor='val_f1', patience=15, mode='max')
    
    logger = False
    if cfg.WANDB_ENABLED:
        print("Logging with Weights & Biases")
        logger = WandbLogger(project=cfg.WANDB_PROJECT_NAME, name=cfg.WANDB_RUN_NAME)
    
    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        precision='16-mixed',
        deterministic=True
    )
    
    trainer.fit(model, dm)
    print(f"Best val_f1: {checkpoint_cb.best_model_score:.4f}")
    
    test_preds = trainer.predict(model, dm.test_dataloader(), ckpt_path='best')
    preds = torch.cat([batch['preds'] for batch in test_preds]).cpu().numpy()
    
    sub = pd.DataFrame({
        'Id': [os.path.basename(dm.test_df.iloc[i].get('hs') or dm.test_df.iloc[i].get('ms') or dm.test_df.iloc[i].get('rgb')) 
               for i in range(len(dm.test_df))],
        'Category': [ID2LBL[p] for p in preds]
    })
    sub_path = os.path.join(cfg.OUT_DIR, 'submission.csv')
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved: {sub_path}")


if __name__ == "__main__":
    main()