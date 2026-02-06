import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import StratifiedKFold
from torchmetrics import Accuracy, F1Score

from src.config import CFG, ID2LBL
from src.models import MultiModalClassifier
from src.utils import seed_everything

torch.set_float32_matmul_precision('medium')


def mixup_data(x, y, alpha=0.2):
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
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
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
        self.train_f1(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=False)
        return loss
    
    def on_train_epoch_end(self):
        train_f1 = self.train_f1.compute()
        self.log('epoch_train_f1', train_f1)
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_f1', 'interval': 'epoch'}}


def main():
    import joblib
    import kornia.augmentation as K
    from torch.utils.data import DataLoader, Subset

    from src.utils import WheatDataset, infer_hs_channels, make_df
    
    cfg = CFG()
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    train_df = make_df(cfg.ROOT, cfg.TRAIN_DIR)
    test_df = make_df(cfg.ROOT, cfg.VAL_DIR)
    hs_ch = infer_hs_channels(train_df, cfg)
    
    pca_model, pca_n_features = None, None
    if cfg.PCA_COMPONENTS > 0 and cfg.USE_HS and os.path.exists(cfg.PCA_PATH):
        pca_data = joblib.load(cfg.PCA_PATH)
        pca_model = pca_data['model'] if isinstance(pca_data, dict) else pca_data
        pca_n_features = pca_data.get('n_features', hs_ch) if isinstance(pca_data, dict) else hs_ch
        hs_ch = cfg.PCA_COMPONENTS
    
    train_transforms = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5), K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0, p=0.5),
        K.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.3),
        K.RandomGaussianNoise(mean=0., std=0.03, p=0.2), data_keys=["image"]
    )
    
    full_dataset = WheatDataset(train_df, cfg, hs_ch, train_transforms, pca_model, pca_n_features)
    test_dataset = WheatDataset(test_df, cfg, hs_ch, None, pca_model, pca_n_features)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    fold_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        print(f"\nFold {fold+1}/5: Train={len(tr_idx)}, Val={len(val_idx)}")
        
        train_subset = Subset(full_dataset, tr_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, cfg.BATCH_SIZE, True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, cfg.BATCH_SIZE, False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
        
        model = WheatClassifier(cfg, hs_ch, 3)
        checkpoint_cb = ModelCheckpoint(dirpath=f'{cfg.OUT_DIR}/fold_{fold}', filename='best', monitor='val_f1', mode='max')
        trainer = pl.Trainer(
            max_epochs=cfg.EPOCHS, callbacks=[checkpoint_cb, EarlyStopping(monitor='val_f1', patience=15, mode='max')],
            logger=WandbLogger(cfg.WANDB_PROJECT_NAME, f'{cfg.WANDB_RUN_NAME}_f{fold}') if cfg.WANDB_ENABLED else False,
            accelerator='auto', devices=1, precision='16-mixed', deterministic=True, enable_progress_bar=False
        )
        trainer.fit(model, train_loader, val_loader)
        fold_scores.append(checkpoint_cb.best_model_score.item())
    
    print(f"\n5-Fold CV: {' | '.join([f'F{i+1}={s:.3f}' for i,s in enumerate(fold_scores)])}")
    print(f"Mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    best_fold = np.argmax(fold_scores)
    model = WheatClassifier.load_from_checkpoint(f'{cfg.OUT_DIR}/fold_{best_fold}/best.ckpt', cfg=cfg, hs_channels=hs_ch, num_classes=3)
    test_loader = DataLoader(test_dataset, cfg.BATCH_SIZE, False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    preds = torch.cat([b['preds'] for b in pl.Trainer(accelerator='auto', devices=1).predict(model, test_loader)]).cpu().numpy()
    
    pd.DataFrame({
        'Id': [os.path.basename(test_df.iloc[i].get('hs','')) for i in range(len(test_df))],
        'Category': [ID2LBL[p] for p in preds]
    }).to_csv(f'{cfg.OUT_DIR}/submission.csv', index=False)


if __name__ == "__main__":
    main()
