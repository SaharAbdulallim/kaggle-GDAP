import os

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score

from src.config import CFG, ID2LBL
from src.utils import WheatDataModule, seed_everything


class MultiModalClassifier(pl.LightningModule):
    def __init__(self, cfg: CFG, hs_channels: int, num_classes: int = 3,
                 label_smoothing: float = 0.0, dropout: float = 0.3, 
                 scheduler_type: str = "cosine"):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.scheduler_type = scheduler_type
        
        self.rgb_enc = timm.create_model(cfg.RGB_BACKBONE, pretrained=True, in_chans=3, num_classes=0)
        self.ms_enc = timm.create_model(cfg.MS_BACKBONE, pretrained=False, in_chans=5, num_classes=0)
        self.hs_enc = timm.create_model(cfg.HS_BACKBONE, pretrained=False, in_chans=hs_channels, num_classes=0)
        
        total_feat_dim = self.rgb_enc.num_features + self.ms_enc.num_features + self.hs_enc.num_features
        self.classifier = nn.Linear(total_feat_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
    def forward(self, modalities):
        rgb_feat = self.rgb_enc(modalities["rgb"])
        ms_feat = self.ms_enc(modalities["ms"])
        hs_feat = self.hs_enc(modalities["hs"])
        
        feat = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)
        return self.classifier(feat)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WD)
        
        if self.scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.LR,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_f1', 'interval': 'epoch'}}
        else:  # cosine
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.EPOCHS)
            return [optimizer], [scheduler]


def main():
    cfg = CFG()
    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    dm = WheatDataModule(cfg)
    dm.setup()
    
    train_labels = [dm.train_ds.df.iloc[i]['label'] for i in range(len(dm.train_ds))]
    val_labels = [dm.val_ds.df.iloc[i]['label'] for i in range(len(dm.val_ds))]
    
    train_dist = pd.Series(train_labels).value_counts().sort_index()
    val_dist = pd.Series(val_labels).value_counts().sort_index()
    

    print("DATASET DISTRIBUTION")

    print(f"\nTrain set ({len(train_labels)} samples):")
    for label, count in train_dist.items():
        pct = 100 * count / len(train_labels)
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nValidation set ({len(val_labels)} samples):")
    for label, count in val_dist.items():
        pct = 100 * count / len(val_labels)
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")

    
    print(f"Mode: MULTIMODAL")
    print(f"Channels: {dm.n_ch} | HS: {dm.hs_ch} | Train: {len(dm.train_ds)} | Val: {len(dm.val_ds)} | Test: {len(dm.test_ds)}")
    
    model = MultiModalClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.OUT_DIR,
        filename='best-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    
    early_stop_cb = EarlyStopping(monitor='val_f1', patience=15, mode='max')
    
    logger = WandbLogger(project=cfg.WANDB_PROJECT_NAME, name=None) if cfg.WANDB_ENABLED else True
    
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
    
    test_preds = trainer.predict(model, dm.test_dataloader(), ckpt_path='best')
    ids = [item for batch in test_preds for item in batch['ids']]
    preds = torch.cat([batch['preds'] for batch in test_preds]).cpu().numpy()
    
    sub = pd.DataFrame({
        'Id': [os.path.basename(dm.test_df.iloc[i].get('hs') or dm.test_df.iloc[i].get('ms') or dm.test_df.iloc[i].get('rgb')) 
               for i in range(len(dm.test_df))],
        'Category': [ID2LBL[p] for p in preds]
    })
    sub.to_csv(os.path.join(cfg.OUT_DIR, 'submission.csv'), index=False)
    print(f"Submission saved: {os.path.join(cfg.OUT_DIR, 'submission.csv')}")
    print(f"Best model: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()