import os

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score

from src.config import CFG, ID2LBL
from src.utils import WheatDataModule, seed_everything


class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x):
        B, N, D = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).mean(dim=1)
        return out


class BilinearFusion(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim1, dim2, out_dim)
        
    def forward(self, x1, x2):
        return self.bilinear(x1, x2)


class TransformerFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.norm(x)
        out, _ = self.attn(x, x, x)
        return out.mean(dim=1)


class MultiModalClassifier(pl.LightningModule):
    def __init__(self, cfg: CFG, hs_channels: int, num_classes: int = 3, fusion_type: str = "concat"):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.fusion_type = fusion_type
        
        self.rgb_enc = timm.create_model(cfg.RGB_BACKBONE, pretrained=True, in_chans=3, num_classes=0)
        self.ms_enc = timm.create_model(cfg.MS_BACKBONE, pretrained=False, in_chans=5, num_classes=0)
        self.hs_enc = timm.create_model(cfg.HS_BACKBONE, pretrained=False, in_chans=hs_channels, num_classes=0)
        
        rgb_dim = self.rgb_enc.num_features
        ms_dim = self.ms_enc.num_features
        hs_dim = self.hs_enc.num_features
        
        if fusion_type == "attention":
            max_dim = max(rgb_dim, ms_dim, hs_dim)
            self.rgb_proj = nn.Linear(rgb_dim, max_dim) if rgb_dim != max_dim else nn.Identity()
            self.ms_proj = nn.Linear(ms_dim, max_dim) if ms_dim != max_dim else nn.Identity()
            self.hs_proj = nn.Linear(hs_dim, max_dim) if hs_dim != max_dim else nn.Identity()
            self.fusion = AttentionFusion(max_dim)
            self.classifier = nn.Linear(max_dim, num_classes)
        elif fusion_type == "bilinear":
            self.fusion = BilinearFusion(rgb_dim + ms_dim, hs_dim, 512)
            self.classifier = nn.Linear(512, num_classes)
        elif fusion_type == "transformer":
            max_dim = max(rgb_dim, ms_dim, hs_dim)
            self.rgb_proj = nn.Linear(rgb_dim, max_dim) if rgb_dim != max_dim else nn.Identity()
            self.ms_proj = nn.Linear(ms_dim, max_dim) if ms_dim != max_dim else nn.Identity()
            self.hs_proj = nn.Linear(hs_dim, max_dim) if hs_dim != max_dim else nn.Identity()
            self.fusion = TransformerFusion(max_dim)
            self.classifier = nn.Linear(max_dim, num_classes)
        else:  # concat
            total_feat_dim = rgb_dim + ms_dim + hs_dim
            self.classifier = nn.Linear(total_feat_dim, num_classes)
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)
        
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
    
    def forward(self, modalities):
        rgb_feat = self.rgb_enc(modalities["rgb"])
        ms_feat = self.ms_enc(modalities["ms"])
        hs_feat = self.hs_enc(modalities["hs"])
        
        if self.fusion_type == "attention":
            rgb_feat = self.rgb_proj(rgb_feat)
            ms_feat = self.ms_proj(ms_feat)
            hs_feat = self.hs_proj(hs_feat)
            stacked = torch.stack([rgb_feat, ms_feat, hs_feat], dim=1)
            fused = self.fusion(stacked)
        elif self.fusion_type == "bilinear":
            combined = torch.cat([rgb_feat, ms_feat], dim=1)
            fused = self.fusion(combined, hs_feat)
        elif self.fusion_type == "transformer":
            rgb_feat = self.rgb_proj(rgb_feat)
            ms_feat = self.ms_proj(ms_feat)
            hs_feat = self.hs_proj(hs_feat)
            stacked = torch.stack([rgb_feat, ms_feat, hs_feat], dim=1)
            fused = self.fusion(stacked)
        else:  # concat
            fused = torch.cat([rgb_feat, ms_feat, hs_feat], dim=1)
        
        return self.classifier(fused)
    
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
        
        if self.cfg.SCHEDULER_TYPE == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.LR,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.3,
                anneal_strategy='cos'
            )
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        elif self.cfg.SCHEDULER_TYPE == "plateau":
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
    logger = False
    if cfg.WANDB_ENABLED:
        print("Initializing Weights & Biases logger...")
        wandb.init(project=cfg.WANDB_PROJECT_NAME, name=cfg.WANDB_RUN_NAME if cfg.WANDB_RUN_NAME else 'default')
        logger = WandbLogger(project=cfg.WANDB_PROJECT_NAME, name=cfg.WANDB_RUN_NAME if cfg.WANDB_RUN_NAME else 'default')
 
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