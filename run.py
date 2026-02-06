import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import pandas as pd
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    import torch

    from src.config import CFG, ID2LBL
    from src.train import MultiModalClassifier

    from src.utils import WheatDataModule, seed_everything

    from src.stats import calculate_stats
    return (
        EarlyStopping,
        ModelCheckpoint,
        MultiModalClassifier,
        WandbLogger,
        WheatDataModule,
        pd,
        pl,
        torch,
    )


@app.cell
def _(torch):
    import wandb 
    wandb.login()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.ipc_collect()
    return


app._unparsable_cell(
    r"""
    cfg = CFG()
    cfg.ROOT = "./data"
    cfg.TRAIN_DIR = "train"
    cfg.VAL_DIR = "test"
    cfg.OUT_DIR = "./outputs"


    cfg.EPOCHS = 50
    cfg.IMG_SIZE = 64
    cfg.BATCH_SIZE = 64
    cfg.LR = 0.00023125569665524058
    cfg.WD = 0.044560031865346926
    cfg.LABEL_SMOOTHING = 0.13396786789053855
    cfg.DROPOUT = 0.44921894226882936
    cfg.SCHEDULER_TYPE = 'onecycle'
    cfg.AUG_STRENGTH = 'medium'
    cfg.FUSION_TYPE = 'concat'

    cfg.RGB_BACKBONE = "resnet18"
    cfg.MS_BACKBONE = "resnet18"
    cfg.HS_BACKBONE = "resnet34"


    cfg.WANDB_ENABLED = True
    cfg.WANDB_RUN_NAME = f'modality_{cfg.FUSION_TYPE}_{cfg.RGB_BACKBONE}}'



    # Best params: {'img_size': 224, 'lr': 0.00023125569665524058, 'wd': 0.044560031865346926, 'batch_size': 64, 'label_smoothing': 0.13396786789053855, 'dropout': 0.44921894226882936, 'scheduler': 'onecycle', 'aug_strength': 'medium'}



    seed_everything(cfg.SEED)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    stats = calculate_stats(cfg, verbose=True)
    """,
    name="_"
)


@app.cell
def _(cfg, stats):
    cfg.MS_MEAN = stats['ms_mean']
    cfg.MS_STD = stats['ms_std']
    cfg.RGB_MEAN = stats['rgb_mean']
    cfg.RGB_STD = stats['rgb_std']
    return


@app.cell
def _(WheatDataModule, cfg, pd):
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


    print(f"Channels: {dm.n_ch} | HS: {dm.hs_ch}")
    print(f"Train samples: {len(dm.train_ds)}")
    print(f"Val samples: {len(dm.val_ds)}")
    print(f"Test samples: {len(dm.test_ds)}")
    return (dm,)


@app.cell
def _(
    EarlyStopping,
    ModelCheckpoint,
    MultiModalClassifier,
    WandbLogger,
    cfg,
    dm,
    pl,
):
    model = MultiModalClassifier(cfg, hs_channels=dm.hs_ch, num_classes=3)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.OUT_DIR,
        filename='best-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )

    early_stop_cb = EarlyStopping(monitor='val_f1', patience=20, mode='max')

    trainer = pl.Trainer(
        max_epochs=cfg.EPOCHS,
        accelerator='auto',
        devices=1,
        logger = WandbLogger(project=cfg.WANDB_PROJECT_NAME,name=cfg.WANDB_RUN_NAME) if cfg.WANDB_ENABLED else False , 
        callbacks=[checkpoint_cb, early_stop_cb],
        precision='16-mixed',
        deterministic=True
    )
    return checkpoint_cb, model, trainer


@app.cell
def _(dm, model, trainer):
    trainer.fit(model, dm)
    return


@app.cell
def _(checkpoint_cb):
    print(f"Best model: {checkpoint_cb.best_model_path}")
    print(f"Best validation F1: {checkpoint_cb.best_model_score:.4f}")
    return


@app.cell
def _():
    # test_preds = trainer.predict(model, dm.test_dataloader())
    # preds = torch.cat([batch['preds'] for batch in test_preds]).cpu().numpy()

    # sub = pd.DataFrame({
    #     'Id': [os.path.basename(dm.test_df.iloc[i].get('hs') or dm.test_df.iloc[i].get('ms') or dm.test_df.iloc[i].get('rgb')) 
    #            for i in range(len(dm.test_df))],
    #     'Category': [ID2LBL[p] for p in preds]
    # })
    # sub.to_csv(os.path.join(cfg.OUT_DIR, 'submission.csv'), index=False)
    # print(f"\nSubmission saved: {os.path.join(cfg.OUT_DIR, 'submission.csv')}")
    # print(sub['Category'].value_counts())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
