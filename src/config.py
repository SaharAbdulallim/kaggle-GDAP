import os
from dataclasses import dataclass, field


@dataclass
class CFG:
    ROOT: str = "./data"
    TRAIN_DIR: str = "train"
    TEST_DIR: str = "test"
    OUT_DIR: str = "./outputs"

    CLASSES: tuple = ("Health", "Other", "Rust")
    N_CLASSES: int = 3

    N_TOP_FEATURES: int = 120
    OPTUNA_TRIALS: int = 40
    CV_FOLDS: int = 5
    SEED: int = 62

    LGB_PARAMS: dict = field(
        default_factory=lambda: {
            "n_estimators": 2425,
            "max_depth": 4,
            "learning_rate": 0.017202488117011486,
            "subsample": 0.5353952394175464,
            "colsample_bytree": 0.4424202471887338,
            "min_child_samples": 11,
            "reg_alpha": 1.927937584863159,
            "reg_lambda": 1.0161807863995584,
            "num_leaves": 10,
            "boosting_type": "dart",
            "drop_rate": 0.09179696963549172,
            "skip_drop": 0.6633063543866614,
            "max_drop": 39,
        }
    )

    HEALTH_WEIGHT: float = 1.2897897441824462
    PSEUDO_THRESHOLD: float = 0.75
    PSEUDO_WEIGHT: float = 0.5405074842498068
    VAR_THRESHOLD: float = 1e-8
    PSEUDO_SEEDS: tuple = (42, 123, 456)

    BLANK_SAMPLES: frozenset = frozenset(
        {
            "Health_hyper_12",
            "Health_hyper_153",
            "Health_hyper_167",
            "Health_hyper_23",
            "Health_hyper_26",
            "Health_hyper_34",
            "Health_hyper_38",
            "Health_hyper_5",
            "Health_hyper_97",
            "Other_hyper_10",
            "Other_hyper_100",
            "Other_hyper_107",
            "Other_hyper_115",
            "Other_hyper_121",
            "Other_hyper_130",
            "Other_hyper_133",
            "Other_hyper_136",
            "Other_hyper_162",
            "Other_hyper_193",
            "Other_hyper_48",
            "Other_hyper_52",
            "Other_hyper_6",
            "Other_hyper_71",
        }
    )

    @property
    def class_map(self):
        return {c: i for i, c in enumerate(self.CLASSES)}

    @property
    def id_to_label(self):
        return {i: c for i, c in enumerate(self.CLASSES)}

    def train_path(self, modality):
        return os.path.join(self.ROOT, self.TRAIN_DIR, modality)

    def test_path(self, modality):
        return os.path.join(self.ROOT, self.TEST_DIR, modality)
