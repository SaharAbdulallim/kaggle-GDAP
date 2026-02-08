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
    OPTUNA_TRIALS: int = 80
    CV_FOLDS: int = 5
    SEED: int = 42

    LGB_PARAMS: dict = field(
        default_factory=lambda: {
            "n_estimators": 1884,
            "max_depth": 3,
            "learning_rate": 0.007643295574118502,
            "subsample": 0.7110816720927418,
            "colsample_bytree": 0.3566403547984045,
            "min_child_samples": 26,
            "reg_alpha": 0.9397882625946417,
            "reg_lambda": 0.7402899334232234,
            "num_leaves": 46,
        }
    )

    HEALTH_WEIGHT: float = 1.5521
    PSEUDO_THRESHOLD: float = 0.75
    PSEUDO_WEIGHT: float = 0.7422
    PSEUDO_SEEDS: tuple = (42, 123, 456, 789, 1234, 2024, 3141)

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
            "Health_hyper_67",
            "Health_hyper_76",
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
