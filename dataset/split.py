import os
import json
from pathlib import Path

TRAIN_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4"]
VALID_FOLDS = ["fold_5"]


def get_kfold_split() -> dict:
    """
    Returns a dictionary containing information about the data split.
    :return: dict containing split info
    """
    split_path = Path(os.path.abspath(__file__)).parent / "split.json"
    with open(split_path, "r") as f:
        kfold_split = json.load(f)

    return kfold_split


def get_trainval_ids() -> tuple[list[int], list[int]]:
    """
    Returns two lists of patient ids for train and validation datasets.
    :return: (train_ids, valid_ids)
    """
    kfold_split = get_kfold_split()
    train_ids = sum([kfold_split[fold] for fold in TRAIN_FOLDS], [])
    valid_ids = sum([kfold_split[fold] for fold in VALID_FOLDS], [])
    return train_ids, valid_ids
