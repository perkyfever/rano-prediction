import os
import sys

import utils
import pypickle
import argparse

import json
import random
import optuna

import numpy as np
import pandas as pd
import scipy.stats as sp

import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import warnings

warnings.simplefilter("ignore")

RANDOM_SEED = 0xBAD5EED


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def map_to_fold(json_name, fold_split):
    idx = int(json_name.split("_")[-1])
    for fold, indices in enumerate(fold_split):
        if idx in indices:
            return fold
    raise ValueError(f"Index {idx} not found in any fold.")


def get_fold_data(fold_idx) -> tuple[np.ndarray, np.array, np.ndarray, np.ndarray]:
    """
    Returns Train and Validation data for a given fold index.
    :param fold_idx: Index of the fold (0 to 4).
    :return: X_train, y_train, X_valid, y_valid
    """
    train_data = dataframe[dataframe["fold"] != fold_idx]
    valid_data = dataframe[dataframe["fold"] == fold_idx]
    X_train = train_data.drop(
        columns=["patient_json_file", "patient_image_path", "case", "response", "fold"]
    ).values
    y_train = train_data["response"].values
    X_valid = valid_data.drop(
        columns=["patient_json_file", "patient_image_path", "case", "response", "fold"]
    ).values
    y_valid = valid_data["response"].values
    return X_train, y_train, X_valid, y_valid


def get_model(config):
    return CatBoostClassifier(
        iterations=config["iterations"],
        learning_rate=config["learning_rate"],
        depth=config["depth"],
        l2_leaf_reg=config["l2_leaf_reg"],
        border_count=config["border_count"],
        bagging_temperature=config["bagging_temperature"],
        random_strength=config["random_strength"],
        rsm=config["rsm"],
        leaf_estimation_method=config["leaf_estimation_method"],
        boosting_type=config["boosting_type"],
        random_seed=RANDOM_SEED,
        verbose=False,
        eval_metric=config["eval_metric"],
        task_type="CPU",
    )


def run_experiment(config):
    seed_everything()
    acc_scores, f1_scores, auc_scores = [], [], []

    model = get_model(config)
    for fold_idx in range(len(fold_split)):
        X_train, y_train, X_valid, y_valid = get_fold_data(fold_idx)
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=config["early_stopping_rounds"],
            verbose=False,
        )
        y_pred = model.predict(X_valid)
        proba_pred = model.predict_proba(X_valid)
        acc_scores.append(accuracy_score(y_valid, y_pred))
        f1_scores.append(f1_score(y_valid, y_pred, average="macro"))
        auc_scores.append(
            roc_auc_score(y_valid, proba_pred, multi_class="ovr", average="macro")
        )

    report = {
        "acc_scores": acc_scores,
        "f1_scores": f1_scores,
        "auc_scores": auc_scores,
    }

    return report


def propose_config(config, trial):
    def propose_hparam_value(hparam_name, obj):
        hparam_value = obj
        if isinstance(obj, dict):
            distribution_type = obj["type"]
            distribution_kwargs = dict(filter(lambda p: p[0] != "type", obj.items()))
            suggest_fn = getattr(trial, f"suggest_{distribution_type}")
            hparam_value = suggest_fn(hparam_name, **distribution_kwargs)
        return hparam_value

    proposal_config = {}
    for hparam_name, obj in config.items():
        hparam_value = propose_hparam_value(hparam_name, obj)
        proposal_config[hparam_name] = hparam_value

    return proposal_config


def calculate_final_score(report):
    # OPTIMIZING FOR F1 SCORE RN
    acc_scores = report["acc_scores"]
    f1_scores = report["f1_scores"]
    auc_scores = report["auc_scores"]

    final_score = sp.hmean(f1_scores)
    return final_score


def run_tuning(base_config):
    def objective(trial: optuna.Trial):
        proposal_config = propose_config(base_config, trial)
        print(json.dumps(proposal_config, indent=4))
        experiment_report = run_experiment(proposal_config)
        sys.stdout.flush()
        return calculate_final_score(experiment_report)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=base_config["num_startup_trials"]
        ),
    )
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=base_config["num_trial"])
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Atlas registration script.")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to dataframe file"
    )
    parser.add_argument(
        "--split", type=str, required=True, help="Path to the split file"
    )

    args = parser.parse_args()

    DATA_PATH = Path(args.data)
    dataframe = pd.read_csv(DATA_PATH)

    split_path = Path(args.split)
    fold_split = list(pypickle.load(split_path, verbose="silent").values())

    dataframe["fold"] = dataframe["patient_json_file"].apply(
        lambda x: map_to_fold(x, fold_split)
    )

    tuning_config = {
        "num_trial": 100,
        "num_startup_trials": 10,
        "iterations": 100,
        "early_stopping_rounds": 20,
        "eval_metric": "TotalF1:average=Macro",
        # ===========================
        # Hyperparameters to tune
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "depth": {"type": "int", "low": 2, "high": 6},
        "l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "border_count": {"type": "int", "low": 32, "high": 128},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
        "random_strength": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "rsm": {"type": "float", "low": 0.5, "high": 1.0},
        "leaf_estimation_method": {
            "type": "categorical",
            "choices": ["Newton", "Gradient"],
        },
        "boosting_type": {"type": "categorical", "choices": ["Plain", "Ordered"]},
        # ===========================
    }

    study = run_tuning(tuning_config)

    best_params = study.best_params
    best_score = study.best_value

    model_params = study.best_params.copy()
    model_params["iterations"] = tuning_config["iterations"]
    model_params["eval_metric"] = tuning_config["eval_metric"]
    model_params["early_stopping_rounds"] = tuning_config["early_stopping_rounds"]

    report = run_experiment(model_params)
    result = pd.DataFrame(report).reset_index().rename(columns={"index": "fold"})

    output_path = DATA_PATH.parent / "optuna.pkl"
    optuna_data = {
        "best_params": best_params,
        "best_score": best_score,
        "result": result,
    }
    pypickle.save(output_path, optuna_data, overwrite=True)
    print(f"Best Params: {best_params}")
