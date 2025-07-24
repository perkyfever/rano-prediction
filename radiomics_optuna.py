import os
import sys
import json

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
from utils import RANDOM_SEED
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import warnings

warnings.simplefilter("ignore")


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


def build_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    # target_features = [
    #     list(set([
    #         "_".join(feature.split("_")[2:])
    #         for feature in dataframe.select_dtypes(include=["number"]).columns
    #         if feature not in ["label", "case_id", "patient_id", "fold"]
    #     ]))
    # ]

    # dataframes = []
    # merged_features = sum(target_features, [])

    # # Extract meta data
    # dataframes.append(dataframe[["fold", "label"]])

    # # Extract modality differences
    # for modality in ["T1", "T1CE", "T2", "FLAIR"]:
    #     baseline_features_keys = [f"baseline_{modality}_{feature}" for feature in merged_features]
    #     followup_features_keys = [f"followup_{modality}_{feature}" for feature in merged_features]

    #     features_delta_keys = [f"{modality}_{feature}_delta" for feature in merged_features]
    #     features_delta = dataframe[followup_features_keys].values - dataframe[baseline_features_keys].values
    #     new_features_keys = baseline_features_keys + followup_features_keys
    #     new_features = dataframe[baseline_features_keys + followup_features_keys]
    #     new_features = pd.concat([new_features, pd.DataFrame(data=features_delta, columns=features_delta_keys)], axis=1)
    #     new_dataframe = new_features
    #     # new_dataframe = pd.DataFrame(data=new_features, columns=new_features_keys)
    #     dataframes.append(new_dataframe)
    
    # dataframe = pd.concat(dataframes, axis=1)
    return dataframe


def get_fold_data(
    dataframe, fold_idx
) -> tuple[np.ndarray, np.array, np.ndarray, np.ndarray]:
    """
    Returns Train and Validation data for a given fold index.
    :param dataframe: dataframe with fold info
    :param fold_idx: Index of the fold (0 to 4)
    :return: X_train, y_train, X_valid, y_valid
    """
    target_features = list(dataframe.select_dtypes(include=["number"]).columns)
    if "label" not in target_features:
        target_features += ["label"]

    train_data = dataframe[dataframe["fold"] != fold_idx][target_features]
    valid_data = dataframe[dataframe["fold"] == fold_idx][target_features]

    X_train = train_data.drop(
        columns=["case_id", "patient_id", "fold", "label"], errors="ignore"
    ).values
    X_valid = valid_data.drop(
        columns=["case_id", "patient_id", "fold", "label"], errors="ignore"
    ).values

    y_train = train_data["label"].values
    y_valid = valid_data["label"].values

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


def run_experiment(dataframe, config, tuning=True):
    seed_everything()
    acc_scores, f1_scores = [], []
    auc_scores, ap_scores = [], []
    model = get_model(config)
    folds = [f"fold_{i}" for i in range(1, 6)]
    for fold_idx in folds:
        X_train, y_train, X_valid, y_valid = get_fold_data(dataframe, fold_idx)
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
        ap_scores.append(
            average_precision_score(y_valid, proba_pred, average="macro", pos_label=1)
        )

    report = {
        "acc_scores": acc_scores,
        "f1_scores": f1_scores,
        "auc_scores": auc_scores,
        "ap_scores": ap_scores,
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
    ap_scores = report["ap_scores"]

    final_score = sp.hmean(f1_scores)
    return final_score


def run_tuning(dataframe, base_config):
    def objective(trial: optuna.Trial):
        proposal_config = propose_config(base_config, trial)
        print(json.dumps(proposal_config, indent=4))
        experiment_report = run_experiment(dataframe, config=proposal_config)
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
    parser.add_argument(
        "--saveto", type=str, required=True, help="Path to save the results"
    )
    parser.add_argument(
        "--num_trials", type=int, default=100, help="Number of trials for tuning"
    )
    parser.add_argument(
        "--num_startup_trials", type=int, default=10, help="Number of startup trials"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of iterations for tuning"
    )
    parser.add_argument(
        "--early_stopping_rounds", type=int, default=20, help="Early stopping rounds"
    )
    parser.add_argument(
        "--exp_name", type=str, default="optuna", help="Experiment name"
    )

    args = parser.parse_args()

    DATA_PATH = Path(args.data)
    dataframe = pd.read_csv(DATA_PATH)

    kfold_split_path = Path(args.split)
    with open(kfold_split_path, "r") as f:
        kfold_split = json.load(f)

    def get_fold(patient_id: int) -> int:
        patient_id = f"patient_{patient_id:03d}"
        for fold_idx, fold_split in kfold_split.items():
            if patient_id in fold_split:
                return fold_idx

    dataframe["fold"] = dataframe["patient_id"].apply(get_fold)
    dataframe = build_dataframe(dataframe)

    tuning_config = {
        "num_trial": args.num_trials,
        "num_startup_trials": args.num_startup_trials,
        "iterations": args.iterations,
        "early_stopping_rounds": args.early_stopping_rounds,
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

    study = run_tuning(dataframe, base_config=tuning_config)

    best_params = study.best_params
    best_score = study.best_value

    model_params = study.best_params.copy()
    model_params["iterations"] = tuning_config["iterations"]
    model_params["eval_metric"] = tuning_config["eval_metric"]
    model_params["early_stopping_rounds"] = tuning_config["early_stopping_rounds"]

    report = run_experiment(dataframe, model_params)
    result = pd.DataFrame(report).reset_index().rename(columns={"index": "fold"})

    # test_report = run_experiment(dataframe, model_params, tuning=False)
    # test_result = pd.DataFrame(test_report)

    SAVE_TO = Path(args.saveto)
    os.makedirs(SAVE_TO / "optuna_results", exist_ok=True)
    output_path = SAVE_TO / "optuna_results" / f"{args.exp_name}.pkl"
    optuna_data = {
        "best_params": best_params,
        "best_score": best_score,
        "result": result,
        # "test_result": test_result,
    }
    pypickle.save(output_path, optuna_data, overwrite=True)
    print(f"Best Params: {best_params}")
    print(f"Successfully saved to {(SAVE_TO / 'optuna_results' / f'{args.exp_name}.pkl').as_posix()}")
