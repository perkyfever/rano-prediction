import json

from pathlib import Path
from mri.dataset.split import get_trainval_ids

import monai.transforms as mt
from monai.data import Dataset as MDataset

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from mri.utils import seed_everything, RANDOM_SEED


def get_dataset(
    data_path: Path,
    patients_ids: list[int] = None,
    transform: mt.Transform = None,
) -> tuple[MDataset, list[int]]:
    """
    Get simple MRI dataset from a directory structure.
    :param data_path: Path to the dataset directory.
    :param patients_ids: Optional list of patient ids to filter the dataset (trainval/test).
    :param transform: Optional MONAI transform to apply to the images.
    :return : A tuple of MONAI Dataset object containing the MRI images + labels and labels for convenience.
    """
    JSON_PATH = data_path / "patients_registered.json"
    with open(JSON_PATH, "r") as f:
        patients_dict = json.load(f)

    labels = []
    cases_data = []
    for patient_id, patient_data in patients_dict.items():
        if patients_ids is None or patient_id in patients_ids:
            for case_id, case_data in patient_data.items():
                case_info = case_data.copy()
                required_keys = ["T1", "T1CE", "T2", "FLAIR", "seg"]
                for key in required_keys:
                    folder = "segmentations" if key == "seg" else "scans"
                    case_info[f"baseline_{key}"] = (
                        data_path / folder / case_info[f"baseline_{key}"]
                    )
                    case_info[f"followup_{key}"] = (
                        data_path / folder / case_info[f"followup_{key}"]
                    )
                # case_info["patient_id"] = int(patient_id.split("_")[-1])
                # case_info["case_id"] = int(case_id.split("_")[-1])
                case_info["patient_id"] = patient_id
                case_info["case_id"] = case_id
                cases_data.append(case_info)
                labels.append(case_info["label"])

    monai_dataset = MDataset(data=cases_data, transform=transform)
    return monai_dataset, labels


def get_trainval_datasets(
    data_path: Path,
    train_transform: mt.Compose = None,
    valid_transform: mt.Compose = None,
) -> tuple[tuple[MDataset, list[int]], tuple[MDataset, list[int]]]:
    """
    Returns train and validation datasets.
    :param data_path: path to data folder
    :param train_transform: transform to apply for training samples
    :param valid_transform: transform to apply for validation samples
    :return: (train_dataset, valid_dataset)
    """
    train_ids, valid_ids = get_trainval_ids()

    if train_transform:
        train_transform = train_transform.set_random_state(
            seed=RANDOM_SEED
        )  # for reproducibility
    if valid_transform:
        valid_transform = valid_transform.set_random_state(
            seed=RANDOM_SEED
        )  # for reproducibility

    train_dataset = get_dataset(
        data_path=data_path, patients_ids=train_ids, transform=train_transform
    )
    valid_dataset = get_dataset(
        data_path=data_path, patients_ids=valid_ids, transform=valid_transform
    )

    return train_dataset, valid_dataset


def get_trainval_dataloaders(
    data_path: Path,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int = 4,
    train_transform: mt.Compose = None,
    valid_transform: mt.Compose = None,
    resample: bool = True,
    temperature: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns train and validation dataloaders.
    :param data_path: path to data folder
    :param train_batch_size: train batch size
    :param valid_batch_size: valid batch size
    :param num_workers: number of workers
    :param train_transform: transform to apply for training samples
    :param valid_transform: transform to apply for validation samples
    :param resample: if to use resampler proportional to class frequencies
    :param temperature: for class weights calculation
    :return: (train_loader, valid_loader)
    """
    generator = seed_everything()  # for reproducibility
    (train_dataset, train_labels), (valid_dataset, _) = get_trainval_datasets(
        data_path=data_path,
        train_transform=train_transform,
        valid_transform=valid_transform,
    )

    class_freqs = torch.bincount(torch.tensor(train_labels))
    class_weights = class_freqs.float() ** (-temperature)
    class_weights /= class_weights.sum()
    train_samples_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=train_samples_weights,
        num_samples=train_batch_size * (len(train_samples_weights) // train_batch_size),
        replacement=True,
        generator=generator,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=None if resample else True,
        sampler=sampler if resample else None,
        num_workers=num_workers,
        generator=generator,
    )

    return train_loader, valid_loader


def get_dataset(
    data_path: Path,
    patients_ids: list[int] = None,
    transform: mt.Transform = None,
) -> tuple[MDataset, list[int]]:
    """
    Get simple MRI dataset from a directory structure.
    :param data_path: Path to the dataset directory.
    :param patients_ids: Optional list of patient ids to filter the dataset (trainval/test).
    :param transform: Optional MONAI transform to apply to the images.
    :return : A MONAI Dataset object containing the MRI images + labels and labels for convenience.
    """
    JSON_PATH = data_path / "patients_registered.json"
    with open(JSON_PATH, "r") as f:
        patients_dict = json.load(f)

    labels = []
    cases_data = []

    for patient_id, patient_data in patients_dict.items():
        if patients_ids is None or patient_id in patients_ids:
            for case_id, case_data in patient_data.items():
                case_info = case_data.copy()
                for mod in ["T1", "T1CE", "T2", "FLAIR"]:
                    case_info[f"baseline_{mod}"] = data_path / "scans" / case_info[f"baseline_{mod}"]
                    case_info[f"followup_{mod}"] = data_path / "scans" / case_info[f"followup_{mod}"]
                
                case_info[f"baseline_seg"] = data_path / "segmentations" / case_info[f"baseline_seg"]
                case_info[f"followup_seg"] = data_path / "segmentations" / case_info[f"followup_seg"]

                case_info["patient_id"] = int(patient_id.split("_")[-1])
                case_info["case_id"] = int(case_id.split("_")[-1])
                cases_data.append(case_info)
                labels.append(case_info["label"])

    monai_dataset = MDataset(data=cases_data, transform=transform)
    return monai_dataset, labels
