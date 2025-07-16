import json
from pathlib import Path

import monai.transforms as mt
from monai.data import Dataset as MDataset


def get_dataset(
    data_path: Path,
    indices: list[int] = None,
    transform: mt.Transform = None,
) -> tuple[MDataset, list[int]]:
    """
    Get simple MRI dataset from a directory structure.
    :param data_path: Path to the dataset directory.
    :param indices: Optional list of patient indices to filter the dataset (trainval/test).
    :param transform: Optional MONAI transform to apply to the images.
    :return : A MONAI Dataset object containing the MRI images + labels and labels for convenience.
    """
    JSON_PATH = data_path / "patients_registered.json"
    with open(JSON_PATH, "r") as f:
        patients_dict = json.load(f)

    labels = []
    cases_data = []
    for patient_id, patient_data in patients_dict.items():
        if indices is None or patient_id in indices:
            for case_id, case_data in patient_data.items():
                case_info = case_data.copy()
                case_info["baseline_T1"] = (
                    data_path / "scans" / case_info["baseline_T1"]
                )
                case_info["baseline_T1CE"] = (
                    data_path / "scans" / case_info["baseline_T1CE"]
                )
                case_info["baseline_T2"] = (
                    data_path / "scans" / case_info["baseline_T2"]
                )
                case_info["baseline_FLAIR"] = (
                    data_path / "scans" / case_info["baseline_FLAIR"]
                )
                case_info["baseline_seg"] = (
                    data_path / "segmentations" / case_info["baseline_seg"]
                )
                case_info["followup_T1"] = (
                    data_path / "scans" / case_info["followup_T1"]
                )
                case_info["followup_T1CE"] = (
                    data_path / "scans" / case_info["followup_T1CE"]
                )
                case_info["followup_T2"] = (
                    data_path / "scans" / case_info["followup_T2"]
                )
                case_info["followup_FLAIR"] = (
                    data_path / "scans" / case_info["followup_FLAIR"]
                )
                case_info["followup_seg"] = (
                    data_path / "segmentations" / case_info["followup_seg"]
                )
                case_info["patient_id"] = int(patient_id.split("_")[-1])
                case_info["case_id"] = int(case_id.split("_")[-1])
                cases_data.append(case_info)
                labels.append(case_info["label"])

    monai_dataset = MDataset(data=cases_data, transform=transform)
    return monai_dataset, labels
