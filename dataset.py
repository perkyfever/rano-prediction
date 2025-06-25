import os
from pathlib import Path
from dataclasses import dataclass

import monai
import monai.transforms as mt

import pandas as pd
from monai.data import Dataset as MDataset


@dataclass
class ImageInfo:
    patient_id: int
    modality: str
    order_type: str

    @classmethod
    def from_path(cls, image_path):
        patient_id = int(image_path.name.split("_")[1])
        modality = image_path.name[:-7].split("_")[-1]
        order_type = image_path.name[:-7].split("_")[-2]
        return cls(patient_id, modality, order_type)


def get_dataset(
    data_path: Path,
    indices: list[int] = None,
    transform: mt.Transform = None,
) -> list[dict]:
    """
    Get simple MRI dataset from a directory structure.
    :param data_path: Path to the dataset directory.
    :param labels: List of labels corresponding to each patient.
    :param indices: Optional list of patient indices to filter the dataset (trainval/test).
    :param transform: Optional MONAI transform to apply to the images.
    :return : A MONAI Dataset object containing the MRI images and labels.
    """
    scans_path = data_path / "scans"
    scans = list(scans_path.glob("*.nii.gz"))
    segmentations_path = data_path / "segmentations"
    segmentations = list(segmentations_path.glob("*.nii.gz"))
    images = sorted(scans + segmentations, key=lambda x: x.name)

    labels_path = data_path / "radiomics_features_corrected.csv"
    labels = pd.read_csv(ATLAS_LABELS_DIR)["response"].values

    cases_data = []
    for idx, label in enumerate(labels):
        patient_id = ImageInfo.from_path(images[10 * idx]).patient_id
        if indices is None or patient_id in indices:
            case_data = []
            for image in images[10 * idx : 10 * (idx + 1)]:
                image_info = ImageInfo.from_path(image)
                case_data.append(
                    (f"{image_info.order_type}_{image_info.modality}", image)
                )

            case_data.append(("label", label))
            cases_data.append(dict(case_data))

    monai_dataset = MDataset(data=cases_data, transform=transform)
    return monai_dataset
