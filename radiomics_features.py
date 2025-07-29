import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.parent.as_posix())

import argparse
import pandas as pd
import SimpleITK as sitk

from dataset import get_dataset

import logging
from tqdm import tqdm

from radiomics import featureextractor
from radiomics import logging as radiomics_logging


DEFAULT_PARAMS = {
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "normalize": True,
    "removeOutliers": 3,
    "label": 1,
    "verbose": False,
}


def extract_features(
    data_path: Path,
    file_name: str = "radiomics.csv",
    extractor_params: dict = DEFAULT_PARAMS,
) -> None:
    """
    Extract and save pyradiomics features (to the same provided data_path).
    :param data_path: path to data
    :param file_name: name for features csv file
    :param extractor_params: pyradiomics extractor params
    """
    dataset, _ = get_dataset(data_path=data_path)

    # Disable all radiomics logging below ERROR level
    radiomics_logger = radiomics_logging.getLogger()
    radiomics_logger.setLevel(logging.ERROR)

    extractor = featureextractor.RadiomicsFeatureExtractor(**extractor_params)

    features = []

    for sample in tqdm(dataset):
        data_dict = {
            "label": sample["label"],
            "case_id": sample["case_id"],
            "patient_id": sample["patient_id"],
        }
        try:
            baseline_mask = sitk.ReadImage(sample["baseline_seg"])
            followup_mask = sitk.ReadImage(sample["followup_seg"])

            for mod in ["T1", "T1CE", "T2", "FLAIR"]:
                baseline_image = sitk.ReadImage(sample[f"baseline_{mod}"])
                followup_image = sitk.ReadImage(sample[f"followup_{mod}"])

                baseline_features = extractor.execute(baseline_image, baseline_mask)
                followup_features = extractor.execute(followup_image, followup_mask)

                baseline_dict = {
                    f"baseline_{mod}_{key}": value
                    for key, value in baseline_features.items()
                }
                followup_dict = {
                    f"followup_{mod}_{key}": value
                    for key, value in followup_features.items()
                }

                data_dict.update(baseline_dict)
                data_dict.update(followup_dict)

            features.append(data_dict)

        except Exception as e:
            print(
                f"Failed for patient {data_dict['patient_id']}, case {data_dict['case_id']}: {e}"
            )

    features_df = pd.DataFrame(features)
    features_df.to_csv(data_path / file_name, index=False)
    print(f"Successfully saved to {(data_path / file_name).as_posix()}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI radiomics features extracting script.")
    parser.add_argument("--data", type=str, default=".", help="Path to data directory")
    parser.add_argument("--filename", type=str, default=".", help="Name for output csv file")

    args = parser.parse_args()
    
    DATA_PATH = Path(args.data)
    FILE_NAME = args.filename
    extract_features(data_path=DATA_PATH, file_name=FILE_NAME)
