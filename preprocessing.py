import os
import json

import argparse
import SimpleITK as sitk

from tqdm import tqdm
from pathlib import Path


def load_patients(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def get_transform(fixed_image, moving_t1_image):
    """
    Calculates the transformation from a (T1) image to the MNI atlas template.
    :param fixed_image: The MNI atlas template image.
    :param moving_t1_image: The patient's T1 scan image.
    :return: The final transformation object.
    """
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.10)

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=0.10,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_t1_image,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(
        sitk.Cast(fixed_image, sitk.sitkFloat32),
        sitk.Cast(moving_t1_image, sitk.sitkFloat32),
    )

    return final_transform


def resample_image(fixed_image, moving_image, transform, is_segmentation=False):
    """
    Registers a moving image to the fixed MNI atlas template using the provided transformation.
    :param fixed_image: The MNI atlas template image.
    :param moving_image: The patient's MRI scan image to be registered.
    :param transform: The transformation object obtained from T1 registration.
    :param is_segmentation: Boolean indicating if the moving image is a segmentation mask.
    :return: The registered image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_segmentation else sitk.sitkLinear
    )
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(moving_image)


def preprocess_and_register_all_scans(patients: dict, data_path: Path, save_to: Path):
    """
    Iterates through all scans, registers them to the MNI atlas, and saves them.
    :param patients: Dictionary containing patient data.
    :param data_path: Path to the directory containing patient scans.
    :param save_to: Path to the directory where registered scans will be saved.
    :return: None
    """
    print("--- Starting Preprocessing: Registering all scans to MNI atlas ---")
    fixed_image = sitk.ReadImage(MNI_TEMPLATE_PATH, sitk.sitkFloat32)
    total_cases = sum(len(cases) for cases in patients.values())
    pbar = tqdm(total=total_cases, desc="Registering cases to atlas", unit="case")

    ATLAS_SCANS_PATH = save_to / "scans"
    ATLAS_SEGMS_PATH = save_to / "segmentations"

    new_patients_dict = {}

    for pid, cases in patients.items():
        patient_cases = []
        for cid, meta in cases.items():
            try:
                # --- Process BASELINE scans ---
                baseline_t1_atlas_path = (
                    ATLAS_SCANS_PATH / f"{pid}_{cid}_baseline_T1.nii.gz"
                )
                if not os.path.exists(baseline_t1_atlas_path):
                    baseline_t1_path = f"{data_path / meta['baseline_registered'].replace('./', '') / meta['baseline_registered'].replace('./images_registered/', '')}_0000.nii.gz"
                    moving_t1_baseline = sitk.ReadImage(
                        baseline_t1_path, sitk.sitkFloat32
                    )

                    # Register the T1 image to the atlas
                    transform = get_transform(fixed_image, moving_t1_baseline)
                    # Resample images to the atlas space
                    for mri_type_idx, mri_type_name in enumerate(
                        ["T1", "T1CE", "T2", "FLAIR"]
                    ):
                        image_path = f"{data_path / meta['baseline_registered'].replace('./', '') / meta['baseline_registered'].replace('./images_registered/', '')}_{mri_type_idx:04d}.nii.gz"
                        moving_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                        registered_image = resample_image(
                            fixed_image, moving_image, transform
                        )
                        output_path = (
                            ATLAS_SCANS_PATH
                            / f"{pid}_{cid}_baseline_{mri_type_name}.nii.gz"
                        )
                        sitk.WriteImage(registered_image, output_path)

                    seg_path = data_path / meta["baseline_seg_registered"].replace(
                        "./", ""
                    )
                    moving_seg = sitk.ReadImage(seg_path)
                    registered_seg = resample_image(
                        fixed_image, moving_seg, transform, is_segmentation=True
                    )
                    output_seg_path = (
                        ATLAS_SEGMS_PATH / f"{pid}_{cid}_baseline_seg.nii.gz"
                    )
                    sitk.WriteImage(registered_seg, output_seg_path)

                # --- Process FOLLOWUP scans ---
                followup_t1_atlas_path = (
                    ATLAS_SCANS_PATH / f"{pid}_{cid}_followup_T1.nii.gz"
                )
                if not os.path.exists(followup_t1_atlas_path):
                    followup_t1_path = f"{data_path / meta['followup_registered'].replace('./', '') / meta['followup_registered'].replace('./images_registered/', '')}_0000.nii.gz"
                    moving_t1_followup = sitk.ReadImage(
                        followup_t1_path, sitk.sitkFloat32
                    )

                    # Register the T1 image to the atlas
                    transform = get_transform(fixed_image, moving_t1_followup)
                    # Resample images to the atlas space
                    for mri_type_idx, mri_type_name in enumerate(
                        ["T1", "T1CE", "T2", "FLAIR"]
                    ):
                        image_path = f"{data_path / meta['followup_registered'].replace('./', '') / meta['followup_registered'].replace('./images_registered/', '')}_{mri_type_idx:04d}.nii.gz"
                        moving_image = sitk.ReadImage(image_path, sitk.sitkFloat32)
                        registered_image = resample_image(
                            fixed_image, moving_image, transform
                        )
                        output_path = (
                            ATLAS_SCANS_PATH
                            / f"{pid}_{cid}_followup_{mri_type_name}.nii.gz"
                        )
                        sitk.WriteImage(registered_image, output_path)

                    seg_path = data_path / meta["followup_seg_registered"].replace(
                        "./", ""
                    )
                    moving_seg = sitk.ReadImage(seg_path)
                    registered_seg = resample_image(
                        fixed_image, moving_seg, transform, is_segmentation=True
                    )
                    output_seg_path = (
                        ATLAS_SEGMS_PATH / f"{pid}_{cid}_followup_seg.nii.gz"
                    )
                    sitk.WriteImage(registered_seg, output_seg_path)

                    patient_cases += [
                        {
                            cid: {
                                "baseline_T1": f"./{pid}_{cid}_baseline_T1.nii.gz",
                                "baseline_T1CE": f"./{pid}_{cid}_baseline_T1CE.nii.gz",
                                "baseline_T2": f"./{pid}_{cid}_baseline_T2.nii.gz",
                                "baseline_FLAIR": f"./{pid}_{cid}_baseline_FLAIR.nii.gz",
                                "baseline_seg": f"./{pid}_{cid}_baseline_seg.nii.gz",
                                "followup_T1": f"./{pid}_{cid}_followup_T1.nii.gz",
                                "followup_T1CE": f"./{pid}_{cid}_followup_T1CE.nii.gz",
                                "followup_T2": f"./{pid}_{cid}_followup_T2.nii.gz",
                                "followup_FLAIR": f"./{pid}_{cid}_followup_FLAIR.nii.gz",
                                "followup_seg": f"./{pid}_{cid}_followup_seg.nii.gz",
                                "label": meta.get("response", -1),
                            }
                        }
                    ]
            except Exception as e:
                print(
                    f"ERROR processing case {pid}/{cid}. True Patient: {pid}. Error: {e}"
                )
            finally:
                pbar.update(1)

        if patient_cases:
            new_patients_dict[pid] = {
                case_id: case_meta
                for case in patient_cases
                for case_id, case_meta in case.items()
            }

    pbar.close()
    print(f"--- Preprocessing complete. All files saved to {save_to} directory. ---")
    return new_patients_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Atlas registration script.")
    parser.add_argument("--data", type=str, default=".", help="Path to data directory")
    parser.add_argument("--username", type=str, required=True, help="HPC username")
    parser.add_argument(
        "--saveto", type=str, default=".", help="Directory to save results"
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="MNI152_T1_1mm_brain.nii.gz",
        help="Atlas name to register",
    )

    args = parser.parse_args()

    MNI_TEMPLATE_PATH = (
        Path("/home") / args.username / "fsl" / "data" / "standard" / args.atlas
    )
    if not os.path.exists(MNI_TEMPLATE_PATH):
        raise FileNotFoundError(
            f"MNI Template not found at: {MNI_TEMPLATE_PATH}. Please update the path."
        )

    # DATA_PATH = Path(os.getcwd()) / "data" / "Lumiere"
    DATA_PATH = Path(args.data)
    JSON_PATH = DATA_PATH / "patients.json"

    SAVE_TO = Path(args.saveto)
    os.makedirs(SAVE_TO, exist_ok=True)
    ATLAS_SCANS_PATH = SAVE_TO / "scans"
    ATLAS_SEGMS_PATH = SAVE_TO / "segmentations"
    REGISTERED_JSON_PATH = SAVE_TO / "patients_registered.json"

    os.makedirs(ATLAS_SCANS_PATH, exist_ok=True)
    os.makedirs(ATLAS_SEGMS_PATH, exist_ok=True)

    patients = load_patients(JSON_PATH)
    patients_registered_dict = preprocess_and_register_all_scans(
        patients=patients, data_path=DATA_PATH, save_to=SAVE_TO
    )

    with open(REGISTERED_JSON_PATH, "w") as f:
        json.dump(patients_registered_dict, f, indent=4)
