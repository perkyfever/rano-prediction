import os
import sys

import json
import argparse

import torch
from torch import autocast
from torch.utils.data import DataLoader

from tqdm import tqdm
import monai.transforms as mt

from mri.utils import RANDOM_SEED
from mri.dataset import ReorganizeTransform, CropAround3DMaskd
from mri.models import BMMAE, load_run_checkpoint
from mri.dataset import get_dataset


def get_transform() -> mt.Compose:
    baseline_image_keys = ["baseline_FLAIR", "baseline_T1", "baseline_T1CE", "baseline_T2", "baseline_seg"]
    followup_image_keys = ["followup_FLAIR", "followup_T1", "followup_T1CE", "followup_T2", "followup_seg"]
    keys = baseline_image_keys + followup_image_keys

    required_keys = ["T1", "T1CE", "T2", "FLAIR", "seg"]
    baseline_keys = [f"baseline_{key}" for key in required_keys]
    followup_keys = [f"followup_{key}" for key in required_keys]
    all_image_keys = baseline_keys + followup_keys
    
    transform = mt.Compose([
        mt.LoadImaged(keys=all_image_keys),
        mt.EnsureChannelFirstd(keys=keys),
        mt.EnsureTyped(keys=all_image_keys),
        mt.Transposed(keys=all_image_keys, indices=(0, 2, 1, 3)),
        mt.Spacingd(keys=[key for key in keys if "seg" in key], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        mt.Spacingd(keys=[key for key in keys if "seg" not in key], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        CropAround3DMaskd(keys=baseline_image_keys, mask_key="baseline_seg"),
        CropAround3DMaskd(keys=followup_image_keys, mask_key="followup_seg"),
        mt.ResizeD(keys=all_image_keys, spatial_size=(128, 128, 128)),
        ReorganizeTransform(required_keys=required_keys)
    ]).set_random_state(seed=RANDOM_SEED)
    
    return transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--chkp_path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data folder.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the output results.")

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Placeholder for inference logic
    print(f"Running inference with model: {args.model_path}")
    print(f"Using input data from: {args.input_data}")
    print(f"Results will be saved in: {args.output_dir}")

    # Here you would load the model and perform inference
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BMMAE()
    best_chkp = load_run_checkpoint(args.chkp_path)[-1] # make it accept filepath actually
    model.load_state_dict(best_chkp["model"])

    transform = get_transform()
    dataset = get_dataset(data_path=args.data_path, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    
    
    predictions = {}
    model = model.to(device)
    model.eval()
    
    for batched_sample in tqdm(dataloader):
        sample = batched_sample[0]
        patient_id = sample["patient_id"]
        case_id = sample["case_id"]
        
        with torch.no_grad():
            with autocast(device_type="cuda"):
                proba = model(batched_sample)[0].cpu().numpy()
            
        if patient_id not in predictions:
            predictions[patient_id] = {}

        predictions[patient_id][case_id] = {"response": proba.tolist()}

    # Save predictions to a JSON file
    output_file = os.path.join(args.output_path, "predictions.json")
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
    