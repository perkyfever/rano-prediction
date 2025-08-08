import os
import sys
 
from pathlib import Path
 
BMMAE_PATH = Path(__file__).parent.parent / "BM-MAE"
sys.path.append(BMMAE_PATH.as_posix())
 
import json
import argparse
 
import torch
import torch.nn as nn
 
from bmmae.model import ViTEncoder
from bmmae.tokenizers import MRITokenizer
 
from torch import autocast
from torch.utils.data import DataLoader
 
from tqdm import tqdm
 
import monai.transforms as mt
from monai.data import Dataset as MDataset
 
 
###################### Constants #########################
 
RANDOM_SEED = 0xBAD5EED
 
###################### Architecture ######################
 
class BMMAE(nn.Module):
    def __init__(
        self,
        encoder_dropout: float = 0.10,
        dropout: float = 0.50,
        num_classes: int = 4,
        logits: bool = False,
        image_keys: list[str] = ["T1", "T1CE", "T2", "FLAIR"],
        unfreeze_layers: list[str] = ["blocks.9", "blocks.10", "blocks.11"]
    ):
        super(BMMAE, self).__init__()
        self.cls_token_idx = 0
 
        # SETTING UP ENCODER
        tokenizers = {
            modality: MRITokenizer(
                patch_size=(16, 16, 16),
                img_size=(128, 128, 128),
                hidden_size=768,
            )
            for modality in image_keys
        }
        self.encoder = ViTEncoder(
            modalities=image_keys,
            tokenizers=tokenizers,
            cls_token=True,
            dropout_rate=encoder_dropout
        )
 
        # LOADING WEIGHTS
        state_dict = torch.load(
            BMMAE_PATH / "pretrained_models" / "bmmae.pth",
            map_location=torch.device("cpu")
        )
        self.encoder.load_state_dict(state_dict, strict=False)
 
        for name, param in self.encoder.named_parameters():
            if all(x not in name for x in unfreeze_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
 
        self.delta = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, self.encoder.hidden_size),
            nn.LayerNorm(self.encoder.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
 
        clf_layers = [
            nn.Linear(3 * self.encoder.hidden_size, self.encoder.hidden_size),
            nn.LayerNorm(self.encoder.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, num_classes),
        ]
 
        if not logits:
            clf_layers.append(nn.Softmax(dim=-1))
 
        self.clf_head = nn.Sequential(*clf_layers)
 
    def forward(self, baseline, followup):
        baseline_embed = self.encoder(baseline)[:, self.cls_token_idx]
        followup_embed = self.encoder(followup)[:, self.cls_token_idx]
 
        delta_embed = self.delta(followup_embed - baseline_embed)
        joint_embed = torch.cat([baseline_embed, followup_embed, delta_embed], dim=1)
 
        outputs = self.clf_head(joint_embed)
 
        return outputs
 
###################### Transforms ########################
 
class ReorganizeTransform(mt.MapTransform):
    def __init__(self, required_keys=("T1", "T1CE", "T2", "FLAIR", "seg")):
        """
        Partitions baseline and followup images.
        :param required_keys: image keys to partition
        """
        super().__init__(keys=None)
        self.required_keys = required_keys
 
    def __call__(self, data):
        if isinstance(data, list):
            data = data[0]
        baseline, followup = {}, {}
        for key in self.required_keys:
            baseline_key = f"baseline_{key}"
            followup_key = f"followup_{key}"
            assert baseline_key in data
            assert followup_key in data
            baseline[key] = data[baseline_key]
            followup[key] = data[followup_key]
            del data[baseline_key]
            del data[followup_key]
 
        data.update({"baseline": baseline, "followup": followup})
 
        return data
 
class CropAround3DMaskd(mt.MapTransform):
    def __init__(self, keys, mask_key, margin=10):
        super().__init__(keys)
        self.mask_key = mask_key
        self.margin = margin
 
    def __call__(self, data):
        d = dict(data)
        # mask = d[self.mask_key][0]  # assuming mask shape (C=1, H, W, D)
        # # Find bounding box of non-zero voxels
        # nonzero = np.nonzero(mask)
 
        # if len(nonzero[0]) == 0:
        #     # Mask is empty: skip cropping and return original images
        #     return d
 
        # minz, maxz = nonzero[0].min(), nonzero[0].max()
        # miny, maxy = nonzero[1].min(), nonzero[1].max()
        # minx, maxx = nonzero[2].min(), nonzero[2].max()
 
        # Expand bounding box by margin, ensuring within image bounds
        # shape = mask.shape
        # minz = max(minz - self.margin, 0)
        # maxz = min(maxz + self.margin + 1, shape[0])
        # miny = max(miny - self.margin, 0)
        # maxy = min(maxy + self.margin + 1, shape[1])
        # minx = max(minx - self.margin, 0)
        # maxx = min(maxx + self.margin + 1, shape[2])
        minz = 15
        maxz = 200
        miny = 45
        maxy = 135
        minx = 45
        maxx = 135
 
        # Crop all keys accordingly
        for key in self.keys:
            img = d[key]
            # img shape assumed (C, H, W, D)
            d[key] = img[:, minz:maxz, miny:maxy, minx:maxx]
 
        return d
 
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
        mt.Spacingd(keys=[image_key for image_key in all_image_keys if "seg" in image_key], pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        mt.Spacingd(keys=[image_key for image_key in all_image_keys if "seg" not in image_key], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        CropAround3DMaskd(keys=baseline_image_keys, mask_key="baseline_seg"),
        CropAround3DMaskd(keys=followup_image_keys, mask_key="followup_seg"),
        mt.ResizeD(keys=all_image_keys, spatial_size=(128, 128, 128)),
        ReorganizeTransform(required_keys=required_keys)
    ]).set_random_state(seed=RANDOM_SEED)
 
    return transform
 
###################### Utilities ######################
 
def load_run_checkpoint(chkp_path: str | Path):
    """
    Loads run checkpoint.
    :param run_name: str
    :return: best_model
    """
    checkpoint_path = Path(chkp_path)
 
    best_model_checkpoint = None
    best_model_path = checkpoint_path / "best_model.pth"
    if os.path.exists(best_model_path):
        best_model_checkpoint = torch.load(
            best_model_path,
            weights_only=False,
            map_location=torch.device("cpu")
        )
 
    return best_model_checkpoint
 
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
    data_path = Path(data_path)
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
 
###################### Inference ######################
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--chkp_path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data folder.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the output results.")
 
    args = parser.parse_args()
 
    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)
 
    # Placeholder for inference logic
    print(f"Running inference with model: {args.chkp_path}")
    print(f"Using input data from: {args.data_path}")
    print(f"Results will be saved in: {args.output_path}")

    # Inference itself
 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = BMMAE()
    best_chkp = load_run_checkpoint(args.chkp_path)
    model.load_state_dict(best_chkp["model"])
 
    transform = get_transform()
    dataset = get_dataset(data_path=args.data_path, transform=transform)[0]
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4
    # )
 
    predictions = {}
    model = model.to(device).float()
    model.eval()
 
    for sample in tqdm(dataset):        
        patient_id = sample["patient_id"]
        case_id = sample["case_id"]

        batched_sample = sample.copy()
        for key in batched_sample["baseline"]:
            batched_sample["baseline"][key] = batched_sample["baseline"][key].unsqueeze(0).to(device).float()
            batched_sample["followup"][key] = batched_sample["followup"][key].unsqueeze(0).to(device).float()

        with torch.no_grad():
            proba = model(batched_sample["baseline"], batched_sample["followup"])[0].cpu().numpy()
 
        if patient_id not in predictions:
            predictions[patient_id] = {}
 
        predictions[patient_id][case_id] = {"response": proba.tolist()}
 
    # Save predictions to a JSON file
    output_file = Path(args.output_path) / "predictions.json"
    with open(output_file, "w") as f:
        json.dump(predictions, f, indent=4)
