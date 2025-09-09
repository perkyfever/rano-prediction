<div align="center">
<h1> Predicting Brain Tumor Response to Therapy using a Hybrid Deep Learning and Radiomics Approach </h1>
</div> 

Accurate evaluation of the response of glioblastoma to therapy is crucial for clinical decision-making and patient management. The
Response Assessment in Neuro-Oncology (RANO) criteria provide a standardized framework to assess patients’ clinical response, but their application can be complex and subject to observer variability. This paper
presents an automated method for classifying the intervention response
from longitudinal MRI scans, developed to predict tumor response during
therapy as part of the BraTS 2025 challenge. We propose a novel hybrid
framework that combines deep learning derived feature extraction and an
extensive set of radiomics and clinically-chosen features. Our approach
utilizes fine-tuned ResNet-18 model to extract features from 2D regions
of interest across four MRI modalities. These deep features are then fused
with a rich set of more than 4,800 radiomic and clinically-driven features,
including 3D radiomics of tumor growth and shrinkage masks, volumetric
changes relative to the nadir, and tumor centroid shift. Using the fused
feature set, a CatBoost classifier achieves a mean ROC AUC of 0.81 and
a Macro F1 score of 0.50 in the 4-class response prediction task (Complete Response, Partial Response, Stable Disease, Progressive Disease).
Our results highlight that synergizing learned image representations with
domain-targeted radiomic features provides a robust and effective solution for automated treatment response assessment in neuro-oncology.


## Running Scripts

### Preprocessing

```python
from pathlib import Path

# Setup paths
USERNAME = "username"
WORK_DIR = Path("/home") / USERNAME
DATA_DIR = WORK_DIR / "mri" / "data" / "Lumiere"

# Atlas for registration (requires FSL)
ATLAS_NAME = "MNI152_T1_1mm_brain.nii.gz"
ATLAS_PATH = WORK_DIR / "fsl" / "data" / "standard" / ATLAS_NAME

# Output directory
SAVE_DIR = "preprocessed_data"
SAVE_TO_DIR = WORK_DIR / "mri" / "data" / SAVE_DIR

# Run preprocessing
!python3 preprocessing.py \
    --data={DATA_DIR.as_posix()} \
    --atlas={ATLAS_PATH.as_posix()} \
    --saveto={SAVE_TO_DIR.as_posix()}
```

### Radiomics Feature Extraction

```python
from pathlib import Path

# Setup paths
USERNAME = "username"
WORK_DIR = Path("/home") / USERNAME
DATA_DIR = WORK_DIR / "mri" / "data" / "preprocessed_data"
OUT_NAME = "radiomics.csv"

# Run feature extraction
!python3 radiomics/radiomics_features.py \
    --data={DATA_DIR.as_posix()} \
    --filename=OUT_NAME
```

### Hyperparameter Tuning with Optuna

```python
from pathlib import Path

# Setup paths
USERNAME = "username"
WORK_DIR = Path("/home") / USERNAME
DATA_DIR = WORK_DIR / "mri" / "data" / "preprocessed_data"
DATA_PATH = DATA_DIR / "radiomics.csv"

SPLIT_PATH = WORK_DIR / "mri" / "dataset" / "split.json"
SAVE_TO_DIR = WORK_DIR / "mri" / "models" / "saved_runs"

# Optuna parameters
NUM_TRIALS = 300
NUM_STARTUP_TRIALS = int(0.10 * NUM_TRIALS)
ITERATIONS = 100
EARLY_STOPPING_ROUNDS = 20
EXP_NAME = "test_run"

# Run Optuna tuning
!python3 radiomics/radiomics_optuna.py \
    --data={DATA_PATH.as_posix()} \
    --split={SPLIT_PATH.as_posix()} \
    --saveto={SAVE_TO_DIR.as_posix()} \
    --num_trials={NUM_TRIALS} \
    --num_startup_trials={NUM_STARTUP_TRIALS} \
    --iterations={ITERATIONS} \
    --early_stopping_rounds={EARLY_STOPPING_ROUNDS} \
    --exp_name={EXP_NAME}
```

### Training Deep Vision Model

```python
raise NotImplementedError
```

