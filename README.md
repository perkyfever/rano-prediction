# 🧠 Predicting Brain Tumor Response to Therapy using a Hybrid Deep Learning and Radiomics Approach

**TODO**: Add a project description here.

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
!python3 radiomics_features.py \
    --data={DATA_DIR.as_posix()} \
    --filename=OUT_NAME
```

### Hyperparameter Tuning with Optuna

```python
from pathlib import Path

# Setup paths
USERNAME = "tihonovda"
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
!python3 radiomics_optuna.py \
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

