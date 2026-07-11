"""Cloud config"""

from pathlib import Path

ROOT = Path(__file__).parent

DATASET_DIR = Path("/root/mural_lineart_v3/DhMurals-inpainting-dataset")
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_EDGES  = DATASET_DIR / "train" / "edges"
VAL_IMAGES   = DATASET_DIR / "val" / "images"
VAL_EDGES    = DATASET_DIR / "val" / "edges"
TEST_IMAGES  = DATASET_DIR / "test" / "images"
TEST_EDGES   = DATASET_DIR / "test" / "edges"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR     = ROOT / "output"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 512
S1_IN_CH=4;S1_OUT_CH=1;S1_BASE_CH=64;S1_NUM_DOWNS=5;UNIFIED_NUM_RES=8
LAMBDA_MSE=100.0;LAMBDA_EDGE=30.0
BATCH_SIZE=6;NUM_EPOCHS=200;LR_G=2e-4
BETA1=0.5;BETA2=0.999
EARLY_STOP_PATIENCE=15;EARLY_STOP_MIN_DELTA=0.001
DEVICE="cuda";NUM_WORKERS=4
