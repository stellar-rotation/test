"""全局配置 — 两阶段壁画线条重建"""

from pathlib import Path

ROOT = Path(__file__).parent

DATASET_DIR = ROOT / "DhMurals-inpainting-dataset"
TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_EDGES  = DATASET_DIR / "train" / "edges"
TRAIN_MASKS  = DATASET_DIR / "train" / "masks"
VAL_IMAGES   = DATASET_DIR / "val" / "images"
VAL_EDGES    = DATASET_DIR / "val" / "edges"
VAL_MASKS    = DATASET_DIR / "val" / "masks"
TEST_IMAGES  = DATASET_DIR / "test" / "images"
TEST_EDGES   = DATASET_DIR / "test" / "edges"
TEST_MASKS   = DATASET_DIR / "test" / "masks"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR     = ROOT / "output"
CHECKPOINT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 512

# ── 阶段一：线稿提取 ──
S1_IN_CH   = 3         # 破损壁画 RGB
S1_OUT_CH  = 1         # 线稿灰度
S1_BASE_CH = 64
S1_NUM_DOWNS = 5
LAMBDA_MSE  = 100.0
LAMBDA_EDGE = 30.0

# ── 阶段二：线稿修复 ──
S2_IN_CH   = 3         # 粗线稿 + Mask + 原图灰度
S2_OUT_CH  = 1
LAMBDA_L1_S2   = 1.0
LAMBDA_ADV_S2  = 0.1
LAMBDA_FM_S2   = 10.0

# ── 训练参数 ──
BATCH_SIZE    = 6
NUM_EPOCHS    = 200
LR_G          = 2e-4
LR_D          = 5e-5
BETA1         = 0.5
BETA2         = 0.999
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001

# ── 阶段一输出（供阶段二训练用）──
S1_OUTPUT_DIR = ROOT / "s1_train_outputs"

# ── 阶段二特有 ──
D2G_LR     = 0.1      # D lr = G lr × D2G_LR
GAN_LOSS   = "hinge"

DEVICE = "cuda"
NUM_WORKERS = 4
