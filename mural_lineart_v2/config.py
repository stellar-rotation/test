"""全局配置 �?Y型双分支壁画线条重建"""

from pathlib import Path

ROOT = Path(__file__).parent

DATASET_DIR = ROOT.parent / "mural_lineart" / "DhMurals-inpainting-dataset"
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

# ── 模型参数 ──
S1_IN_CH   = 4
S1_OUT_CH  = 1
S1_BASE_CH = 64
S1_NUM_DOWNS = 5
UNIFIED_NUM_RES = 8

# ── 损失权重 ──
LAMBDA_MSE  = 100.0
LAMBDA_EDGE = 30.0

# 分支 B 损失权重
LAMBDA_EDGE_B  = 20.0   # 梯度损失（逼迫线条变细变锐利）
L1_HOLE_WEIGHT = 1.5    # 破损区 L1（降权释放 GAN 脑补能力）
LAMBDA_SKEL_B  = 0.1    # 骨架损失（降权防拓扑捷径）
LAMBDA_ADV_B   = 0.5    # 对抗损失（提权鼓励大胆生成）

# ── 训练参数 ──
BATCH_SIZE    = 6
NUM_EPOCHS    = 200
LR_G          = 2e-4
LR_D          = 5e-5
D2G_LR        = 0.1
BETA1         = 0.5
BETA2         = 0.999
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.001

DEVICE = "cuda"
NUM_WORKERS = 4
