"""全局配置 — 壁画线稿提取"""

from pathlib import Path

ROOT = Path(__file__).parent

DATASET_DIR = ROOT / "DhMurals-inpainting-dataset"
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

# ── 图像参数 ──
IMAGE_SIZE      = 512          # 训练/推理统一尺寸
TRAIN_IN_CH     = 4            # 输入: RGB 壁画(3ch) + 损伤Mask(1ch)
TRAIN_OUT_CH    = 1            # 输出: 灰度线稿

# ── 模型参数 ──
STAGE1_BASE_CH  = 64           # U-Net base channels
NUM_DOWNS       = 5            # 下采样次数

# ── 损失权重（端到端：破损壁画 → 完整线稿）──
LAMBDA_L1         = 20.0      # 像素忠实度
LAMBDA_EDGE       = 8.0       # 梯度约为 L1 的 1.5 倍
LAMBDA_SSIM       = 4.0       # 与 Edge/拓扑高度相关，作为辅助项
LAMBDA_SKEL       = 3.0       # soft-clDice 梯度较强，避免主导训练
LAMBDA_DIR        = 6.0       # 仅在线条边缘生效，保持较低梯度占比
LAMBDA_ADV        = 0.05      # 对抗项只做低权重细化
LAMBDA_TVERSKY    = 0.90      # 仅在破损区平衡漏线与伪线；当前候选参数略偏召回
TVERSKY_ALPHA     = 0.40      # 假阳性（伪线）惩罚
TVERSKY_BETA      = 0.60      # 假阴性（漏线）惩罚，略高于伪线
TVERSKY_TEMPERATURE = 0.1     # 将灰度输出平滑转换为线条概率

# 重建区域权重：重点提高破损区真实线条的召回率，而非放大整片背景
VALID_RECON_WEIGHT     = 1.0
HOLE_BACKGROUND_WEIGHT = 1.0
HOLE_LINE_WEIGHT       = 4.0
LINE_FOCUS_KERNEL      = 5     # 覆盖线条及其两侧 Sobel/SSIM 邻域

# ── DWA 动态权重平均 ──
DWA_ENABLED       = False     # 先完成梯度尺度校准，再评估是否需要动态微调
DWA_TEMP          = 0.5       # 温度系数，越小越敏感（权重调整幅度越大）
SKEL_START_EPOCH  = 10
GAN_START_EPOCH   = 30
RESUME_G_ONLY_GAN_WARMUP_EPOCHS = 5  # 从仅含生成器的 best 恢复时，先让 G 稳定几轮再开 GAN

# ── 训练参数 ──
BATCH_SIZE        = 6          # 4090D-24G
NUM_EPOCHS        = 200
LR_G              = 2e-4
LR_D              = 5e-5         # D 学习率更低，防过强
BETA1             = 0.5
BETA2             = 0.999
LR_DECAY_EPOCH    = 100
EARLY_STOP_PATIENCE = 100    # 增加 patience，防止在 GAN 阶段因 L1 loss 波动而早停
EARLY_STOP_MIN_DELTA = 0.001

# ── 硬件 ──
DEVICE = "cuda"
NUM_WORKERS = 4
