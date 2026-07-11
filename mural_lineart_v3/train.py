"""
Y型双分支联合训练：纯数学重建版 (无GAN，绝对稳定)
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import config
from models.unified_net import UnifiedGenerator
from losses.gradient_loss import GradientLoss
from losses.skeleton_loss import SkeletonLoss
from dataset import Stage1Dataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = Stage1Dataset(str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES),
                             image_size=config.IMAGE_SIZE, augment=True)
    val_ds = Stage1Dataset(str(config.VAL_IMAGES), str(config.VAL_EDGES),
                           image_size=config.IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    G = UnifiedGenerator(in_ch=4, base_ch=config.S1_BASE_CH,
                         num_downs=config.S1_NUM_DOWNS, num_res=8).to(device)

    criterion_edge = GradientLoss().to(device)
    criterion_skel = SkeletonLoss(input_range="sigmoid").to(device)
    criterion_l1 = nn.L1Loss(reduction='none')

    opt_G = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    best_val = float("inf")
    patience = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            img = batch["img"].to(device)
            mask = batch["mask"].to(device)
            target_broken = batch["broken_edge"].to(device)
            target_perfect = batch["perfect_edge"].to(device)

            with autocast():
                pred_A, pred_B = G(img, mask)
                valid = 1.0 - mask

                # 分支 A：纯提取
                mse_A = F.mse_loss(pred_A * valid, target_broken * valid, reduction='sum') / (valid.sum() + 1e-6)
                grad_A = criterion_edge(pred_A, target_perfect, mask=valid)
                loss_A = config.LAMBDA_MSE * mse_A + config.LAMBDA_EDGE * grad_A

                # 分支 B：纯数学约束修复
                l1_all = criterion_l1(pred_B, target_perfect)
                l1_hole = (l1_all * mask).sum() / (mask.sum() + 1e-6)
                l1_valid = (l1_all * valid).sum() / (valid.sum() + 1e-6)
                loss_l1_B = l1_valid + 6.0 * l1_hole

                loss_skel_B = criterion_skel(pred_B, target_perfect)
                loss_gray = (pred_B * (1.0 - pred_B) * mask).sum() / (mask.sum() + 1e-6)
                grad_B = criterion_edge(pred_B, target_perfect)

                loss_B = (10.0 * loss_l1_B + 2.0 * loss_skel_B +
                          10.0 * loss_gray + 20.0 * grad_B)
                loss_G = loss_A + loss_B

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            pbar.set_postfix(G=f"{loss_G.item():.1f}", H=f"{l1_hole.item():.4f}",
                             Sk=f"{loss_skel_B.item():.4f}")

        G.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast():
            for batch in val_loader:
                img = batch["img"].to(device)
                mask = batch["mask"].to(device)
                target = batch["perfect_edge"].to(device)
                _, pred = G(img, mask)
                val_loss += F.l1_loss(pred, target).item()
        val_loss /= len(val_loader)
        print(f"  Val L1: {val_loss:.5f}")
        scheduler_G.step()

        checkpoint_state = {
            "epoch": epoch, "val_loss": val_loss,
            "model_G": G.state_dict(),
            "opt_G": opt_G.state_dict(),
            "scheduler_G": scheduler_G.state_dict(),
            "scaler": scaler.state_dict(),
        }

        if val_loss < best_val - 0.0001:
            best_val = val_loss; patience = 0
            torch.save(checkpoint_state, config.CHECKPOINT_DIR / "unified_best.pt")
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch}, best: {best_val:.5f}")
                break
        if epoch % 20 == 0:
            torch.save(checkpoint_state, config.CHECKPOINT_DIR / f"unified_e{epoch}.pt")

    print(f"Done. Best: {best_val:.5f}")


if __name__ == "__main__":
    train()
