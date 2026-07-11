"""
Y型双分支：FP32 LSGAN 稳定对抗版
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
from models.modules import Discriminator
from losses.gradient_loss import GradientLoss
from losses.skeleton_loss import SkeletonLoss
from dataset import Stage1Dataset


def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = True


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
    D = Discriminator(in_ch=1).to(device)

    criterion_edge = GradientLoss().to(device)
    criterion_skel = SkeletonLoss(input_range="sigmoid").to(device)
    criterion_l1 = nn.L1Loss(reduction='none')

    opt_G = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=config.LR_G * 0.1,
                       betas=(config.BETA1, config.BETA2), weight_decay=1e-4)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=config.NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    best_val = float("inf"); patience = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            img = batch["img"].to(device); mask = batch["mask"].to(device)
            target_perfect = batch["perfect_edge"].to(device)

            # ── G 前向 ──
            with autocast():
                pred_A, pred_B = G(img, mask)
                valid = 1.0 - mask

            # ── D 更新 (FP32 隔离，LSGAN) ──
            opt_D.zero_grad()
            d_real, d_real_feats = D(target_perfect.float(), return_features=True)
            d_fake, _ = D(pred_B.detach().float(), return_features=True)
            loss_D = 0.5 * ((d_real - 1.0) ** 2).mean() + 0.5 * (d_fake ** 2).mean()
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 10.0)
            opt_D.step()

            # ── G 更新 ──
            with autocast():
                # 分支 A：纯提取
                mse_A = F.mse_loss(pred_A * valid, target_perfect * valid, reduction='sum') / (valid.sum() + 1e-6)
                grad_A = criterion_edge(pred_A, target_perfect, mask=valid)
                loss_A = config.LAMBDA_MSE * mse_A + config.LAMBDA_EDGE * grad_A

                # 分支 B：数学约束
                l1_all = criterion_l1(pred_B, target_perfect)
                l1_hole = (l1_all * mask).sum() / (mask.sum() + 1e-6)
                l1_valid = (l1_all * valid).sum() / (valid.sum() + 1e-6)
                loss_l1_B = l1_valid + 6.0 * l1_hole
                loss_skel_B = criterion_skel(pred_B, target_perfect)
                loss_gray = (pred_B * (1.0 - pred_B) * mask).sum() / (mask.sum() + 1e-6)
                # 破损区不碰 GradLoss，交给 GAN
                grad_B = criterion_edge(pred_B, target_perfect, mask=valid)

            # LSGAN + FM (FP32)
            d_fake_G, d_fake_feats = D(pred_B.float(), return_features=True)
            loss_adv_B = ((d_fake_G - 1.0) ** 2).mean()
            loss_fm_B = sum(F.l1_loss(df, dr.detach())
                             for df, dr in zip(d_fake_feats, d_real_feats)) / len(d_fake_feats)

            loss_G = loss_A + (10.0 * loss_l1_B + 2.0 * loss_skel_B +
                                10.0 * loss_gray + 20.0 * grad_B +
                                0.5 * loss_adv_B + 10.0 * loss_fm_B)

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 10.0)
            scaler.step(opt_G)
            scaler.update()

            pbar.set_postfix(G=f"{loss_G.item():.1f}", D=f"{loss_D.item():.2f}",
                             H=f"{l1_hole.item():.4f}", Sk=f"{loss_skel_B.item():.4f}")

        # ── Val (只测破损区) ──
        G.eval(); val_loss = 0.0; nan_count = 0
        with torch.no_grad(), autocast():
            for batch in val_loader:
                img = batch["img"].to(device); mask = batch["mask"].to(device)
                target = batch["perfect_edge"].to(device)
                _, pred = G(img, mask)
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    nan_count += 1
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
                diff = F.l1_loss(pred, target, reduction='none')
                val_loss += (diff * mask).sum().item() / (mask.sum() + 1e-6)
        val_loss /= len(val_loader)
        nan_msg = f" (NaN:{nan_count})" if nan_count > 0 else ""
        print(f"  Val Hole L1: {val_loss:.5f}{nan_msg}")
        scheduler_G.step(); scheduler_D.step()

        checkpoint_state = {
            "epoch": epoch, "val_loss": val_loss,
            "model_G": G.state_dict(), "model_D": D.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "scheduler_G": scheduler_G.state_dict(), "scheduler_D": scheduler_D.state_dict(),
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
