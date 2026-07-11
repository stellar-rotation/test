"""
Y型双分支：条件GAN (cGAN) + LSGAN + FP32隔离
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import math
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import config
from models.unified_net import UnifiedGenerator
from models.modules import MultiScaleDiscriminator
from losses.gradient_loss import GradientLoss
from losses.cldice_loss import CLDiceLoss
from dataset import Stage1Dataset


def set_seed(seed=42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = Stage1Dataset(str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES),
                             image_size=config.IMAGE_SIZE, augment=True,
                             real_mask_dir=str(config.REAL_MASK_DIR),
                             real_mask_ratio=config.REAL_MASK_RATIO)
    val_ds = Stage1Dataset(str(config.VAL_IMAGES), str(config.VAL_EDGES),
                           image_size=config.IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True,
                            persistent_workers=True)

    G = UnifiedGenerator(in_ch=4, base_ch=config.S1_BASE_CH,
                         num_downs=config.S1_NUM_DOWNS, num_res=8).to(device)
    # 多尺度条件判别器：in_ch=5 = img(3)+mask(1)+pred(1)
    D = MultiScaleDiscriminator(in_ch=5, num_scales=config.NUM_D_SCALES).to(device)

    criterion_edge = GradientLoss().to(device)
    criterion_cldice = CLDiceLoss().to(device)
    criterion_l1 = nn.L1Loss(reduction='none')

    # EMA: 推理时比raw G更稳定
    ema_G_state = {k: v.clone() for k, v in G.state_dict().items()}
    ema_decay = config.EMA_DECAY

    opt_G = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=config.LR_D,
                       betas=(config.BETA1, config.BETA2), weight_decay=1e-4)
    warmup_G = optim.lr_scheduler.LinearLR(opt_G, start_factor=0.1, total_iters=config.WARMUP_EPOCHS)
    warmup_D = optim.lr_scheduler.LinearLR(opt_D, start_factor=0.1, total_iters=config.WARMUP_EPOCHS)
    cosine_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=config.NUM_EPOCHS-config.WARMUP_EPOCHS, eta_min=1e-6)
    cosine_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=config.NUM_EPOCHS-config.WARMUP_EPOCHS, eta_min=1e-6)
    scheduler_G = optim.lr_scheduler.SequentialLR(opt_G, schedulers=[warmup_G, cosine_G], milestones=[config.WARMUP_EPOCHS])
    scheduler_D = optim.lr_scheduler.SequentialLR(opt_D, schedulers=[warmup_D, cosine_D], milestones=[config.WARMUP_EPOCHS])
    scaler = GradScaler()
    best_val = float("inf"); patience = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train(); D.train()

        # Dynamic loss weight: 前期靠结构监督，后期GAN/FM接管
        progress = (epoch - 1) / max(config.NUM_EPOCHS - 1, 1)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        grad_weight_dyn = config.B_GRAD_WEIGHT_MIN + (config.B_GRAD_WEIGHT - config.B_GRAD_WEIGHT_MIN) * cos_factor

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            img = batch["img"].to(device); mask = batch["mask"].to(device)
            target_perfect = batch["perfect_edge"].to(device)

            with autocast():
                pred_A, pred_B, feats_A, feats_B = G(img, mask, return_decoder_features=True)
                valid = 1.0 - mask

            # 条件张量 [B, 4, H, W] = img(3) + mask(1)
            cond = torch.cat([img, mask], dim=1).float()

            # ── D: 多尺度条件LSGAN ──
            opt_D.zero_grad()
            real_pair = torch.cat([cond, target_perfect.float()], dim=1)
            fake_pair = torch.cat([cond, pred_B.detach().float()], dim=1)
            d_real_list = D(real_pair, return_features=True)
            d_fake_list = D(fake_pair, return_features=True)
            loss_D = sum(
                0.5 * ((dr[0] - 0.9) ** 2).mean() + 0.5 * (df[0] ** 2).mean()
                for dr, df in zip(d_real_list, d_fake_list)
            ) / config.NUM_D_SCALES
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), config.GRAD_CLIP_NORM)
            opt_D.step()

            # ── G ──
            with autocast():
                mse_A = F.mse_loss(pred_A * valid, target_perfect * valid, reduction='sum') / (valid.sum() + 1e-6)
                grad_A = criterion_edge(pred_A, target_perfect, mask=valid)
                cldice_A = criterion_cldice(pred_A, target_perfect)
                loss_A = config.LAMBDA_MSE * mse_A + grad_weight_dyn * grad_A + config.B_CLDICE_WEIGHT * cldice_A

                l1_all = criterion_l1(pred_B, target_perfect)
                l1_hole = (l1_all * mask).sum() / (mask.sum() + 1e-6)
                l1_valid = (l1_all * valid).sum() / (valid.sum() + 1e-6)
                loss_l1_B = l1_valid + config.B_L1_HOLE_MUL * l1_hole

                loss_gray = (pred_B * (1.0 - pred_B) * mask).sum() / (mask.sum() + 1e-6)
                grad_B = criterion_edge(pred_B, target_perfect, mask=valid)
                cldice_B = criterion_cldice(pred_B, target_perfect)
                loss_cons = (F.l1_loss(pred_A, pred_B, reduction='none') * valid).sum() / (valid.sum() + 1e-6)
                loss_feat_cons = sum(F.l1_loss(fa, fb)
                                     for fa, fb in zip(feats_A[-2:], feats_B[-2:])) / 2

            # 多尺度 cGAN + FM (FP32)
            fake_pair_G = torch.cat([cond, pred_B.float()], dim=1)
            d_fake_G_list = D(fake_pair_G, return_features=True)
            loss_adv_B = sum(
                ((df[0] - 1.0) ** 2).mean() for df in d_fake_G_list
            ) / config.NUM_D_SCALES
            loss_fm_B = 0.0
            for (_, dr_feats), (_, df_feats) in zip(d_real_list, d_fake_G_list):
                scale_fm = sum(F.l1_loss(df, dr.detach())
                               for df, dr in zip(df_feats, dr_feats)) / len(df_feats)
                loss_fm_B += scale_fm
            loss_fm_B /= config.NUM_D_SCALES

            loss_G = loss_A + (config.B_L1_WEIGHT * loss_l1_B +
                                config.B_GRAY_WEIGHT * loss_gray +
                                grad_weight_dyn * grad_B +
                                config.B_ADV_WEIGHT * loss_adv_B +
                                config.B_FM_WEIGHT * loss_fm_B +
                                config.B_CONS_WEIGHT * loss_cons +
                                config.B_FEAT_CONS_WEIGHT * loss_feat_cons +
                                config.B_CLDICE_WEIGHT * cldice_B)

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(opt_G)
            scaler.update()

            # EMA 原地更新（no_grad + copy_ 防显存泄漏与计算图残留）
            with torch.no_grad():
                for k, v in G.state_dict().items():
                    ema_G_state[k].copy_(ema_decay * ema_G_state[k] + (1 - ema_decay) * v)

            pbar.set_postfix(G=f"{loss_G.item():.1f}", D=f"{loss_D.item():.2f}",
                             H=f"{l1_hole.item():.4f}", Cl=f"{cldice_B.item():.4f}")

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
            "ema_G": ema_G_state,
            "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
            "scheduler_G": scheduler_G.state_dict(), "scheduler_D": scheduler_D.state_dict(),
            "scaler": scaler.state_dict(),
        }
        if val_loss < best_val - 0.0001:
            best_val = val_loss; patience = 0
            torch.save(checkpoint_state, config.CHECKPOINT_DIR / "unified_best.pt")
            torch.save({"model_G": ema_G_state}, config.CHECKPOINT_DIR / "unified_ema.pt")
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
