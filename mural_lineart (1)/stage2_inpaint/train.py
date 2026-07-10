"""
EdgeConnect 风格线稿修复训练
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import config
from models.edge_generator import EdgeGenerator, Discriminator
from stage2_inpaint.data import Stage2Dataset


def hinge_d_loss(d_real, d_fake):
    return (F.relu(1.0 - d_real)).mean() + (F.relu(1.0 + d_fake)).mean()


def hinge_g_loss(d_fake):
    return -d_fake.mean()


import torch.nn.functional as F


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    s1_out = str(config.S1_OUTPUT_DIR) if config.S1_OUTPUT_DIR.exists() else None
    train_ds = Stage2Dataset(
        str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES), str(config.TRAIN_MASKS),
        s1_output_dir=s1_out, image_size=config.IMAGE_SIZE, augment=True,
    )
    val_ds = Stage2Dataset(
        str(config.VAL_IMAGES), str(config.VAL_EDGES), str(config.VAL_MASKS),
        s1_output_dir=s1_out, image_size=config.IMAGE_SIZE, augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # 模型
    G = EdgeGenerator(in_ch=config.S2_IN_CH, out_ch=config.S2_OUT_CH).to(device)
    D = Discriminator(in_ch=1).to(device)  # D 只判别线稿真假

    criterion_l1 = nn.L1Loss()

    opt_G = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=config.LR_G * config.D2G_LR, betas=(config.BETA1, config.BETA2))
    scaler = GradScaler()
    best_val = float("inf")
    patience_counter = 0
    PATIENCE = 15
    MIN_DELTA = 0.0001

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for inp, target in pbar:
            inp = inp.to(device); target = target.to(device)
            mask = inp[:, 1:2]  # mask 通道

            # ── D ──
            with autocast():
                fake = G(inp)
                d_real, d_real_feats = D(target, return_features=True)
                d_fake, _ = D(fake.detach(), return_features=True)
                loss_D = hinge_d_loss(d_real, d_fake)

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)
            scaler.update()

            # ── G ──
            with autocast():
                fake = G(inp)
                d_fake, d_fake_feats = D(fake, return_features=True)
                _, d_real_feats = D(target, return_features=True)

                # L1 加权：Mask 区域权重更高
                l1 = criterion_l1(fake, target)
                mask_mean = mask.mean() + 1e-6
                loss_l1 = l1 / mask_mean

                # Hinge GAN
                loss_adv = hinge_g_loss(d_fake)

                # Feature Matching
                loss_fm = sum(F.l1_loss(df, dr.detach())
                              for df, dr in zip(d_fake_feats, d_real_feats))

                loss_G = (config.LAMBDA_L1_S2 * loss_l1 +
                          config.LAMBDA_ADV_S2 * loss_adv +
                          config.LAMBDA_FM_S2 * loss_fm)

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            pbar.set_postfix(D=f"{loss_D.item():.2f}", G=f"{loss_G.item():.2f}",
                             L1=f"{loss_l1.item():.4f}")

        # Val
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, target in val_loader:
                inp = inp.to(device); target = target.to(device)
                val_loss += criterion_l1(G(inp), target).item()
        val_loss /= len(val_loader)
        print(f"  Val L1: {val_loss:.4f}")

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss; patience_counter = 0
            torch.save({"epoch": epoch, "model": G.state_dict(), "val": val_loss},
                       config.CHECKPOINT_DIR / "edge_generator_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch}, best: {best_val:.5f}")
                break
        if epoch % 20 == 0:
            torch.save({"epoch": epoch, "model": G.state_dict()},
                       config.CHECKPOINT_DIR / f"edge_generator_e{epoch}.pt")

    print(f"Done. Best: {best_val:.5f}")


if __name__ == "__main__":
    train()
