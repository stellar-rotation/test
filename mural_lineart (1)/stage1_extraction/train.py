"""阶段一训练：破损壁画 → 线稿 (MSE + Edge，纯提取)"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import config
from models.pix2pix import ExtractionGenerator
from losses.gradient_loss import GradientLoss
from stage1_extraction.data import Stage1Dataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = Stage1Dataset(str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES),
                             str(config.TRAIN_MASKS), image_size=config.IMAGE_SIZE, augment=True)
    val_ds = Stage1Dataset(str(config.VAL_IMAGES), str(config.VAL_EDGES),
                           str(config.VAL_MASKS), image_size=config.IMAGE_SIZE, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    G = ExtractionGenerator(in_ch=config.S1_IN_CH, out_ch=config.S1_OUT_CH,
                            base_ch=config.S1_BASE_CH, num_downs=config.S1_NUM_DOWNS).to(device)

    criterion_mse = nn.MSELoss()
    criterion_edge = GradientLoss().to(device)

    opt = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    scaler = GradScaler()
    best_val = float("inf")
    patience = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for degraded_img, real_edge in pbar:
            degraded_img = degraded_img.to(device); real_edge = real_edge.to(device)
            with autocast():
                fake = G(degraded_img)
                loss = config.LAMBDA_MSE * criterion_mse(fake, real_edge) + \
                       config.LAMBDA_EDGE * criterion_edge(fake, real_edge)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(L=f"{loss.item():.1f}")

        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded_img, real_edge in val_loader:
                degraded_img = degraded_img.to(device); real_edge = real_edge.to(device)
                val_loss += criterion_mse(G(degraded_img), real_edge).item()
        val_loss /= len(val_loader)
        print(f"  Val MSE: {val_loss:.4f}")

        if val_loss < best_val - config.EARLY_STOP_MIN_DELTA:
            best_val = val_loss; patience = 0
            torch.save({"epoch": epoch, "model": G.state_dict(), "val": val_loss},
                       config.CHECKPOINT_DIR / "stage1_best.pt")
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch}, best: {best_val:.4f}")
                break
        if epoch % 20 == 0:
            torch.save({"epoch": epoch, "model": G.state_dict()},
                       config.CHECKPOINT_DIR / f"stage1_e{epoch}.pt")

    print(f"Done. Best: {best_val:.4f}")


if __name__ == "__main__":
    train()
