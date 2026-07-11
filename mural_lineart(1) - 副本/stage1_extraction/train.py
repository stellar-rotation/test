"""
端到端线稿重建训练：破损壁画 → 完整线稿
策略：重建损失与损伤区 Tversky，之后逐步加入 Skeleton + LSGAN
DWA (Dynamic Weight Averaging) 自动平衡各重建损失权重
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import config
from models.pix2pix import ExtractionGenerator, PatchGANDiscriminator
from losses.gradient_loss import GradientLoss
from losses.skeleton_loss import SkeletonLoss
from losses.direction_loss import DirectionLoss
from losses.tversky_loss import MaskedTverskyLoss
from data.damaged_mural_dataset import DamagedMuralDataset
import math


def compute_dwa(rates, T):
    """计算 DWA 动态权重乘子（带温度的 Softmax，均值归一化到 1.0）"""
    K = len(rates)
    max_r = max(r / T for r in rates)
    exp_rates = [math.exp(r / T - max_r) for r in rates]
    sum_exp = sum(exp_rates)
    return [K * e / sum_exp for e in exp_rates]


def weighted_per_sample_mean(loss_map, pixel_weights):
    """Normalize a weighted pixel loss per image, then average the batch."""
    if loss_map.shape != pixel_weights.shape:
        raise ValueError(
            f"loss_map and pixel_weights must match, got "
            f"{loss_map.shape} and {pixel_weights.shape}"
        )
    reduce_dims = tuple(range(1, loss_map.ndim))
    numerator = (loss_map * pixel_weights).sum(dim=reduce_dims)
    denominator = pixel_weights.sum(dim=reduce_dims).clamp_min(1.0)
    return (numerator / denominator).mean()


def train(resume_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 数据 ──
    train_ds = DamagedMuralDataset(
        str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES),
        image_size=config.IMAGE_SIZE, augment=True,
    )
    val_ds = DamagedMuralDataset(
        str(config.VAL_IMAGES), str(config.VAL_EDGES),
        image_size=config.IMAGE_SIZE, augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    # ── 模型 ──
    G = ExtractionGenerator(in_ch=config.TRAIN_IN_CH, out_ch=config.TRAIN_OUT_CH,
                            base_ch=config.STAGE1_BASE_CH, num_downs=config.NUM_DOWNS).to(device)
    D = PatchGANDiscriminator(in_ch=config.TRAIN_IN_CH + config.TRAIN_OUT_CH).to(device)

    # ── 损失 ──
    criterion_l1 = nn.L1Loss()
    criterion_edge = GradientLoss().to(device)
    criterion_skel = SkeletonLoss().to(device)
    criterion_dir = DirectionLoss().to(device)
    criterion_tversky = MaskedTverskyLoss(
        alpha=config.TVERSKY_ALPHA,
        beta=config.TVERSKY_BETA,
        temperature=config.TVERSKY_TEMPERATURE,
    ).to(device)

    def ssim_loss(pred, target, window_size=11):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        p = (pred + 1.0) / 2.0; t = (target + 1.0) / 2.0
        mu_x = torch.nn.functional.avg_pool2d(p, window_size, 1, window_size // 2)
        mu_y = torch.nn.functional.avg_pool2d(t, window_size, 1, window_size // 2)
        mx2, my2 = mu_x ** 2, mu_y ** 2; mxy = mu_x * mu_y
        sx = torch.nn.functional.avg_pool2d(p ** 2, window_size, 1, window_size // 2) - mx2
        sy = torch.nn.functional.avg_pool2d(t ** 2, window_size, 1, window_size // 2) - my2
        sxy = torch.nn.functional.avg_pool2d(p * t, window_size, 1, window_size // 2) - mxy
        s = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2) + 1e-8)
        return 1.0 - s.mean()

    # ── 优化器 ──
    opt_G = optim.Adam(G.parameters(), lr=config.LR_G, betas=(config.BETA1, config.BETA2))
    opt_D = optim.Adam(D.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
    scaler = GradScaler()
    
    # ── 恢复训练状态 ──
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    loss_history = []  # 每个 Epoch 的各项 Loss 均值，用于 DWA
    resume_g_only = False
    gan_resume_start_epoch = None
    
    if resume_ckpt:
        import os
        if os.path.exists(resume_ckpt):
            print(f"==> Resuming from checkpoint: {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location=device)
            G.load_state_dict(checkpoint['model'])
            if 'D' in checkpoint:
                D.load_state_dict(checkpoint['D'])
            else:
                resume_g_only = True
            if 'opt_G' in checkpoint:
                opt_G.load_state_dict(checkpoint['opt_G'])
            elif 'opt' in checkpoint:
                opt_G.load_state_dict(checkpoint['opt'])
            if 'opt_D' in checkpoint:
                opt_D.load_state_dict(checkpoint['opt_D'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float("inf"))
            print(f"    Resumed at epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")
            if resume_g_only and start_epoch > config.GAN_START_EPOCH:
                gan_resume_start_epoch = start_epoch + config.RESUME_G_ONLY_GAN_WARMUP_EPOCHS
                print(
                    "    Checkpoint has no discriminator state; "
                    f"GAN will restart at epoch {gan_resume_start_epoch}."
                )
        else:
            print(f"==> Warning: Checkpoint not found at {resume_ckpt}, starting from scratch.")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        # LR decay
        if epoch > config.LR_DECAY_EPOCH:
            d = 1.0 - (epoch - config.LR_DECAY_EPOCH) / (config.NUM_EPOCHS - config.LR_DECAY_EPOCH)
            for pg in opt_G.param_groups: pg["lr"] = config.LR_G * max(d, 0.0)
            for pd in opt_D.param_groups: pd["lr"] = config.LR_D * max(d, 0.0)

        # 阶段性策略
        use_gan = epoch > config.GAN_START_EPOCH
        if gan_resume_start_epoch is not None and epoch < gan_resume_start_epoch:
            use_gan = False
        use_skel = epoch > config.SKEL_START_EPOCH
        w_gan = config.LAMBDA_ADV if use_gan else 0.0
        w_skel = config.LAMBDA_SKEL if use_skel else 0.0

        # ── DWA 动态权重 ──
        # 基础组 (L1, Edge, SSIM): 从第 3 个 Epoch 起生效
        if config.DWA_ENABLED and len(loss_history) >= 2:
            h1, h2 = loss_history[-1], loss_history[-2]
            rates_base = [
                h1['l1']   / (h2['l1']   + 1e-8),
                h1['edge'] / (h2['edge'] + 1e-8),
                h1['ssim'] / (h2['ssim'] + 1e-8),
            ]
            dwa_base = compute_dwa(rates_base, config.DWA_TEMP)
        else:
            dwa_base = [1.0, 1.0, 1.0]
        # 延迟组 (Skel, Dir): 开启后积累 2 个 Epoch 才生效
        if (config.DWA_ENABLED and use_skel and len(loss_history) >= 2
                and loss_history[-1]['skel'] is not None
                and loss_history[-2]['skel'] is not None):
            rates_late = [
                h1['skel'] / (h2['skel'] + 1e-8),
                h1['dir']  / (h2['dir']  + 1e-8),
            ]
            dwa_late = compute_dwa(rates_late, config.DWA_TEMP)
        else:
            dwa_late = [1.0, 1.0]

        G.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
        # 累计各 Loss 用于 DWA 历史记录
        epoch_sums = {
            'l1': 0., 'edge': 0., 'ssim': 0., 'skel': 0., 'dir': 0.,
            'tversky': 0., 'adv': 0.,
        }
        num_batches = 0

        for degraded_img, real_edge in pbar:
            degraded_img = degraded_img.to(device)
            real_edge = real_edge.to(device)

            # ── G 前向（唯一一次）──
            with autocast():
                fake_edge = G(degraded_img)

            # ── 训练 D（仅 epoch > 20, fp32 避免 LSGAN 平方溢出）──
            if use_gan:
                real_pair = torch.cat([degraded_img, real_edge], dim=1)
                fake_pair = torch.cat([degraded_img, fake_edge.detach().float()], dim=1)
                pred_real = D(real_pair)
                pred_fake = D(fake_pair)
                loss_D = ((pred_real - 1.0) ** 2).mean() * 0.5 + \
                         (pred_fake ** 2).mean() * 0.5
                opt_D.zero_grad()
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()

            # ── 训练 G（复用 fake_edge，计算图完好）──
            with autocast():
                # Line-aware 权重图：重点监督破损区内的真实线条及其邻域。
                damage_mask_bin = (degraded_img[:, 3:4] > 0).float()  # [B,1,H,W]
                target_line = (real_edge < 0).float()
                line_focus = torch.nn.functional.max_pool2d(
                    target_line,
                    kernel_size=config.LINE_FOCUS_KERNEL,
                    stride=1,
                    padding=config.LINE_FOCUS_KERNEL // 2,
                )
                hole_weights = (
                    config.HOLE_BACKGROUND_WEIGHT
                    + (config.HOLE_LINE_WEIGHT - config.HOLE_BACKGROUND_WEIGHT)
                    * line_focus
                )
                pixel_weights = (
                    config.VALID_RECON_WEIGHT * (1.0 - damage_mask_bin)
                    + hole_weights * damage_mask_bin
                )

                # Mask-aware L1
                per_pixel_l1 = torch.abs(fake_edge - real_edge)
                loss_G_l1 = weighted_per_sample_mean(
                    per_pixel_l1, pixel_weights
                )

                # Mask-aware Edge Loss (Sobel 梯度域)
                grad_pred = criterion_edge._gradient(fake_edge)
                grad_target = criterion_edge._gradient(real_edge)
                per_pixel_edge = torch.abs(grad_pred - grad_target)
                loss_G_edge = weighted_per_sample_mean(
                    per_pixel_edge, pixel_weights
                )

                # Mask-aware SSIM Loss
                C1, C2 = 0.01 ** 2, 0.03 ** 2
                p = (fake_edge + 1.0) / 2.0; t = (real_edge + 1.0) / 2.0
                ws = 11
                mu_x = torch.nn.functional.avg_pool2d(p, ws, 1, ws // 2)
                mu_y = torch.nn.functional.avg_pool2d(t, ws, 1, ws // 2)
                mx2, my2 = mu_x ** 2, mu_y ** 2; mxy = mu_x * mu_y
                sx = torch.nn.functional.avg_pool2d(p ** 2, ws, 1, ws // 2) - mx2
                sy = torch.nn.functional.avg_pool2d(t ** 2, ws, 1, ws // 2) - my2
                sxy = torch.nn.functional.avg_pool2d(p * t, ws, 1, ws // 2) - mxy
                ssim_map = ((2 * mxy + C1) * (2 * sxy + C2)) / ((mx2 + my2 + C1) * (sx + sy + C2) + 1e-8)
                per_pixel_ssim = 1.0 - ssim_map  # 越大越差
                loss_G_ssim = weighted_per_sample_mean(
                    per_pixel_ssim, pixel_weights
                )

                # Hole-only Tversky: directly balance missed and hallucinated lines.
                loss_G_tversky = criterion_tversky(
                    fake_edge, real_edge, damage_mask_bin
                )

                loss_G = dwa_base[0] * config.LAMBDA_L1 * loss_G_l1 + \
                         dwa_base[1] * config.LAMBDA_EDGE * loss_G_edge + \
                         dwa_base[2] * config.LAMBDA_SSIM * loss_G_ssim + \
                         config.LAMBDA_TVERSKY * loss_G_tversky

                if use_skel:
                    loss_G_skel = criterion_skel(fake_edge, real_edge)
                    loss_G_dir = criterion_dir(fake_edge, real_edge)
                    loss_G += dwa_late[0] * w_skel * loss_G_skel + \
                              dwa_late[1] * config.LAMBDA_DIR * loss_G_dir

                if use_gan:
                    fake_pair = torch.cat([degraded_img, fake_edge], dim=1)
                    loss_G_adv = ((D(fake_pair) - 1.0) ** 2).mean()
                    loss_G += w_gan * loss_G_adv

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            scaler.step(opt_G)
            scaler.update()  # 每个 iteration 仅调用一次

            # 累计各 Loss 均值
            epoch_sums['l1']   += loss_G_l1.item()
            epoch_sums['edge'] += loss_G_edge.item()
            epoch_sums['ssim'] += loss_G_ssim.item()
            epoch_sums['tversky'] += loss_G_tversky.item()
            if use_skel:
                epoch_sums['skel'] += loss_G_skel.item()
                epoch_sums['dir']  += loss_G_dir.item()
            if use_gan:
                epoch_sums['adv'] += loss_G_adv.item()
            num_batches += 1

            info = {"L1": f"{loss_G_l1.item():.3f}"}
            if use_gan:
                info["D"] = f"{loss_D.item():.2f}"
            if use_skel:
                info["Sk"] = f"{loss_G_skel.item():.3f}"
            info["Tv"] = f"{loss_G_tversky.item():.3f}"
            info["\u03bb"] = "/".join(f"{v:.2f}" for v in dwa_base + (dwa_late if use_skel else []))
            pbar.set_postfix(**info)

        # ── 记录 DWA 历史 ──
        n = max(num_batches, 1)
        loss_history.append({
            'l1':   epoch_sums['l1']   / n,
            'edge': epoch_sums['edge'] / n,
            'ssim': epoch_sums['ssim'] / n,
            'skel': epoch_sums['skel'] / n if use_skel else None,
            'dir':  epoch_sums['dir']  / n if use_skel else None,
        })
        means = {key: value / n for key, value in epoch_sums.items()}
        weighted = {
            'l1': config.LAMBDA_L1 * dwa_base[0] * means['l1'],
            'edge': config.LAMBDA_EDGE * dwa_base[1] * means['edge'],
            'ssim': config.LAMBDA_SSIM * dwa_base[2] * means['ssim'],
            'tversky': config.LAMBDA_TVERSKY * means['tversky'],
        }
        if use_skel:
            weighted['skel'] = config.LAMBDA_SKEL * dwa_late[0] * means['skel']
            weighted['dir'] = config.LAMBDA_DIR * dwa_late[1] * means['dir']
        if use_gan:
            weighted['adv'] = config.LAMBDA_ADV * means['adv']

        raw_text = " / ".join(f"{key}={means[key]:.4f}" for key in weighted)
        weighted_text = " / ".join(f"{key}={value:.3f}" for key, value in weighted.items())
        print(f"  Raw loss: {raw_text}")
        print(f"  Weighted contribution: {weighted_text}")
        if config.DWA_ENABLED:
            all_lambdas = dwa_base + (dwa_late if use_skel else [])
            print(f"  DWA λ: {' / '.join(f'{v:.3f}' for v in all_lambdas)}")

        # ── 验证与可视化 ──
        G.eval()
        val_abs_sum = 0.0
        val_pixel_count = 0
        val_hole_sum = 0.0
        val_hole_count = 0.0
        val_valid_sum = 0.0
        val_valid_count = 0.0
        val_hole_pred_count = 0.0
        val_hole_target_count = 0.0
        val_hole_precision_hits = 0.0
        val_hole_recall_hits = 0.0
        with torch.no_grad():
            for batch_idx, (degraded_img, real_edge) in enumerate(val_loader):
                degraded_img = degraded_img.to(device); real_edge = real_edge.to(device)
                fake_edge = G(degraded_img)
                abs_error = torch.abs(fake_edge - real_edge)
                damage_mask = (degraded_img[:, 3:4] > 0).to(abs_error.dtype)
                valid_mask = 1.0 - damage_mask

                val_abs_sum += abs_error.sum().item()
                val_pixel_count += abs_error.numel()
                val_hole_sum += (abs_error * damage_mask).sum().item()
                val_hole_count += damage_mask.sum().item()
                val_valid_sum += (abs_error * valid_mask).sum().item()
                val_valid_count += valid_mask.sum().item()

                pred_line = (fake_edge < 0).to(abs_error.dtype)
                target_line = (real_edge < 0).to(abs_error.dtype)
                dilated_pred = torch.nn.functional.max_pool2d(
                    pred_line, kernel_size=5, stride=1, padding=2
                )
                dilated_target = torch.nn.functional.max_pool2d(
                    target_line, kernel_size=5, stride=1, padding=2
                )
                val_hole_pred_count += (pred_line * damage_mask).sum().item()
                val_hole_target_count += (target_line * damage_mask).sum().item()
                val_hole_precision_hits += (
                    pred_line * dilated_target * damage_mask
                ).sum().item()
                val_hole_recall_hits += (
                    target_line * dilated_pred * damage_mask
                ).sum().item()

                # 每个 Epoch 保存第一批数据的可视化对比图
                if batch_idx == 0:
                    import torchvision
                    import os
                    vis_dir = config.CHECKPOINT_DIR / "visualizations"
                    os.makedirs(vis_dir, exist_ok=True)
                    # 取前 4 张图拼起来 (RGB需要去掉第4通道mask)
                    n_vis = min(4, degraded_img.size(0))
                    # degraded_img是 4ch，取前 3ch 显示
                    rgb_vis = degraded_img[:n_vis, :3]
                    pred_vis = fake_edge[:n_vis].repeat(1, 3, 1, 1) # 转成3通道方便查看
                    gt_vis = real_edge[:n_vis].repeat(1, 3, 1, 1)
                    
                    grid = torch.cat([rgb_vis, pred_vis, gt_vis], dim=0)
                    torchvision.utils.save_image(
                        grid, 
                        vis_dir / f"epoch_{epoch:03d}.png", 
                        nrow=n_vis, 
                        normalize=True, 
                        value_range=(-1, 1) if rgb_vis.min() < 0 else (0, 1)
                    )

        val_full_l1 = val_abs_sum / max(val_pixel_count, 1)
        val_hole_l1 = val_hole_sum / max(val_hole_count, 1.0)
        val_valid_l1 = val_valid_sum / max(val_valid_count, 1.0)
        val_hole_precision = val_hole_precision_hits / max(val_hole_pred_count, 1.0)
        val_hole_recall = val_hole_recall_hits / max(val_hole_target_count, 1.0)
        val_hole_f1 = (
            2.0 * val_hole_precision * val_hole_recall
            / max(val_hole_precision + val_hole_recall, 1e-8)
        )
        val_loss = 1.0 - val_hole_f1
        print(
            f"  Val L1: full={val_full_l1:.4f} / hole={val_hole_l1:.4f} "
            f"/ valid={val_valid_l1:.4f}  |  "
            f"Hole line P/R/F1={val_hole_precision:.4f}/"
            f"{val_hole_recall:.4f}/{val_hole_f1:.4f}  "
            f"(GAN={'on' if use_gan else 'off'}, Skel={'on' if use_skel else 'off'})"
        )

        if val_loss < best_val_loss - config.EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss; patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": G.state_dict(),
                "D": D.state_dict(),
                "val_loss": val_loss,
                "val_full_l1": val_full_l1,
                "val_hole_l1": val_hole_l1,
                "val_valid_l1": val_valid_l1,
                "val_hole_precision": val_hole_precision,
                "val_hole_recall": val_hole_recall,
                "val_hole_f1": val_hole_f1,
            },
                       config.CHECKPOINT_DIR / "stage1_extractor_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch}, best: {best_val_loss:.4f}")
                break
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "scaler": scaler.state_dict(),
            }, config.CHECKPOINT_DIR / f"stage1_extractor_e{epoch}.pt")

    print(f"Done. Best: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train extraction generator")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (e.g. checkpoints/stage1_extractor_best.pt)")
    args = parser.parse_args()
    train(resume_ckpt=args.resume)
