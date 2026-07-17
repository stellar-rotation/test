"""
端到端线稿重建训练：破损壁画 → 完整线稿
策略：重建损失与损伤区 Tversky，之后逐步加入 Skeleton + LSGAN
DWA (Dynamic Weight Averaging) 自动平衡各重建损失权重
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import os
import csv
import json
import platform
import random
import numpy as np
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
from losses.metrics import evaluate_lineart
from data.damaged_mural_dataset import DamagedMuralDataset
import math


METRIC_NAMES = (
    "hole_precision",
    "hole_recall",
    "hole_f1",
    "hole_cldice",
    "hole_hd95",
    "valid_hallucination_rate",
)


def append_epoch_log(path, row):
    """Append one epoch to a stable, analysis-ready CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed, deterministic=True):
    """Fix all RNGs used by model initialization, augmentation and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    """Give every DataLoader worker a reproducible, distinct RNG stream."""
    del worker_id  # worker seed is already encoded in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def train(resume_ckpt=None, experiment="full"):
    if experiment not in config.EXPERIMENT_PRESETS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; "
            f"choose from {tuple(config.EXPERIMENT_PRESETS)}"
        )
    loss_flags = config.EXPERIMENT_PRESETS[experiment].copy()
    if config.DWA_ENABLED:
        raise ValueError("Ablation presets require DWA_ENABLED=False to keep weights fixed")
    experiment_name = f"seed{config.SEED}_{experiment}"
    experiment_dir = config.CHECKPOINT_DIR / "ablations" / experiment_name
    visualization_dir = experiment_dir / "visualizations"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = experiment_dir / "training_metrics.csv"
    if metrics_path.exists() and resume_ckpt is None:
        raise FileExistsError(
            f"Experiment output already exists: {metrics_path}. "
            "Use --resume with a matching checkpoint or move the old directory."
        )

    set_seed(config.SEED, config.DETERMINISTIC)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Experiment: {experiment_name} | Device: {device} | Seed: {config.SEED} | "
        f"Deterministic: {config.DETERMINISTIC}"
    )
    gan_enabled = loss_flags.get("gan", True)
    print(
        f"Losses: L1=on Edge=on GAN={'on' if gan_enabled else 'off'} | "
        f"{loss_flags}"
    )

    experiment_metadata = {
        "experiment": experiment,
        "experiment_name": experiment_name,
        "seed": config.SEED,
        "losses": {"l1": True, "edge": True, "gan": gan_enabled, **loss_flags},
        "loss_weights": {
            "l1": config.LAMBDA_L1,
            "edge": config.LAMBDA_EDGE,
            "ssim": config.LAMBDA_SSIM,
            "tversky": config.LAMBDA_TVERSKY,
            "skeleton": config.LAMBDA_SKEL,
            "direction": config.LAMBDA_DIR,
            "gan": config.LAMBDA_ADV,
        },
        "training": {
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "num_epochs": config.NUM_EPOCHS,
            "early_stop_patience": config.EARLY_STOP_PATIENCE,
            "skeleton_start_epoch": config.SKEL_START_EPOCH,
            "gan_start_epoch": config.GAN_START_EPOCH,
            "lr_g": config.LR_G,
            "lr_d": config.LR_D,
            "lr_decay_epoch": config.LR_DECAY_EPOCH,
            "deterministic": config.DETERMINISTIC,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    with (experiment_dir / "experiment_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(experiment_metadata, stream, ensure_ascii=False, indent=2)

    train_generator = torch.Generator()
    train_generator.manual_seed(config.SEED)
    val_generator = torch.Generator()
    val_generator.manual_seed(config.SEED + 1)

    # ── 数据 ──
    train_ds = DamagedMuralDataset(
        str(config.TRAIN_IMAGES), str(config.TRAIN_EDGES),
        image_size=config.IMAGE_SIZE, augment=True,
    )
    val_ds = DamagedMuralDataset(
        str(config.VAL_IMAGES), str(config.VAL_EDGES),
        image_size=config.IMAGE_SIZE, augment=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=val_generator)

    # ── 模型 ──
    G = ExtractionGenerator(in_ch=config.TRAIN_IN_CH, out_ch=config.TRAIN_OUT_CH,
                            base_ch=config.STAGE1_BASE_CH, num_downs=config.NUM_DOWNS).to(device)
    D = (
        PatchGANDiscriminator(
            in_ch=config.TRAIN_IN_CH + config.TRAIN_OUT_CH
        ).to(device)
        if gan_enabled
        else None
    )

    # ── 损失 ──
    criterion_l1 = nn.L1Loss()
    criterion_edge = GradientLoss().to(device)
    # Skeleton is absent from the adopted final configuration. Instantiate it
    # only when reproducing an older ablation preset that explicitly enables it.
    criterion_skel = SkeletonLoss().to(device) if loss_flags["skeleton"] else None
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
    opt_D = (
        optim.Adam(D.parameters(), lr=config.LR_D, betas=(config.BETA1, config.BETA2))
        if D is not None
        else None
    )
    scaler = GradScaler()
    
    # ── 恢复训练状态 ──
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    loss_history = []  # 每个 Epoch 的各项 Loss 均值，用于 DWA
    resume_g_only = False
    gan_resume_start_epoch = None
    
    if resume_ckpt:
        import os
        if os.path.exists(resume_ckpt):
            print(f"==> Resuming from checkpoint: {resume_ckpt}")
            checkpoint = torch.load(resume_ckpt, map_location=device)
            checkpoint_experiment = checkpoint.get("experiment")
            if checkpoint_experiment not in (None, experiment):
                raise ValueError(
                    f"Checkpoint belongs to {checkpoint_experiment!r}, "
                    f"not requested experiment {experiment!r}"
                )
            G.load_state_dict(checkpoint['model'])
            if D is not None and 'D' in checkpoint:
                D.load_state_dict(checkpoint['D'])
            elif D is not None:
                resume_g_only = True
            if 'opt_G' in checkpoint:
                opt_G.load_state_dict(checkpoint['opt_G'])
            elif 'opt' in checkpoint:
                opt_G.load_state_dict(checkpoint['opt'])
            if opt_D is not None and 'opt_D' in checkpoint:
                opt_D.load_state_dict(checkpoint['opt_D'])
            if 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float("inf"))
            best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch', 0))
            print(f"    Resumed at epoch {start_epoch - 1}, best val loss: {best_val_loss:.4f}")
            if resume_g_only and start_epoch > config.GAN_START_EPOCH:
                gan_resume_start_epoch = start_epoch + config.RESUME_G_ONLY_GAN_WARMUP_EPOCHS
                print(
                    "    Checkpoint has no discriminator state; "
                    f"GAN will restart at epoch {gan_resume_start_epoch}."
                )
        else:
            print(f"==> Warning: Checkpoint not found at {resume_ckpt}, starting from scratch.")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        last_epoch = epoch
        # LR decay
        if epoch > config.LR_DECAY_EPOCH:
            d = 1.0 - (epoch - config.LR_DECAY_EPOCH) / (config.NUM_EPOCHS - config.LR_DECAY_EPOCH)
            for pg in opt_G.param_groups: pg["lr"] = config.LR_G * max(d, 0.0)
            if opt_D is not None:
                for pd in opt_D.param_groups:
                    pd["lr"] = config.LR_D * max(d, 0.0)

        # 阶段性策略
        use_gan = gan_enabled and epoch > config.GAN_START_EPOCH
        if gan_resume_start_epoch is not None and epoch < gan_resume_start_epoch:
            use_gan = False
        # Keep checkpoint selection and early-stop windows identical between
        # GAN and no-GAN experiments for a fair objective-level comparison.
        selection_stage_ready = epoch > config.GAN_START_EPOCH
        topology_stage = epoch > config.SKEL_START_EPOCH
        use_skeleton = topology_stage and loss_flags["skeleton"]
        use_direction = topology_stage and loss_flags["direction"]
        use_topology = use_skeleton or use_direction
        w_gan = config.LAMBDA_ADV if use_gan else 0.0
        w_skel = config.LAMBDA_SKEL if use_skeleton else 0.0

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
        if (config.DWA_ENABLED and use_skeleton and use_direction and len(loss_history) >= 2
                and loss_history[-1]['skel'] is not None
                and loss_history[-2]['skel'] is not None):
            rates_late = [
                h1['skel'] / (h2['skel'] + 1e-8),
                h1['dir']  / (h2['dir']  + 1e-8),
            ]
            dwa_late = compute_dwa(rates_late, config.DWA_TEMP)
        else:
            dwa_late = [1.0, 1.0]

        G.train()
        if D is not None:
            D.train()
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

                if loss_flags["ssim"]:
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
                    per_pixel_ssim = 1.0 - ssim_map
                    loss_G_ssim = weighted_per_sample_mean(
                        per_pixel_ssim, pixel_weights
                    )

                if loss_flags["tversky"]:
                    loss_G_tversky = criterion_tversky(
                        fake_edge, real_edge, damage_mask_bin
                    )

                loss_G = dwa_base[0] * config.LAMBDA_L1 * loss_G_l1 + \
                         dwa_base[1] * config.LAMBDA_EDGE * loss_G_edge
                if loss_flags["ssim"]:
                    loss_G += dwa_base[2] * config.LAMBDA_SSIM * loss_G_ssim
                if loss_flags["tversky"]:
                    loss_G += config.LAMBDA_TVERSKY * loss_G_tversky

                if use_skeleton:
                    loss_G_skel = criterion_skel(fake_edge, real_edge)
                    loss_G += dwa_late[0] * w_skel * loss_G_skel
                if use_direction:
                    loss_G_dir = criterion_dir(fake_edge, real_edge)
                    loss_G += dwa_late[1] * config.LAMBDA_DIR * loss_G_dir

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
            if loss_flags["ssim"]:
                epoch_sums['ssim'] += loss_G_ssim.item()
            if loss_flags["tversky"]:
                epoch_sums['tversky'] += loss_G_tversky.item()
            if use_skeleton:
                epoch_sums['skel'] += loss_G_skel.item()
            if use_direction:
                epoch_sums['dir']  += loss_G_dir.item()
            if use_gan:
                epoch_sums['adv'] += loss_G_adv.item()
            num_batches += 1

            info = {"L1": f"{loss_G_l1.item():.3f}"}
            if use_gan:
                info["D"] = f"{loss_D.item():.2f}"
            if use_skeleton:
                info["Sk"] = f"{loss_G_skel.item():.3f}"
            if use_direction:
                info["Dir"] = f"{loss_G_dir.item():.3f}"
            if loss_flags["tversky"]:
                info["Tv"] = f"{loss_G_tversky.item():.3f}"
            info["\u03bb"] = "/".join(f"{v:.2f}" for v in dwa_base + (dwa_late if use_topology else []))
            pbar.set_postfix(**info)

        # ── 记录 DWA 历史 ──
        n = max(num_batches, 1)
        loss_history.append({
            'l1':   epoch_sums['l1']   / n,
            'edge': epoch_sums['edge'] / n,
            'ssim': epoch_sums['ssim'] / n,
            'skel': epoch_sums['skel'] / n if use_skeleton else None,
            'dir':  epoch_sums['dir']  / n if use_direction else None,
        })
        means = {key: value / n for key, value in epoch_sums.items()}
        weighted = {
            'l1': config.LAMBDA_L1 * dwa_base[0] * means['l1'],
            'edge': config.LAMBDA_EDGE * dwa_base[1] * means['edge'],
        }
        if loss_flags["ssim"]:
            weighted['ssim'] = config.LAMBDA_SSIM * dwa_base[2] * means['ssim']
        if loss_flags["tversky"]:
            weighted['tversky'] = config.LAMBDA_TVERSKY * means['tversky']
        if use_skeleton:
            weighted['skel'] = config.LAMBDA_SKEL * dwa_late[0] * means['skel']
        if use_direction:
            weighted['dir'] = config.LAMBDA_DIR * dwa_late[1] * means['dir']
        if use_gan:
            weighted['adv'] = config.LAMBDA_ADV * means['adv']

        raw_text = " / ".join(f"{key}={means[key]:.4f}" for key in weighted)
        weighted_text = " / ".join(f"{key}={value:.3f}" for key, value in weighted.items())
        print(f"  Raw loss: {raw_text}")
        print(f"  Weighted contribution: {weighted_text}")
        if config.DWA_ENABLED:
            all_lambdas = dwa_base + (dwa_late if use_topology else [])
            print(f"  DWA λ: {' / '.join(f'{v:.3f}' for v in all_lambdas)}")

        # ── 验证与可视化 ──
        G.eval()
        val_abs_sum = 0.0
        val_pixel_count = 0
        val_hole_sum = 0.0
        val_hole_count = 0.0
        val_valid_sum = 0.0
        val_valid_count = 0.0
        metric_sums = {name: 0.0 for name in METRIC_NAMES}
        metric_image_count = 0
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

                pred_np = fake_edge[:, 0].float().cpu().numpy()
                target_np = real_edge[:, 0].float().cpu().numpy()
                damage_np = damage_mask[:, 0].bool().cpu().numpy()
                for pred_item, target_item, damage_item in zip(
                    pred_np, target_np, damage_np
                ):
                    item_metrics = evaluate_lineart(
                        pred_item, target_item, damage_item
                    )
                    for name in METRIC_NAMES:
                        metric_sums[name] += item_metrics[name]
                    metric_image_count += 1

                # 每个 Epoch 保存第一批数据的可视化对比图
                if batch_idx == 0:
                    import torchvision
                    import os
                    # 取前 4 张图拼起来 (RGB需要去掉第4通道mask)
                    n_vis = min(4, degraded_img.size(0))
                    # degraded_img是 4ch，取前 3ch 显示
                    rgb_vis = degraded_img[:n_vis, :3]
                    pred_vis = fake_edge[:n_vis].repeat(1, 3, 1, 1) # 转成3通道方便查看
                    gt_vis = real_edge[:n_vis].repeat(1, 3, 1, 1)
                    
                    grid = torch.cat([rgb_vis, pred_vis, gt_vis], dim=0)
                    torchvision.utils.save_image(
                        grid, 
                        visualization_dir / f"epoch_{epoch:03d}.png", 
                        nrow=n_vis, 
                        normalize=True, 
                        value_range=(-1, 1) if rgb_vis.min() < 0 else (0, 1)
                    )

        val_full_l1 = val_abs_sum / max(val_pixel_count, 1)
        val_hole_l1 = val_hole_sum / max(val_hole_count, 1.0)
        val_valid_l1 = val_valid_sum / max(val_valid_count, 1.0)
        val_metrics = {
            name: metric_sums[name] / max(metric_image_count, 1)
            for name in METRIC_NAMES
        }
        val_hole_precision = val_metrics["hole_precision"]
        val_hole_recall = val_metrics["hole_recall"]
        val_hole_f1 = val_metrics["hole_f1"]
        val_loss = 1.0 - val_hole_f1
        print(
            f"  Val L1: full={val_full_l1:.4f} / hole={val_hole_l1:.4f} "
            f"/ valid={val_valid_l1:.4f}  |  "
            f"Hole line P/R/F1={val_hole_precision:.4f}/"
            f"{val_hole_recall:.4f}/{val_hole_f1:.4f}  |  "
            f"clDice={val_metrics['hole_cldice']:.4f} / "
            f"HD95={val_metrics['hole_hd95']:.2f}px / "
            f"VHR={val_metrics['valid_hallucination_rate']:.4f}  "
            f"(GAN={'on' if use_gan else 'off'}, "
            f"Skel={'on' if use_skeleton else 'off'}, "
            f"Dir={'on' if use_direction else 'off'})"
        )

        log_row = {
            "epoch": epoch,
            "seed": config.SEED,
            "val_full_l1": val_full_l1,
            "val_hole_l1": val_hole_l1,
            "val_valid_l1": val_valid_l1,
            **val_metrics,
        }
        append_epoch_log(
            metrics_path,
            log_row,
        )

        # A checkpoint can become the experiment winner only after every
        # fixed loss (especially GAN) is active. Pre-GAN metrics remain in CSV
        # for diagnosis but cannot select the final model.
        if selection_stage_ready and val_loss < best_val_loss - config.EARLY_STOP_MIN_DELTA:
            best_val_loss = val_loss; best_epoch = epoch; patience_counter = 0
            best_state = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "seed": config.SEED,
                "experiment": experiment,
                "loss_flags": loss_flags,
                "model": G.state_dict(),
                "val_loss": val_loss,
                "val_full_l1": val_full_l1,
                "val_hole_l1": val_hole_l1,
                "val_valid_l1": val_valid_l1,
                "val_metrics": val_metrics,
                "val_hole_precision": val_hole_precision,
                "val_hole_recall": val_hole_recall,
                "val_hole_f1": val_hole_f1,
            }
            if D is not None:
                best_state["D"] = D.state_dict()
            torch.save(best_state, experiment_dir / "best.pt")
        else:
            # Do not consume early-stop patience before every fixed loss has
            # actually joined training. In particular, GAN starts after its
            # warm-up (and may start later when resuming a G-only checkpoint).
            if selection_stage_ready:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOP_PATIENCE:
                    print(
                        f"  Early stop at epoch {epoch} after "
                        f"{patience_counter} non-improving selection-stage epochs, "
                        f"best: {best_val_loss:.4f}"
                    )
                    break
            else:
                patience_counter = 0
        if epoch % 20 == 0:
            periodic_state = {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "seed": config.SEED,
                "experiment": experiment,
                "loss_flags": loss_flags,
                "model": G.state_dict(),
                "opt_G": opt_G.state_dict(),
                "scaler": scaler.state_dict(),
            }
            if D is not None:
                periodic_state["D"] = D.state_dict()
                periodic_state["opt_D"] = opt_D.state_dict()
            torch.save(periodic_state, experiment_dir / f"epoch_{epoch:03d}.pt")

    run_summary = {
        "experiment": experiment,
        "experiment_name": experiment_name,
        "seed": config.SEED,
        "last_epoch": last_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(experiment_dir / "best.pt"),
    }
    with (experiment_dir / "run_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(run_summary, stream, ensure_ascii=False, indent=2)
    print(
        f"Done. Best: {best_val_loss:.4f} at epoch {best_epoch} | "
        f"Output: {experiment_dir}"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train extraction generator")
    parser.add_argument(
        "--experiment",
        choices=tuple(config.EXPERIMENT_PRESETS),
        default=config.FINAL_EXPERIMENT,
        help=f"Loss configuration (default final model: {config.FINAL_EXPERIMENT})",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to a matching experiment checkpoint")
    args = parser.parse_args()
    train(resume_ckpt=args.resume, experiment=args.experiment)
