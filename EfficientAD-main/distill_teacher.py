#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Distill a 384-channel Teacher into a 192-channel small Teacher.

No ImageNet needed — uses the original Teacher as the target and only
requires the normal training images (train/good/).

Usage:
    python distill_teacher.py \
        --teacher EfficientAD-main/output/verytiny-batch=4/trainings/mvtec_ad/my_product/teacher_final.pth \
        --train-dir mydataset/my_product/train \
        --output models/teacher_tiny192.pth \
        --target-channels 192
"""
import argparse
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import get_pdn_small, get_pdn_tiny, ImageFolderWithoutTarget


def get_argparse():
    parser = argparse.ArgumentParser(
        description="Distill Teacher to fewer output channels")
    parser.add_argument("--teacher", required=True,
                        help="Path to original 384ch teacher .pth")
    parser.add_argument("--train-dir", required=True,
                        help="Path to train/good/ images")
    parser.add_argument("--output", default="models/teacher_distilled.pth",
                        help="Output path for distilled teacher")
    parser.add_argument("--target-channels", type=int, default=192,
                        help="Output channels for distilled teacher "
                             "(default 192, half of original 384)")
    parser.add_argument("--model-size", default="small",
                        choices=["small", "medium", "tiny"],
                        help="Distilled teacher architecture")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


# ── data ──────────────────────────────────────────────────────────────

image_size = 256
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── main ──────────────────────────────────────────────────────────────

def main():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Target channels: {config.target_channels}")
    print(f"Architecture: {config.model_size}")

    # ── Load original teacher (384ch, frozen) ──
    print(f"\nLoading original teacher: {config.teacher}")
    teacher_big = torch.load(config.teacher, map_location=device,
                              weights_only=False)
    teacher_big.eval()
    teacher_big.requires_grad_(False)
    print(f"  Original teacher: {sum(p.numel() for p in teacher_big.parameters()):,} params")

    # ── Create small teacher (target_channels, trainable) ──
    if config.model_size == "small":
        teacher_small = get_pdn_small(config.target_channels, padding=False)
    elif config.model_size == "tiny":
        teacher_small = get_pdn_tiny(config.target_channels, padding=False)
    else:
        teacher_small = get_pdn_small(config.target_channels, padding=False)
    teacher_small.train()
    teacher_small.to(device)
    n_small = sum(p.numel() for p in teacher_small.parameters())
    n_big = sum(p.numel() for p in teacher_big.parameters())
    print(f"  Distilled teacher: {n_small:,} params "
          f"({n_small / n_big * 100:.0f}% of original, "
          f"last layer channels: {config.target_channels})")

    # ── Data ──
    train_set = ImageFolderWithoutTarget(
        config.train_dir,
        transform=transforms.Lambda(default_transform),
    )
    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    print(f"\nTraining images: {len(train_set)}")

    # ── Compute teacher_mean/std from big teacher ──
    print("Computing teacher normalization...")
    means = []
    with torch.no_grad():
        for images in tqdm(train_loader, desc="Teacher mean"):
            images = images.to(device)
            out = teacher_big(images)
            means.append(torch.mean(out, dim=[0, 2, 3]))
    teacher_mean = torch.mean(torch.stack(means), dim=0)[None, :, None, None]

    vars_list = []
    with torch.no_grad():
        for images in tqdm(train_loader, desc="Teacher std"):
            images = images.to(device)
            out = teacher_big(images)
            vars_list.append(torch.mean((out - teacher_mean) ** 2, dim=[0, 2, 3]))
    teacher_std = torch.sqrt(torch.mean(torch.stack(vars_list), dim=0))
    teacher_std = teacher_std[None, :, None, None].to(device)
    teacher_mean = teacher_mean.to(device)
    print(f"  mean shape: {teacher_mean.shape}  std shape: {teacher_std.shape}")

    # ── Training ──
    optimizer = torch.optim.Adam(
        teacher_small.parameters(), lr=config.lr, weight_decay=1e-5,
    )
    total_steps = config.epochs * len(train_loader)
    print(f"\nTraining {config.epochs} epochs ({total_steps} steps)...")

    best_loss = float("inf")
    global_step = 0

    for epoch in range(config.epochs):
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for images in pbar:
            images = images.to(device)

            with torch.no_grad():
                target_full = teacher_big(images)
                target_full = (target_full - teacher_mean) / (teacher_std + 1e-6)
                # Take only the first target_channels from big teacher
                target = target_full[:, :config.target_channels]

            output = teacher_small(images)
            loss = F.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = np.mean(epoch_losses)
        print(f"  Epoch {epoch+1}: avg loss = {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(config.output), exist_ok=True)
            torch.save(teacher_small, config.output)
            print(f"  → Saved best model to {config.output}")

    print(f"\nDone. Best loss: {best_loss:.6f}")
    print(f"Distilled teacher saved to: {config.output}")
    print(f"\nNext: update efficientad2.py teacher_channels from 384 "
          f"to {config.target_channels} and use --weights {config.output}")


if __name__ == "__main__":
    main()
