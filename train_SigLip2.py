# -*- coding: utf-8 -*-
"""
Multi-task Image Quality Assessment — DroneIQA training script

Models  : SigLIP2_ViTL_384_DroneIQA_MT   (ViT-L/16 SigLIP2 384px, shared backbone + 3 heads)
          SigLIP2_So400m_378_DroneIQA_MT  (ViT-SO400M/14 SigLIP2 378px, shared backbone + 3 heads)
Tasks   : global_quality_mean, target_quality_mean, background_quality_mean
Dataset : DroneIQA  (/data/sunwei_data/DroneIQA/train + train.csv)
Split   : random 80% train / 20% val
"""

import argparse
import csv
import gc
import os
import random
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
from PIL import Image
from open_clip import create_model_from_pretrained

from scipy import stats
from scipy.optimize import curve_fit
from timm.layers import Format as TimmFormat

# ────────────────────────────────────────────────────────────────────────────
# Loss & evaluation utilities
# ────────────────────────────────────────────────────────────────────────────
def _logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logistic_part = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    return bayta2 + np.divide(bayta1 - bayta2, logistic_part)


def performance_fit(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(_logistic_func, y_output, y_label, p0=beta, maxfev=100_000_000)
    y_output_logistic = _logistic_func(y_output, *popt)

    PLCC = stats.pearsonr(y_output_logistic, y_label)[0]
    SRCC = stats.spearmanr(y_output, y_label)[0]
    KRCC = stats.kendalltau(y_output, y_label)[0]
    RMSE = np.sqrt(((y_output_logistic - y_label) ** 2).mean())
    return PLCC, SRCC, KRCC, RMSE


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


# ────────────────────────────────────────────────────────────────────────────
# Task definitions
# ────────────────────────────────────────────────────────────────────────────
TASK_NAMES = ['global_quality_mean', 'target_quality_mean', 'background_quality_mean']

_SIGLIP_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_STD  = [0.5, 0.5, 0.5]

MODEL_NORM_STATS = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (_SIGLIP_MEAN, _SIGLIP_STD),
    'SigLIP2_So400m_378_DroneIQA_MT': (_SIGLIP_MEAN, _SIGLIP_STD),
}

MODEL_DEFAULT_SIZES = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (432, 384),
    'SigLIP2_So400m_378_DroneIQA_MT': (432, 378),   # 378 = 27 * 14, native for So400m
}


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────
class DroneIQADataset(data.Dataset):
    """
    Multi-task dataset for DroneIQA.

    Reads image paths and three quality labels from a CSV file.
    Random 80/20 split controlled by `seed`.
    """

    def __init__(self, img_dir, csv_path, split, transform, seed=0):
        super().__init__()
        assert split in ('train', 'val'), "split must be 'train' or 'val'"

        self.img_dir   = img_dir
        self.transform = transform

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            all_entries = list(reader)

        indices = list(range(len(all_entries)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train = int(len(indices) * 0.8)
        if split == 'train':
            selected_idx = indices[:n_train]
        else:
            selected_idx = indices[n_train:]

        selected = [all_entries[i] for i in selected_idx]

        self.filenames = [e['filename'] for e in selected]
        self.labels = {
            name: [float(e[name]) for e in selected]
            for name in TASK_NAMES
        }
        self.length = len(self.filenames)

        print(f'\n[{split.upper()}]  {self.length} images  '
              f'(total {len(all_entries)}, seed={seed})')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename  = self.filenames[idx]
        full_path = os.path.join(self.img_dir, filename)

        image = Image.open(full_path).convert('RGB')
        image = self.transform(image)

        label_dict = {
            name: torch.FloatTensor(np.array(self.labels[name][idx]))
            for name in TASK_NAMES
        }

        return image, label_dict, filename


# ────────────────────────────────────────────────────────────────────────────
# Models (multi-task, 3 heads)
# ────────────────────────────────────────────────────────────────────────────
class SigLIP2_ViTL_384_DroneIQA_MT(nn.Module):
    """Multi-task SigLIP2 ViT-L/16 384px — 3 quality heads for DroneIQA."""

    def __init__(self):
        super().__init__()
        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP2-384')
        self.feature_extraction = model.visual

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1024, 128),
                nn.Linear(128, 1),
            )
            for name in TASK_NAMES
        })

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat = torch.flatten(feat, 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}


class SigLIP2_So400m_378_DroneIQA_MT(nn.Module):
    """Multi-task SigLIP2 ViT-SO400M/14 378px — 3 quality heads for DroneIQA.
    patch_size=14, embed_dim=1152.  Supports higher resolution via --resize/--crop_size
    (e.g. 518=37*14, 630=45*14)."""

    def __init__(self):
        super().__init__()
        model, _ = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP2-378')
        self.feature_extraction = model.visual
        self.feature_extraction.trunk.dynamic_img_size = True
        self.feature_extraction.trunk.patch_embed.strict_img_size = False
        self.feature_extraction.trunk.patch_embed.flatten = False
        self.feature_extraction.trunk.patch_embed.output_fmt = TimmFormat('NHWC')

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(1152, 128),
                nn.Linear(128, 1),
            )
            for name in TASK_NAMES
        })

    def forward(self, x):
        feat = self.feature_extraction(x)
        feat = torch.flatten(feat, 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}


SUPPORTED_MODELS = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   SigLIP2_ViTL_384_DroneIQA_MT,
    'SigLIP2_So400m_378_DroneIQA_MT': SigLIP2_So400m_378_DroneIQA_MT,
}


# ────────────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, n_samples):
    """Evaluate on the val set; returns per-task metrics."""
    model.eval()
    labels  = {t: np.zeros(n_samples) for t in TASK_NAMES}
    outputs = {t: np.zeros(n_samples) for t in TASK_NAMES}

    for i, (images, label_dict, _) in enumerate(loader):
        images = images.to(device)
        preds = model(images)
        for t in TASK_NAMES:
            labels[t][i]  = label_dict[t].item()
            outputs[t][i] = preds[t].item()

    metrics = {}
    for t in TASK_NAMES:
        plcc, srcc, krcc, rmse = performance_fit(labels[t], outputs[t])
        metrics[t] = dict(plcc=plcc, srcc=srcc, krcc=krcc, rmse=rmse)

    return metrics, labels, outputs


# ────────────────────────────────────────────────────────────────────────────
# Single experiment (isolated function so all locals are freed on return)
# ────────────────────────────────────────────────────────────────────────────
def run_experiment(config, exp_i):
    """Run one train/val split and return best_metrics dict."""
    seed = exp_i + config.random_seed
    print(f'\n{"="*72}')
    print(f'Experiment {exp_i}  (seed={seed})')
    print(f'{"="*72}')

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Model ─────────────────────────────────────────────────────── #
    assert config.model_name in SUPPORTED_MODELS, \
        f'Unknown model: {config.model_name}, choose from {list(SUPPORTED_MODELS)}'
    model = SUPPORTED_MODELS[config.model_name]()
    if config.multi_gpu:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    model = model.to(device)

    n_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    print(f'Model: {config.model_name}  |  Params: {n_params / 1e6:.2f} M')

    # ── Transforms ────────────────────────────────────────────────── #
    norm_mean, norm_std = MODEL_NORM_STATS[config.model_name]

    default_resize, default_crop = MODEL_DEFAULT_SIZES.get(
        config.model_name, (config.resize, config.crop_size))
    eff_resize    = config.resize    if config.resize    != 432 \
                    else default_resize
    eff_crop_size = config.crop_size if config.crop_size != 384 \
                    else default_crop
    print(f'Transforms: resize={eff_resize}  crop={eff_crop_size}  '
          f'norm={norm_mean}')

    tf_train = transforms.Compose([
        transforms.Resize(eff_resize),
        transforms.RandomCrop(eff_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(eff_resize),
        transforms.CenterCrop(eff_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    # ── Datasets & loaders ────────────────────────────────────────── #
    trainset = DroneIQADataset(config.img_dir, config.csv_path,
                               'train', tf_train, seed=seed)
    valset   = DroneIQADataset(config.img_dir, config.csv_path,
                               'val',   tf_val,  seed=seed)

    print(f'Train: {len(trainset)} images  |  Val: {len(valset)} images')

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ── Optimiser / scheduler ─────────────────────────────────────── #
    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr, weight_decay=1e-7)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.decay_interval,
        gamma=config.decay_ratio,
    )

    # ── Training loop ─────────────────────────────────────────────── #
    best_avg_score = -1.0
    best_metrics   = None
    old_ckpt       = None
    old_mat        = None

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        step_losses  = []
        t0 = time.time()

        for step, (images, label_dict, _) in enumerate(train_loader):
            images = images.to(device)
            preds = model(images)

            loss = torch.tensor(0.0, device=device)
            for t in TASK_NAMES:
                gt = label_dict[t].to(device).float()
                loss = loss + plcc_loss(gt, preds[t])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            step_losses.append(loss.item())

            if config.print_steps > 0 and \
                    (step + 1) % config.print_steps == 0:
                print(f'  [{epoch+1}/{config.epochs}] '
                      f'step {step+1}/{len(train_loader)}  '
                      f'loss={sum(step_losses)/len(step_losses):.4f}  '
                      f'({time.time()-t0:.1f}s)')
                step_losses = []
                t0 = time.time()

        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f'Epoch {epoch+1:>3}  avg_loss={avg_loss:.4f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

        # ── Evaluation ────────────────────────────────────────────── #
        metrics, lbl, pred = evaluate(model, val_loader, device,
                                      len(valset))

        print(f'         val results:')
        for t in TASK_NAMES:
            m = metrics[t]
            print(f'           {t:>25s}  SRCC={m["srcc"]:.4f}  '
                  f'KRCC={m["krcc"]:.4f}  PLCC={m["plcc"]:.4f}  '
                  f'RMSE={m["rmse"]:.4f}')

        avg_srcc = np.mean([metrics[t]['srcc'] for t in TASK_NAMES])
        avg_plcc = np.mean([metrics[t]['plcc'] for t in TASK_NAMES])
        avg_score = (avg_srcc + avg_plcc) / 2.0
        print(f'           {"avg":>25s}  SRCC={avg_srcc:.4f}  '
              f'PLCC={avg_plcc:.4f}  score={avg_score:.4f}')

        gq_srcc = metrics['global_quality_mean']['srcc']
        gq_plcc = metrics['global_quality_mean']['plcc']
        gq_score = (gq_srcc + gq_plcc) / 2.0
        print(f'           {"global_quality score":>25s}  '
              f'(SRCC+PLCC)/2={gq_score:.4f}')

        # ── Save best (by global_quality_mean (SRCC+PLCC)/2) ───── #
        if gq_score > best_avg_score:
            best_avg_score = gq_score
            best_metrics   = metrics
            print(f'         ** new best  global_quality (SRCC+PLCC)/2={gq_score:.4f} **')

            os.makedirs(config.ckpt_path, exist_ok=True)
            ckpt_name = os.path.join(
                config.ckpt_path,
                f'{config.model_name}_DroneIQA_v{exp_i}_ep{epoch+1}'
                f'_SRCC{gq_srcc:.4f}.pth')
            mat_name = ckpt_name.replace('.pth', '.mat')

            if old_ckpt and os.path.exists(old_ckpt):
                os.remove(old_ckpt)
            if old_mat and os.path.exists(old_mat):
                os.remove(old_mat)

            raw = model.module if isinstance(model, nn.DataParallel) \
                  else model
            torch.save(raw.state_dict(), ckpt_name)

            mat_data = {}
            for t in TASK_NAMES:
                mat_data[f'{t}_pred']  = pred[t]
                mat_data[f'{t}_label'] = lbl[t]
            scio.savemat(mat_name, mat_data)

            old_ckpt = ckpt_name
            old_mat  = mat_name

    # ── Per-experiment summary ──────────────────────────────────── #
    gq_best = best_metrics['global_quality_mean']
    gq_best_score = (gq_best['srcc'] + gq_best['plcc']) / 2.0
    print(f'\nExp {exp_i} best val results:')
    for t in TASK_NAMES:
        m = best_metrics[t]
        print(f'  {t:>25s}  SRCC={m["srcc"]:.4f}  KRCC={m["krcc"]:.4f}  '
              f'PLCC={m["plcc"]:.4f}  RMSE={m["rmse"]:.4f}')
    print(f'  {"global_quality (SRCC+PLCC)/2":>25s}  = {gq_best_score:.4f}')

    return best_metrics


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main(config):
    all_results = {t: dict(srcc=[], krcc=[], plcc=[], rmse=[]) for t in TASK_NAMES}

    for exp_i in range(config.n_exp):
        best_metrics = run_experiment(config, exp_i)

        gc.collect()
        torch.cuda.empty_cache()

        for t in TASK_NAMES:
            m = best_metrics[t]
            all_results[t]['srcc'].append(m['srcc'])
            all_results[t]['krcc'].append(m['krcc'])
            all_results[t]['plcc'].append(m['plcc'])
            all_results[t]['rmse'].append(m['rmse'])

    # ── Overall summary ─────────────────────────────────────────────── #
    print(f'\n{"="*72}')
    print(f'DroneIQA Multi-task results  ({config.n_exp} splits)')
    print(f'{"="*72}')

    for t in TASK_NAMES:
        r = all_results[t]
        print(f'\n{"─"*72}')
        print(f'  Task: {t}')
        print(f'  {"Split":>5}  {"SRCC":>8}  {"KRCC":>8}  {"PLCC":>8}  {"RMSE":>8}')
        print(f'{"─"*72}')
        for i in range(len(r['srcc'])):
            print(f'  {i:>5d}  {r["srcc"][i]:>8.4f}  {r["krcc"][i]:>8.4f}  '
                  f'{r["plcc"][i]:>8.4f}  {r["rmse"][i]:>8.4f}')
        print(f'{"─"*72}')
        for label, fn in [('Mean', np.mean), ('Std', np.std), ('Median', np.median)]:
            print(f'  {label:>7}  {fn(r["srcc"]):>8.4f}  {fn(r["krcc"]):>8.4f}  '
                  f'{fn(r["plcc"]):>8.4f}  {fn(r["rmse"]):>8.4f}')

    # Average across all tasks
    print(f'\n{"="*72}')
    print(f'  Average across all tasks')
    print(f'  {"Metric":>10}  {"SRCC":>8}  {"KRCC":>8}  {"PLCC":>8}  {"RMSE":>8}')
    print(f'{"─"*72}')
    avg_srcc_all = np.mean([np.mean(all_results[t]['srcc']) for t in TASK_NAMES])
    avg_krcc_all = np.mean([np.mean(all_results[t]['krcc']) for t in TASK_NAMES])
    avg_plcc_all = np.mean([np.mean(all_results[t]['plcc']) for t in TASK_NAMES])
    avg_rmse_all = np.mean([np.mean(all_results[t]['rmse']) for t in TASK_NAMES])
    print(f'  {"Mean":>10}  {avg_srcc_all:>8.4f}  {avg_krcc_all:>8.4f}  '
          f'{avg_plcc_all:>8.4f}  {avg_rmse_all:>8.4f}')

    # global_quality_mean (SRCC+PLCC)/2 statistics
    gq = all_results['global_quality_mean']
    gq_scores = [(s + p) / 2.0
                 for s, p in zip(gq['srcc'], gq['plcc'])]
    print(f'\n{"─"*72}')
    print(f'  global_quality_mean (SRCC+PLCC)/2 per split:')
    print(f'  {"Split":>5}  {"(SRCC+PLCC)/2":>14}')
    print(f'{"─"*72}')
    for i, sc in enumerate(gq_scores):
        print(f'  {i:>5d}  {sc:>14.4f}')
    print(f'{"─"*72}')
    print(f'  {"Mean":>7}  {np.mean(gq_scores):>14.4f}')
    print(f'  {"Std":>7}  {np.std(gq_scores):>14.4f}')
    print(f'  {"Median":>7}  {np.median(gq_scores):>14.4f}')
    print(f'{"="*72}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-task IQA training — DroneIQA dataset')

    # Data
    parser.add_argument('--img_dir', type=str,
                        default='/root/autodl-tmp/DroneIQA/train')
    parser.add_argument('--csv_path', type=str,
                        default='train.csv')

    # Model
    parser.add_argument('--model_name', type=str,
                        default='SigLIP2_ViTL_384_DroneIQA_MT',
                        choices=list(SUPPORTED_MODELS.keys()))

    # Hyper-parameters
    parser.add_argument('--lr',              type=float, default=1e-5)
    parser.add_argument('--decay_ratio',     type=float, default=0.95)
    parser.add_argument('--decay_interval',  type=int,   default=2)
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--train_batch_size',type=int,   default=8)
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--n_exp',           type=int,   default=10)
    parser.add_argument('--random_seed',     type=int,   default=0)

    # Image size (SigLIP2 384 defaults)
    parser.add_argument('--resize',    type=int, default=432)
    parser.add_argument('--crop_size', type=int, default=384)

    # Checkpoint
    parser.add_argument('--ckpt_path', type=str, default='/root/autodl-tmp/IQAModels/ckpts_drone_iqa_mt')

    # Logging
    parser.add_argument('--print_steps', type=int, default=20,
                        help='Print loss every N steps (0 = silent)')

    # Multi-GPU
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None)

    config = parser.parse_args()
    main(config)
