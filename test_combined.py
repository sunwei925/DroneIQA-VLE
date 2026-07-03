# -*- coding: utf-8 -*-
"""
Combined DroneIQA inference — Qwen3.5-9B + SigLIP2 ensemble.

Runs both pipelines independently, then averages their per-image
global_quality_mean predictions and saves to a single CSV.

Usage
-----
python test_combined.py
python test_combined.py --val_dir /path/to/val --output_csv combined_result.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from open_clip import create_model
from timm.layers import Format as TimmFormat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ════════════════════════════════════════════════════════════════════════════
#  Shared
# ════════════════════════════════════════════════════════════════════════════

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logistic_part = 1 + np.exp(-(X - bayta3) / np.abs(bayta4))
    return bayta2 + (bayta1 - bayta2) / logistic_part


# ════════════════════════════════════════════════════════════════════════════
#  Part A — Qwen3.5-9B pipeline (from test_val.py)
# ════════════════════════════════════════════════════════════════════════════

QWEN_ADAPTER_PATH = os.path.join(
    BASE_DIR, "output_droneIQA/v12-20260408-234616/checkpoint-675")
QWEN_POPT_PATH = os.path.join(BASE_DIR, "popt.mat")

QWEN_ENV_OVERRIDES = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HOME": os.path.join(BASE_DIR, "hf_cache"),
    "IMAGE_MAX_TOKEN_NUM": "2048",
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": BASE_DIR,
}

QWEN_SYSTEM_PROMPT = "You are doing the image quality assessment task for UAV images."
QWEN_USER_PROMPT = (
    "Please evaluate the global quality, target quality, and background quality "
    "of the UAV image [<image>]. Global quality is the overall visual quality of "
    "the image, target quality is the visual quality of the target object framed "
    "in the image, and background quality is the background quality of the image."
)


def _qwen_generate_jsonl(image_dir, out_path):
    filenames = sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('._')
    )
    with open(out_path, 'w') as fout:
        for fn in filenames:
            record = {
                'messages': [
                    {'role': 'system', 'content': QWEN_SYSTEM_PROMPT},
                    {'role': 'user', 'content': QWEN_USER_PROMPT},
                ],
                'images': [os.path.join(image_dir, fn)],
                'label': [0.0, 0.0, 0.0],
            }
            fout.write(json.dumps(record) + '\n')
    print(f'[Qwen] Generated {len(filenames)} entries -> {out_path}')


def _qwen_run_infer(val_dataset, result_path):
    cmd = [
        sys.executable, os.path.join(BASE_DIR, "swift/cli/infer.py"),
        "--model", os.path.join(BASE_DIR, "output_droneIQA/Qwen3.5-9B"),
        "--adapters", QWEN_ADAPTER_PATH,
        "--infer_backend", "transformers",
        "--task_type", "seq_cls",
        "--problem_type", "regression",
        "--num_labels", "3",
        "--max_batch_size", "64",
        "--temperature", "0",
        "--stream", "false",
        "--result_path", result_path,
        "--val_dataset", val_dataset,
        "--model_kwargs", '{"tie_word_embeddings": false}',
    ]
    env = {**os.environ, **QWEN_ENV_OVERRIDES}
    print(f'[Qwen] Running inference ...')
    subprocess.run(cmd, env=env, check=True)


def _qwen_parse_result_jsonl(result_path):
    rows = []
    with open(result_path) as f:
        for line in f:
            obj = json.loads(line)
            img_path = obj['images'][0]['path']
            filename = os.path.basename(img_path)
            global_quality_mean = obj['response'][0]
            rows.append((filename, global_quality_mean))
    rows.sort(key=lambda x: x[0])
    return rows


def run_qwen_pipeline(val_dir):
    """Returns dict {filename: mapped_global_quality_mean}."""
    popt = scio.loadmat(QWEN_POPT_PATH)['popt'].flatten()
    print(f'[Qwen] Loaded popt: {popt}')

    with tempfile.TemporaryDirectory(dir=BASE_DIR) as tmpdir:
        val_jsonl = os.path.join(tmpdir, 'val_dataset.jsonl')
        val_result = os.path.join(tmpdir, 'val_result.jsonl')

        _qwen_generate_jsonl(val_dir, val_jsonl)
        _qwen_run_infer(val_jsonl, val_result)
        val_rows = _qwen_parse_result_jsonl(val_result)

    df = pd.DataFrame(val_rows, columns=['filename', 'global_quality_mean'])
    df['global_quality_mean'] = logistic_func(
        df['global_quality_mean'].values, *popt)

    result = dict(zip(df['filename'], df['global_quality_mean']))
    print(f'[Qwen] Got predictions for {len(result)} images  '
          f'range=[{min(result.values()):.4f}, {max(result.values()):.4f}]')
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Part B — SigLIP2 ensemble pipeline (from test_SigLIP.py)
# ════════════════════════════════════════════════════════════════════════════

TASK_NAMES = ['global_quality_mean', 'target_quality_mean',
              'background_quality_mean']

_SIGLIP_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_STD  = [0.5, 0.5, 0.5]

MODEL_NORM_STATS = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (_SIGLIP_MEAN, _SIGLIP_STD),
    'SigLIP2_So400m_378_DroneIQA_MT': (_SIGLIP_MEAN, _SIGLIP_STD),
}

MODEL_DEFAULT_SIZES = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (432, 384),
    'SigLIP2_So400m_378_DroneIQA_MT': (432, 378),
}


class SigLIP2_ViTL_384_DroneIQA_MT(nn.Module):
    def __init__(self):
        super().__init__()
        model = create_model('ViT-L-16-SigLIP2-384')
        self.feature_extraction = model.visual
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(1024, 128), nn.Linear(128, 1))
            for name in TASK_NAMES
        })

    def forward(self, x):
        feat = torch.flatten(self.feature_extraction(x), 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}


class SigLIP2_So400m_378_DroneIQA_MT(nn.Module):
    def __init__(self):
        super().__init__()
        model = create_model('ViT-SO400M-14-SigLIP2-378')
        self.feature_extraction = model.visual
        self.feature_extraction.trunk.dynamic_img_size = True
        self.feature_extraction.trunk.patch_embed.strict_img_size = False
        self.feature_extraction.trunk.patch_embed.flatten = False
        self.feature_extraction.trunk.patch_embed.output_fmt = TimmFormat('NHWC')
        self.heads = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(1152, 128), nn.Linear(128, 1))
            for name in TASK_NAMES
        })

    def forward(self, x):
        feat = torch.flatten(self.feature_extraction(x), 1)
        return {name: head(feat).squeeze(1) for name, head in self.heads.items()}


SUPPORTED_MODELS = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   SigLIP2_ViTL_384_DroneIQA_MT,
    'SigLIP2_So400m_378_DroneIQA_MT': SigLIP2_So400m_378_DroneIQA_MT,
}

MODEL_CHECKPOINTS = {
    'SigLIP2_ViTL_384_DroneIQA_MT': [
        ('SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v0_ep8_SRCC0.9271.pth',
         'SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v0_ep8_SRCC0.9271.mat'),
        ('SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v1_ep7_SRCC0.9233.pth',
         'SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v1_ep7_SRCC0.9233.mat'),
        ('SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v2_ep6_SRCC0.9312.pth',
         'SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v2_ep6_SRCC0.9312.mat'),
    ],
    'SigLIP2_So400m_378_DroneIQA_MT': [
        ('SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v0_ep12_SRCC0.9274.pth',
         'SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v0_ep12_SRCC0.9274.mat'),
        ('SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v1_ep10_SRCC0.9259.pth',
         'SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v1_ep10_SRCC0.9259.mat'),
        ('SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v2_ep5_SRCC0.9268.pth',
         'SigLIP2_So400m_378_DroneIQA_MT_DroneIQA_v2_ep5_SRCC0.9268.mat'),
    ],
}

MODEL_NAMES = list(MODEL_CHECKPOINTS.keys())


class DroneIQAValDataset(Dataset):
    def __init__(self, val_dir, transform):
        self.val_dir   = val_dir
        self.transform = transform
        self.filenames = sorted(
            f for f in os.listdir(val_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        )
        if not self.filenames:
            raise RuntimeError(f'No images found in {val_dir}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.val_dir, filename)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, filename


def _siglip_fit_mapping_params(y_output, y_label):
    y_output = y_output.astype(np.float64)
    y_label  = y_label.astype(np.float64)
    beta0 = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label,
                        p0=beta0, maxfev=100_000_000)
    return popt


def _siglip_fit_all_task_mappings(mat_path):
    data = scio.loadmat(mat_path)
    mapping = {}
    for t in TASK_NAMES:
        pred_key  = f'{t}_pred'
        label_key = f'{t}_label'
        if pred_key not in data or label_key not in data:
            print(f'  [Warning] {mat_path} missing {pred_key}/{label_key}, '
                  f'skipping task {t}')
            continue
        y_output = data[pred_key].flatten()
        y_label  = data[label_key].flatten()
        mapping[t] = _siglip_fit_mapping_params(y_output, y_label)
    return mapping


def _siglip_load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ckpt['state_dict']
             if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)
    return {k.replace('module.', '', 1): v for k, v in state.items()}


@torch.no_grad()
def _siglip_run_single_split(model_name, pth_path, mat_path, loader, device):
    model = SUPPORTED_MODELS[model_name]()
    state_dict = _siglip_load_checkpoint(pth_path, device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    task_mappings = _siglip_fit_all_task_mappings(mat_path)

    preds = {}
    for images, filenames in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        for j, fname in enumerate(filenames):
            mapped = {}
            for t in TASK_NAMES:
                raw = outputs[t][j].detach().cpu().item()
                if t in task_mappings:
                    mapped[t] = float(
                        logistic_func(np.array([raw]), *task_mappings[t])[0])
                else:
                    mapped[t] = raw
            preds[fname] = mapped

    del model
    torch.cuda.empty_cache()
    return preds


def _siglip_build_loader(val_dir, model_name, batch_size, num_workers):
    norm_mean, norm_std = MODEL_NORM_STATS[model_name]
    default_resize, default_crop = MODEL_DEFAULT_SIZES[model_name]

    tf_test = transforms.Compose([
        transforms.Resize(default_resize),
        transforms.CenterCrop(default_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    dataset = DroneIQAValDataset(val_dir, tf_test)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataset, loader


def run_siglip_pipeline(val_dir, ckpt_dir, batch_size=16, num_workers=4):
    """Returns dict {filename: averaged_global_quality_mean}."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_preds = {}
    total_splits = 0

    for model_name in MODEL_NAMES:
        print(f'\n[SigLIP] {"="*50}')
        print(f'[SigLIP] Model: {model_name}')

        dataset, loader = _siglip_build_loader(
            val_dir, model_name, batch_size, num_workers)
        print(f'[SigLIP] Val images: {len(dataset)}')

        ckpt_list = MODEL_CHECKPOINTS[model_name]
        model_split_preds = {}
        for split_idx, (pth_name, mat_name) in enumerate(ckpt_list):
            pth_path = os.path.join(ckpt_dir, pth_name)
            mat_path = os.path.join(ckpt_dir, mat_name)
            print(f'[SigLIP] Split {split_idx}: {pth_name}')
            preds = _siglip_run_single_split(
                model_name, pth_path, mat_path, loader, device)
            model_split_preds[split_idx] = preds
            total_splits += 1

        all_preds[model_name] = model_split_preds

    first_model = next(iter(all_preds.values()))
    filenames = sorted(next(iter(first_model.values())).keys())

    model_avgs = {}
    for model_name, split_preds in all_preds.items():
        avgs = {}
        for fname in filenames:
            vals = [
                split_preds[i][fname]['global_quality_mean']
                for i in sorted(split_preds)
                if fname in split_preds[i]
            ]
            avgs[fname] = float(np.mean(vals))
        model_avgs[model_name] = avgs

    result = {}
    for fname in filenames:
        grand_vals = [model_avgs[m][fname]
                      for m in MODEL_NAMES if m in model_avgs]
        result[fname] = float(np.mean(grand_vals))

    print(f'[SigLIP] Got predictions for {len(result)} images  '
          f'range=[{min(result.values()):.4f}, {max(result.values()):.4f}]')
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Combined: average both pipelines and save
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    print('=' * 60)
    print('  Combined DroneIQA — Qwen + SigLIP2 ensemble')
    print('=' * 60)

    # --- Part A: Qwen ---
    print('\n>>> Running Qwen3.5-9B pipeline ...')
    qwen_preds = run_qwen_pipeline(args.val_dir)

    # --- Part B: SigLIP2 ---
    print('\n>>> Running SigLIP2 ensemble pipeline ...')
    siglip_preds = run_siglip_pipeline(
        args.val_dir, args.ckpt_dir, args.batch_size, args.num_workers)

    # --- Merge & average ---
    all_filenames = sorted(set(qwen_preds.keys()) | set(siglip_preds.keys()))
    print(f'\n>>> Merging results for {len(all_filenames)} images ...')

    rows = []
    for fname in all_filenames:
        vals = []
        if fname in qwen_preds:
            vals.append(qwen_preds[fname])
        if fname in siglip_preds:
            vals.append(siglip_preds[fname])
        avg_gq = float(np.mean(vals))
        if args.round is not None:
            avg_gq = round(avg_gq, args.round)
        rows.append({'filename': fname, 'global_quality_mean': avg_gq})

    # --- Save CSV ---
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'global_quality_mean'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n[Output] wrote {len(rows)} rows to {args.output_csv}')

    # --- Preview ---
    preview = rows[:5]
    print(f'\nPreview (first 5):')
    print(f'  {"filename":>16s}  {"Qwen":>10s}  {"SigLIP":>10s}  {"AVG":>10s}')
    for r in preview:
        fname = r['filename']
        q = qwen_preds.get(fname, float('nan'))
        s = siglip_preds.get(fname, float('nan'))
        a = r['global_quality_mean']
        print(f'  {fname:>16s}  {q:>10.4f}  {s:>10.4f}  {a:>10.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combined DroneIQA — Qwen3.5-9B + SigLIP2 ensemble average')

    parser.add_argument('--ckpt_dir', type=str, default='output_droneIQA/ckpts_SigCLIP',
                        help='SigLIP checkpoint directory')
    parser.add_argument('--val_dir', type=str,
                        default='/root/autodl-tmp/DroneIQA/val',
                        help='Validation image directory')
    parser.add_argument('--output_csv', type=str,
                        default='droneIQA_combined_result.csv',
                        help='Output CSV path')
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--round', type=int, default=4,
                        help='Decimal places for rounding (None to disable)')

    main(parser.parse_args())
