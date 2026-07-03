# -*- coding: utf-8 -*-
"""
Single-image SigLIP2 DroneIQA inference (no ensemble).

Loads ONE trained SigLIP2 checkpoint (.pth) + its logistic mapping (.mat),
scores ONE image, and prints the global / target / background quality scores.

Usage
-----
python test_siglip_one_image.py \
    --image_path /path/to/img.jpg \
    --model SigLIP2_ViTL_384_DroneIQA_MT \
    --ckpt_path output_droneIQA/ckpts_SigCLIP/SigLIP2_ViTL_384_DroneIQA_MT_DroneIQA_v0_ep8_SRCC0.9271.pth

# Explicitly specify the .mat (defaults to the .pth stem with .mat)
python test_siglip_one_image.py \
    --image_path /path/to/img.jpg \
    --model SigLIP2_So400m_378_DroneIQA_MT \
    --ckpt_path path/to/xxx.pth \
    --mat_path path/to/xxx.mat
"""

import argparse
import os

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import curve_fit
from torchvision import transforms
from open_clip import create_model
from timm.layers import Format as TimmFormat

TASK_NAMES = ['global_quality_mean', 'target_quality_mean',
              'background_quality_mean']

_SIGLIP_MEAN = [0.5, 0.5, 0.5]
_SIGLIP_STD = [0.5, 0.5, 0.5]

MODEL_NORM_STATS = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (_SIGLIP_MEAN, _SIGLIP_STD),
    'SigLIP2_So400m_378_DroneIQA_MT': (_SIGLIP_MEAN, _SIGLIP_STD),
}

MODEL_DEFAULT_SIZES = {
    'SigLIP2_ViTL_384_DroneIQA_MT':   (432, 384),
    'SigLIP2_So400m_378_DroneIQA_MT': (432, 378),
}


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logistic_part = 1 + np.exp(-(X - bayta3) / np.abs(bayta4))
    return bayta2 + (bayta1 - bayta2) / logistic_part


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


def _fit_mapping_params(y_output, y_label):
    y_output = y_output.astype(np.float64)
    y_label = y_label.astype(np.float64)
    beta0 = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label,
                        p0=beta0, maxfev=100_000_000)
    return popt


def fit_all_task_mappings(mat_path):
    data = scio.loadmat(mat_path)
    mapping = {}
    for t in TASK_NAMES:
        pred_key = f'{t}_pred'
        label_key = f'{t}_label'
        if pred_key not in data or label_key not in data:
            print(f'  [Warning] {mat_path} missing {pred_key}/{label_key}, '
                  f'skipping task {t}')
            continue
        y_output = data[pred_key].flatten()
        y_label = data[label_key].flatten()
        mapping[t] = _fit_mapping_params(y_output, y_label)
    return mapping


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = (ckpt['state_dict']
             if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt)
    return {k.replace('module.', '', 1): v for k, v in state.items()}


def build_transform(model_name):
    norm_mean, norm_std = MODEL_NORM_STATS[model_name]
    default_resize, default_crop = MODEL_DEFAULT_SIZES[model_name]
    return transforms.Compose([
        transforms.Resize(default_resize),
        transforms.CenterCrop(default_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])


@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mat_path = args.mat_path
    if mat_path is None:
        mat_path = os.path.splitext(args.ckpt_path)[0] + '.mat'

    print(f'[SigLIP] Model     : {args.model}')
    print(f'[SigLIP] Checkpoint: {args.ckpt_path}')
    print(f'[SigLIP] Mapping   : {mat_path}')
    print(f'[SigLIP] Device    : {device}')

    model = SUPPORTED_MODELS[args.model]()
    state_dict = load_checkpoint(args.ckpt_path, device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    task_mappings = fit_all_task_mappings(mat_path)

    transform = build_transform(args.model)
    image = Image.open(args.image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    outputs = model(image)

    scores = {}
    for t in TASK_NAMES:
        raw = outputs[t][0].detach().cpu().item()
        if t in task_mappings:
            value = float(logistic_func(np.array([raw]), *task_mappings[t])[0])
        else:
            value = raw
        if args.round is not None:
            value = round(value, args.round)
        scores[t] = value

    print('\n' + '=' * 50)
    print(f'  Image: {args.image_path}')
    print('=' * 50)
    print(f'  global_quality_mean     : {scores["global_quality_mean"]}')
    print(f'  target_quality_mean     : {scores["target_quality_mean"]}')
    print(f'  background_quality_mean : {scores["background_quality_mean"]}')
    print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single-image SigLIP2 DroneIQA inference (one checkpoint)')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to a single image')
    parser.add_argument('--model', type=str, default='SigLIP2_ViTL_384_DroneIQA_MT',
                        choices=list(SUPPORTED_MODELS.keys()),
                        help='SigLIP2 architecture')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to a single trained .pth checkpoint')
    parser.add_argument('--mat_path', type=str, default=None,
                        help='Path to the .mat for logistic mapping '
                             '(defaults to the .pth stem with .mat)')
    parser.add_argument('--round', type=int, default=4,
                        help='Decimal places for rounding (None to disable)')

    main(parser.parse_args())
