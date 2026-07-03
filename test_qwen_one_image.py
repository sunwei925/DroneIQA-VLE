# -*- coding: utf-8 -*-
"""
Single-image Qwen3.5-9B DroneIQA inference.

Runs the LoRA-adapted Qwen3.5-9B pipeline on ONE image and prints the
global / target / background quality scores. The global quality score is
mapped onto the MOS scale by the logistic parameters stored in popt.mat.

Usage
-----
python test_qwen_one_image.py --image_path /path/to/img.jpg

python test_qwen_one_image.py \
    --image_path /path/to/img.jpg \
    --adapter_path output_droneIQA/v12-20260408-234616/checkpoint-675 \
    --popt_path popt.mat
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import scipy.io as scio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ADAPTER_PATH = os.path.join(
    BASE_DIR, "output_droneIQA/v12-20260408-234616/checkpoint-675")
DEFAULT_POPT_PATH = os.path.join(BASE_DIR, "popt.mat")
DEFAULT_BASE_MODEL = os.path.join(BASE_DIR, "output_droneIQA/Qwen3.5-9B")

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

TASK_NAMES = ['global_quality_mean', 'target_quality_mean',
              'background_quality_mean']


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logistic_part = 1 + np.exp(-(X - bayta3) / np.abs(bayta4))
    return bayta2 + (bayta1 - bayta2) / logistic_part


def _generate_jsonl(image_path, out_path):
    record = {
        'messages': [
            {'role': 'system', 'content': QWEN_SYSTEM_PROMPT},
            {'role': 'user', 'content': QWEN_USER_PROMPT},
        ],
        'images': [os.path.abspath(image_path)],
        'label': [0.0, 0.0, 0.0],
    }
    with open(out_path, 'w') as fout:
        fout.write(json.dumps(record) + '\n')


def _run_infer(base_model, adapter_path, val_dataset, result_path):
    cmd = [
        sys.executable, os.path.join(BASE_DIR, "swift/cli/infer.py"),
        "--model", base_model,
        "--adapters", adapter_path,
        "--infer_backend", "transformers",
        "--task_type", "seq_cls",
        "--problem_type", "regression",
        "--num_labels", "3",
        "--max_batch_size", "1",
        "--temperature", "0",
        "--stream", "false",
        "--result_path", result_path,
        "--val_dataset", val_dataset,
        "--model_kwargs", '{"tie_word_embeddings": false}',
    ]
    env = {**os.environ, **QWEN_ENV_OVERRIDES}
    print('[Qwen] Running inference ...')
    subprocess.run(cmd, env=env, check=True)


def _parse_result_jsonl(result_path):
    with open(result_path) as f:
        line = f.readline()
    obj = json.loads(line)
    return obj['response']


def main(args):
    popt = scio.loadmat(args.popt_path)['popt'].flatten()

    print(f'[Qwen] Base model : {args.base_model}')
    print(f'[Qwen] Adapter    : {args.adapter_path}')
    print(f'[Qwen] popt       : {popt}')

    with tempfile.TemporaryDirectory(dir=BASE_DIR) as tmpdir:
        val_jsonl = os.path.join(tmpdir, 'val_dataset.jsonl')
        val_result = os.path.join(tmpdir, 'val_result.jsonl')

        _generate_jsonl(args.image_path, val_jsonl)
        _run_infer(args.base_model, args.adapter_path, val_jsonl, val_result)
        response = _parse_result_jsonl(val_result)

    raw = {t: float(response[i]) for i, t in enumerate(TASK_NAMES)}
    global_mapped = float(logistic_func(
        np.array([raw['global_quality_mean']]), *popt)[0])

    scores = {
        'global_quality_mean': global_mapped,
        'target_quality_mean': raw['target_quality_mean'],
        'background_quality_mean': raw['background_quality_mean'],
    }
    if args.round is not None:
        scores = {k: round(v, args.round) for k, v in scores.items()}

    print('\n' + '=' * 50)
    print(f'  Image: {args.image_path}')
    print('=' * 50)
    print(f'  global_quality_mean     : {scores["global_quality_mean"]}  '
          f'(logistic-mapped)')
    print(f'  target_quality_mean     : {scores["target_quality_mean"]}  (raw)')
    print(f'  background_quality_mean : {scores["background_quality_mean"]}  (raw)')
    print('=' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single-image Qwen3.5-9B DroneIQA inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to a single image')
    parser.add_argument('--base_model', type=str, default=DEFAULT_BASE_MODEL,
                        help='Qwen3.5-9B base model directory')
    parser.add_argument('--adapter_path', type=str, default=DEFAULT_ADAPTER_PATH,
                        help='Qwen LoRA adapter directory')
    parser.add_argument('--popt_path', type=str, default=DEFAULT_POPT_PATH,
                        help='Path to popt.mat (Qwen logistic mapping params)')
    parser.add_argument('--round', type=int, default=4,
                        help='Decimal places for rounding (None to disable)')

    main(parser.parse_args())
