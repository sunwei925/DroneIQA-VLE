#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$(pwd)/hf_cache

CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=2048 \
PYTHONPATH=$(pwd) \
python $(pwd)/swift/cli/sft.py \
  --model Qwen/Qwen3.5-9B \
  --dataset $(pwd)/DroneIQA.jsonl \
  --task_type seq_cls \
  --add_non_thinking_prefix true \
  --num_labels 3 \
  --problem_type regression \
  --tuner_type lora \
  --target_modules all-linear \
  --freeze_vit false \
  --freeze_aligner false \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --gradient_checkpointing true \
  --max_length 4096 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --lora_rank 64 \
  --lora_alpha 128 \
  --warmup_ratio 0.05 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 8 \
  --logging_steps 5 \
  --save_steps 500 \
  --save_total_limit 1 \
  --output_dir $(pwd)/output_droneIQA \
  --model_kwargs '{"tie_word_embeddings": false}' \
  --loss_type mixed