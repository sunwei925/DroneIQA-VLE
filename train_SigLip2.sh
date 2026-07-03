CUDA_VISIBLE_DEVICES=0 python -u train_SigLip2.py \
    --model_name SigLIP2_ViTL_384_DroneIQA_MT \
    --multi_gpu \
    --epochs 10 \
    --train_batch_size 32 \
    --n_exp 3 \
    --lr 1e-5 \
    >> train_SigLIP2_ViTL_384_DroneIQA_MT_3splits.log



CUDA_VISIBLE_DEVICES=0 python -u train_SigLip2.py \
    --model_name SigLIP2_So400m_378_DroneIQA_MT \
    --multi_gpu \
    --epochs 10 \
    --train_batch_size 32 \
    --n_exp 3 \
    --lr 1e-5 \
    >> train_SigLIP2_So400m_378_DroneIQA_MT_3splits.log