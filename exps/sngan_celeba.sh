#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python train.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset celeba \
--img_size 64 \
--max_iter 50000 \
--model sngan_celeba64 \
--latent_dim 100 \
--gf_dim 128 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 5 \
--val_freq 5 \
--exp_name sngan_64 \
--data_path ./dataset \
--test_data_path ./testdata