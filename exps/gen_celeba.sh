#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

rm -rf out/*

python generate.py \
--random_seed 1234 \
--load_path ./logs/sngan_64_2022_10_28_01_23_45/Model/checkpoint_10.pth \
--out_path ./out \
--latent_dim 100 \
--img_size 64 \
--gf_dim 128 \
--df_dim 128 \
--model sngan_celeba64 \
--exp_name celba1
