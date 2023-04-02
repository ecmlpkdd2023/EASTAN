#!/usr/bin/env bash
# sh scripts for EAST

# EAST*-PEMS08
python3 train.py --dataset PEMS08 --K 4 --max_epoch 1000 --batch_size 32 --node_num 170 --learning_rate 0.001 --patience 20 --emb_dim 16 --encoder_layer 1 --decoder_layer 1

# EAST-PEMS04
python3 train.py --dataset PEMS04 --K 4 --max_epoch 1000 --batch_size 32 --node_num 307 --learning_rate 0.001 --patience 20 --emb_dim 128 --encoder_layer 2 --decoder_layer 3
