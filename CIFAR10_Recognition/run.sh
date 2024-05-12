#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -name EXP -dataset cifar10 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -weight_decay 5e-5 -drop_rate 0.1 -T 4

wait
echo "Success!"
