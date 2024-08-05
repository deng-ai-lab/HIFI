#!/bin/bash

### CIFAR-10
CUDA_VISIBLE_DEVICES=0 python train.py -name EXP -dataset cifar10 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0.1 -T 2 &

CUDA_VISIBLE_DEVICES=1 python train.py -name EXP -dataset cifar10 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0.1 -T 4 &

CUDA_VISIBLE_DEVICES=2 python train.py -name EXP -dataset cifar10 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0.1 -T 6 &

### CIFAR-100
CUDA_VISIBLE_DEVICES=3 python train.py -name EXP -data_dir -dataset cifar100 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0 -T 2 &

CUDA_VISIBLE_DEVICES=4 python train.py -name EXP -data_dir -dataset cifar100 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0 -T 4 &

CUDA_VISIBLE_DEVICES=5 python train.py -name EXP -data_dir -dataset cifar100 -model spiking_resnet18 -amp -T_max 1024 -epochs 1024 -interval 20 -weight_decay 5e-5 -drop_rate 0 -T 6 &

wait
echo "Success!"
