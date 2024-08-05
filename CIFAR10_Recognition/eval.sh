CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar10 -model spiking_resnet18 -T 2 -ckpt cifar10_ts2.pth

# CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar10 -model spiking_resnet18 -T 4 -ckpt cifar10_ts4.pth

# CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar10 -model spiking_resnet18 -T 6 -ckpt cifar10_ts6.pth

# CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar100 -model spiking_resnet18 -T 2 -ckpt cifar100_ts2.pth

# CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar100 -model spiking_resnet18 -T 4 -ckpt cifar100_ts4.pth

# CUDA_VISIBLE_DEVICES=0 python test.py -dataset cifar100 -model spiking_resnet18 -T 6 -ckpt cifar100_ts6.pth