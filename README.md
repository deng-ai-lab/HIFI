# HIFI: Heterogeneous Spiking Framework with Self-Inhibiting Neurons
This is the official repository of HIFI, an accurate, efficient and low-latency learning framework for general AI tasks.
![image](https://github.com/deng-ai-lab/HIFI/blob/main/Overview%20of%20HIFI/overview.png)
# License
The code is licensed for non-commercial academic research purpose only.
# System Requirements
 * No special system requirements.
 * The software has been tested on the Linux desktop (Ubuntu 20.04.3 LTS operation system) with Intel(R) Xeon(R) Gold 6226R CPU and NVIDIA Geforce RTX 3090 (Driver Version: 535.171.04, CUDA Version: 12.2).
# HIFI software package:
 * Dependencies and Libraries
   - joblib==1.3.2
   - matplotlib==3.8.3
   - numpy==1.24.1
   - Pillow==10.3.0
   - progress==1.6
   - spikingjelly==0.0.0.0.14
   - torch==2.2.1+cu118
   - torchvision==0.17.1+cu118
 * Installation
   - install requirements by typing `pip install -r requirements.txt`.
#  HIFI on general AI datasets
   * This part is provided in the `CIFAR10_Recognition/` document.
   * Run run.sh to directly train HlFl with default parameter settings.
   * Pre-trained spiking resnet18 with HIFI neuron on CIFAR10 can be download in `https://www.dropbox.com/scl/fo/2f2sqe3vy8un5pxz9ahvj/ACb8st_gkTXhj5RtnVnD6vs?rlkey=fsxb7xj0h0nwuzn446ojg58qf&st=j1w60rv4&dl=0`. Run eval.sh to evaluate the accuracy and FLOPS after downloading the pre-trained weights.
   * Our model is scalable for different network architectures and general for different datasets. The network architecture and the datasets can be replaced as you need.
