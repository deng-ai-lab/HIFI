# HIFI: Heterogeneous Spiking Framework with Self-Inhibiting Neurons
This is the official repository of HIFI, an accurate, efficient and low-latency learning framework for general AI tasks.
![image](https://github.com/bo-wang-up/HIFI/blob/master/Overview%20of%20HIFI/overview.png)
# License
The code is licensed for non-commercial academic research purpose only.
# System Requirements
 * No special system requirements.
 * The software has been tested on the Linux desktop (Ubuntu 20.04.3 LTS operation system) with Intel(R) Xeon(R) Gold 6226R CPU and NVIDIA Geforce RTX 3090 (Driver Version: 470.63.01, CUDA Version: 11.4).
# HIFI software package:
 * Dependencies and Libraries
   - python==3.8.11
   - torch==1.9.0
   - numpy==1.22.2
   - matplotlib==3.4.2
   - scikit-learn==1.0.2
   - seaborn==0.11.2
   - PyYAML==6.0
   - torchvision==0.10.0

 * Installation
   - install requirements by typing `pip install -r requirements.txt`.
# Simulating the spiking dynamics of brain neurons
   * This part is provided in the `biological_fidelity/` document.
   * We provide the training and testing datasets of mouse brain neurons, i.e. the `.npy` files in the `data/` document.
   * The calcium-evoked fluorescent video can be downloaded in `https://www.dropbox.com/s/jhk8hr7wb40mb7m/20210818_66_iron.avi?dl=0`
   * The demo for neurons can be run by **Jupyter Notebook**, i.e. `biological_fidelity_demo.ipynb`.
   * The code will train the surrogate neurons for different neurons, and visualize the testing results.
#  HIFI on general AI datasets
   * This part is provided in the `cifar10_recognition/` document.
   * The demo is applied for CIFAR10 dataset on different network architectures. Here, we provide two sample network architectures in the `arch.yaml`.
   * Run `main.py` to directly train HIFI with default parameter settings. 
   * Run `test.py` to test accuracy of the trained HIFI model. Pre-trained models for two sample network architectures can be downloaded in `https://www.dropbox.com/scl/fo/r0h1588ctns4iop0jalpq/h?dl=0&rlkey=s4eztpin5pf2chnrdpqesv6qh`.
   * Our model is scalable for different network architectures and general for different datasets. The network architecture and the datasets can be replaced as you need.
