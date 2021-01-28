# Deep_Dup-Attack

The wide deployment of Deep Neural Networks (DNN) in high-performance cloud computing platforms brought to light field-programmable gate arrays (FPGA) as a popular choice of accelerator to boost performance due to its hardware reprogramming flexibility. To improve the efficiency of hardware resource utilization, growing efforts have been invested in FPGA virtualization, enabling the co-existence of multiple independent tenants in a shared FPGA chip. Such multi-tenant FPGA setup for DNN acceleration potentially exposes DNN interference task under severe threat from malicious users. This work, to the best of our knowledge, is the first to explore DNN model vulnerabilities in multi-tenant FPGAs. We propose a novel adversarial attack framework: Deep-DupA, in which the adversarial tenant can inject faults to the DNN model of victim tenant in FPGA. Specifically, she can aggressively overload the shared power distribution system of FPGA with malicious power-plundering circuits, achieving adversarial weight duplication (AWD) hardware attack that duplicates certain DNN weight packages during data transmission between off-chip memory and on-chip buffer, with the objective to hijack DNN function of the victim tenant. 
Further, to identify the most vulnerable DNN weight packages for a given malicious objective, we propose a generic vulnerable weight package searching algorithm, called Progressive Differential Evolution Search (P-DES), which is, for the first time, adaptive to both deep learning white-box and black-box attack models. 
Unlike prior works only working in a deep learning white-box setup, our adaptiveness mainly comes from the fact that the proposed P-DES does not require any gradient information of DNN model. The proposed Deep-DupA is validated in a developed multi-tenant FPGA prototype, for two popular deep learning applications, i.e., Object Detection and Image Classification. Successful attacks are demonstrated in six popular DNN architectures (e.g., YOLOv2, ResNet-50, MobileNet, etc.) on three datasets (COCO, CIFAR-10, and ImageNet). The experimental results demonstrate the effectiveness of the proposed Deep-DupA attack framework. For example, Deep-DupA can successfully attack MobileNetV2 (i.e., degrade test accuracy from 70 % to lower than 0.2 %), with only ~2 weight package duplication out of 2.1 Million on ImageNet dataset.


## Description of The Code.
In the repository, we provide a sample code to implement Deep_DupA attack for CIFAR10 dataset. The paper can be find in the arxiv link https://arxiv.org/abs/2011.03006. The link to get the VGG-11 and Resnet-20 model is:  https://drive.google.com/drive/folders/1_9jP1becCM2efD-4wTHyEoUWTePOwApK?usp=sharing. 

## Commands to Run the code:




## Parameters:

iteration = number of attack iteration

picks = numberof weights picked

weight_p_clk = number of weights at each package constant throughout the paper

shift_p_clk =  number of clock shift at each iteration constant thourghout the paper

evolution = number of evolution = picks = number of initial candidate chosen

targeted = target attack class if targetd attack

BATCH_SIZE = batch_size



## Dependencies Commands:

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

Python --> 3.4

Scipy ---> 0.15

pip install xlwt

pip install pandas
