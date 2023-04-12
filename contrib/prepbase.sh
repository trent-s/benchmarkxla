#!/bin/bash

# setup script for base line case:
# works with following docker container
# docker run --gpus all -p 6006:6006 -v ~/data:/data -it --rm ghcr.io/pytorch/pytorch-nightly:latest

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1

apt-get update
apt-get -y install git vim 
cd
mkdir git
cd git
git clone https://github.com/trent-s/benchmarkxla.git
cd benchmarkxla
pip install --pre torch torchvision torchtext torchaudio -f https://download.pytorch.org/whl/nightly/cu117/torch_nightly.html
pip install pyyaml
pip install numba

pip install torch_tb_profiler
pip install -U tensorboard-plugin-profile
pip install tensorflow

python install.py BERT_pytorch vgg16 resnet18 resnet50 resnext50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer

pip install 'numpy<1.23.0,>1.22.0'

# then run benchrunbase.sh or simplerunbase.sh to run baseline benchmarks using cuda and cpu
# When using GPU be sure to set export GPU_NUM_DEVICES=1

# to use tensorboard run following command:
# tensorboard --logdir=logs-xla --bind_all &
