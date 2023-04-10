#!/bin/bash

# suggestion: this script works well with the following docker container:
# docker run --gpus all -it  --rm   gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8

set -x

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1

# pip install https://storage.googleapis.com/tpu-pytorch/wheels/cuda/118/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
pip install pyyaml
pip install numba

cd 
mkdir -p git
cd git

git clone https://github.com/pytorch/data.git
cd data
python setup.py install
cd -

git clone https://github.com/pytorch/text torchtext
cd torchtext
git submodule update --init --recursive
python setup.py clean install
cd -

git clone --recursive https://github.com/pytorch/kineto.git
cd kineto/libkineto
mkdir -p build
cd build
cmake .. && make && make install
cd 
cd git

apt-get -y update
apt-get -y install ffmpeg
pip install ffmpeg
git clone https://github.com/pytorch/audio
cd audio
python setup.py clean install
cd -

# add some sanity testing

pip install 'numpy<1.23.0,>1.22.0'

cd benchmarkxla
python install.py BERT_pytorch vgg16 resnet18 resnet50 resnext50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer

# now ready to run xla benchmarking with benchrun.sh or simplerun.sh
