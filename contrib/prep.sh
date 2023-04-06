#!/bin/bash

set -x

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1

# pip install https://storage.googleapis.com/tpu-pytorch/wheels/cuda/118/torch_xla-nightly-cp38-cp38-linux_x86_64.whl
pip install pyyaml
pip install 'numpy<1.23.0'


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

cd benchmarkxla
python install.py BERT_pytorch vgg16 resnet18
