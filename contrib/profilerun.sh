#/bin/bash

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1
export TYPE=xla
# export TYPE=cpu
# export TYPE=cuda

# other choices may include:
#  cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, ort, xla, lazy, vulkan, mps, meta, hpu, mtia,

top=$(dirname "$0")/..

for MODE in eager jit
do
    for model in BERT_pytorch vgg16 resnet18 resnet50 resnext50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer
    do
	echo
	echo Trying profile ${MODE} run ${TYPE} on model ${model}
	echo
       	python ${top}/run.py  -m ${MODE} -d ${TYPE} -t eval --profile --profile-detailed ${model} 2>&1 | tee /data/${TYPE}-${MODE}-${model}-out.txt
	mv logs /data/${TYPE}-${MODE}-${model}-logs
	echo
	echo
    done
done
