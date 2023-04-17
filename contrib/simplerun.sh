#/bin/bash

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1

top=$(dirname "$0")/..

TYPE=xla
# TYPE=cpu
# TYPE=cuda

for model in BERT_pytorch vgg16 resnet18 resnet50 resnext50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer
do
	echo
	echo Trying simple run ${TYPE} on model ${model}
	echo
        python ${top}/run.py -d ${TYPE} -t eval ${model}
	echo
	echo
done
