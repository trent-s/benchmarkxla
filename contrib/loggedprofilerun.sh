#/bin/bash

export GPU_NUM_DEVICES=1
export FORCE_CUDA=1
export TYPE=xla


top=$(dirname "$0")/..

for MODE in eager jit
do
    for model in BERT_pytorch vgg16 resnet18 resnet50 resnext50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer
    do
	echo
	echo Trying simple logged profile ${MODE} run ${TYPE} on model ${model}
	echo
        ${top}/contrib/loggedrun.sh ${top}/run.py --vlog -m ${MODE} -d xla -t eval ${model} 2>&1 | tee /data/${TYPE}-${MODE}-${model}-out.txt
	mv logs /data/${TYPE}-${MODE}-${model}-logs
	mv run.log /data/${TYPE}-${MODE}-${model}-run.log
	echo
	echo
    done
done
