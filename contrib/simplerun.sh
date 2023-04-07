#/bin/bash

top=$(dirname "$0")/..

for model in BERT_pytorch vgg16 resnet18 resnet50 resnetxt50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer geomean
do
	echo
	echo Trying simple run on model ${model}
	echo
        python ${top}/run_xla.py -d xla -t eval ${model}
	echo
	echo
done
