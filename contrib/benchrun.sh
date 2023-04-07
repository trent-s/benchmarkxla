#/bin/bash

top=$(dirname "$0")/..

for model in BERT_pytorch vgg16 resnet18 resnet50 resnetxt50_32x4d alexnet mobilenet_v2 mnasnet1_0 squeezenet1_1 timm_vision_transformer geomean
do
	echo
	echo Trying pytest bench on model ${model}
	echo
	pytest ${top}/test_bench_xla.py --ignore_machine_config -k "test_eval[${model}"
	echo
	echo
done
