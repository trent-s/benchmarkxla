#/bin/bash

top=$(dirname "$0")/..

for model in BERT_pytorch vgg16 resnet18
do
	echo
	echo Trying pytest bench on model ${model}
	echo
	pytest ${top}/test_bench_xla.py --ignore_machine_config -k "test_eval[${model}"
	echo
done
