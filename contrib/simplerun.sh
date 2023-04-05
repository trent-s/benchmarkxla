#/bin/bash

top=$(dirname "$0")/..

for model in BERT_pytorch vgg16 resnet18
do
	echo
	echo Trying simple run on model ${model}
	echo
        python ${top}/runxla.py -d xla -t eval ${model}
	echo
done
