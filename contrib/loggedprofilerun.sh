#/bin/bash

export TRITON_CACHE_DIR=$top/triton
export TORCHINDUCTOR_CACHE_DIR=$top/inductor
export TORCH_COMPILE_DEBUG=1

## Dynamo Env Options
export TORCHDYNAMO_VERBOSE=1
export PYTORCH_MATCHER_LOGLEVEL=1
export TORCHDYNAMO_PRINT_MISSING=1
export TORCHDYNAMO_PRINT_GUARDS=1

## Inductor Env Options
export INDUCTOR_WRITE_SCHEDULER_GRAPH=1
export TORCHINDUCTOR_WRITE_MISSING_OPS=1

## Functorch/AOTAutograd/Other Env Options
export AOT_FX_GRAPHS=True
export AOT_PARTITIONER_DEBUG=True
export PYTORCH_MATCHER_LOGLEVEL=DEBUG
export PYTORCH_JIT_STATS=True

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
       	python ${top}/run.py --vlog -m ${MODE} -d xla -t eval ${model} 2>&1 | tee /data/${TYPE}-${MODE}-${model}-out.txt
	mv logs /data/${TYPE}-${MODE}-${model}-logs
	echo
	echo
    done
done
