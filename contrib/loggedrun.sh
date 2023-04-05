#! /bin/bash
#top=$(dirname "$0")
top=$(pwd)

unset script
if [[ $# -lt 1 ]]; then
  scripts=( $(echo *.py) )
  if [[ ${#scripts[*]} -eq 0 ]]; then
    echo "ERROR: No python script is specified/found"
    exit 1
  elif [[ ${#scripts[*]} -gt 1 ]]; then
    echo "ERROR: More than one python scripts are found"
    exit 1
  fi
  script="${scripts[0]}"
else
  script="${@: -1}"
  if [[ x"${script%.py}" = x"$script" ]]; then
    echo "ERROR: No python script is specified"
    exit 1
  elif [[ ! -f "$script" ]]; then
    echo "ERROR: No such python script: $script"
    exit 1
  fi
fi
logfile=${script%.py}.log

set -x

export TRITON_CACHE_DIR=$top/triton
export TORCHINDUCTOR_CACHE_DIR=$top/inductor
export TORCH_COMPILE_DEBUG=1

## Dynamo Env Options
export TORCHDYNAMO_VERBOSE=1
export PYTORCH_MATCHER_LOGLEVEL=1
export TORCHDYNAMO_PRINT_MISSING=1
export TORCHDYNAMO_PRINT_GUARDS=1
#export TORCHDYNAMO_DISABLE=1
#export TORCH_COMPILE_DISABLE=1

## Inductor Env Options
export INDUCTOR_WRITE_SCHEDULER_GRAPH=1
export TORCHINDUCTOR_WRITE_MISSING_OPS=1
#export TORCHINDUCTOR_MAX_AUTOTUNE=1
#export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1
#export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1
#export TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE=1

## Functorch/AOTAutograd/Other Env Options
export AOT_FX_GRAPHS=True
export AOT_PARTITIONER_DEBUG=True
export PYTORCH_MATCHER_LOGLEVEL=DEBUG
export PYTORCH_JIT_STATS=True
#export PYTORCH_JIT_DISABLE=True

python "$@" $script 2>&1 | tee $logfile
