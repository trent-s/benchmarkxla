# PyTorch Benchmarks - xla container version
This is a collection of open source benchmarks used to evaluate PyTorch performance that is tweaked to run in an xla container.


## Suggested installation for running xla

This is work in progress. I strongly suggest carefully reviewing output for errors or warnings.

```
docker run --gpus all -p6006:6006 -v ~/data:/data -it --rm gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8
cd
mkdir -p git
cd git
git clone https://github.com/trent-s/benchmarkxla.git
sh benchmarkxla/contrib/prep.sh
export GPU_NUM_DEVICES=1
```

This might be a good point for some sanity checking.
Then try a simple xla test such as:

```
python benchmarkxla/run_xla.py -d xla -t eval resnet18
```

If all goes well, you might see output like this:

```
Running eval method from resnet18 on xla in eager mode with input batch size 8 and precision fp32.
CPU Total Wall Time:   2.573 milliseconds
CPU Peak Memory:                2.3271 GB
```

Then try something like this:
```
pytest test_bench_xla.py --ignore_machine_config -k "test_eval[resnet18"
```

output:
```
============================= test session starts ==============================
platform linux -- Python 3.8.8, pytest-6.2.3, py-1.10.0, pluggy-0.13.1
benchmark: 4.0.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /root/git/benchmarkxla
plugins: anyio-2.2.0, hypothesis-6.29.3, benchmark-4.0.0
collected 359 items / 357 deselected / 2 selected

../test_bench_xla.py ..                                                  [100%]

=============================== warnings summary ===============================
../../../anaconda3/envs/pytorch/lib/python3.8/site-packages/sympy/external/importtools.py:158
../../../anaconda3/envs/pytorch/lib/python3.8/site-packages/sympy/external/importtools.py:158
  /root/anaconda3/envs/pytorch/lib/python3.8/site-packages/sympy/external/importtools.py:158: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
    if LooseVersion(modversion) < LooseVersion(min_module_version):

-- Docs: https://docs.pytest.org/en/stable/warnings.html

--------------------------------------------------------------------------------------------- benchmark 'hub': 2 tests ---------------------------------------------------------------------------------------------
Name (time in us)                        Min                    Max                  Mean              StdDev                Median                IQR            Outliers         OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_eval[resnet18-xla-jit]         872.8649 (1.0)      14,515.2670 (2.27)       962.1254 (1.0)      670.7067 (3.25)       913.6440 (1.0)      58.0552 (1.0)          7;20  1,039.3655 (1.0)         421           1
test_eval[resnet18-xla-eager]     2,317.0619 (2.65)      6,391.0780 (1.0)      2,402.9897 (2.50)     206.1519 (1.0)      2,366.6259 (2.59)     96.2181 (1.66)          2;2    416.1483 (0.40)        413           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Legend:
  Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
  OPS: Operations Per Second, computed as 1 / Mean
========== 2 passed, 357 deselected, 2 warnings in 107.32s (0:01:47) ===========
```

## suggested installation for baseline testing
```
docker run --gpus all -p 6006:6006 -v ~/data:/data -it --rm ghcr.io/pytorch/pytorch-nightly:latest
cd
mkdir -p git
cd git
git clone https://github.com/trent-s/benchmarkxla.git
sh benchmarkxla/contrib/prepbase.sh
export GPU_NUM_DEVICES=1
```
See ./prepbase.sh for commands to prepare for baseline testing.

## profiling tips
Sample profiling on a standard baseline container:
```
export GPU_NUM_DEVICES=1
python run.py squeezenet1_1 -d cuda -m eager -t eval --profile --profile-detailed
```

Sample profiling on a xla container:
```
export GPU_NUM_DEVICES=1
python run_xla.py squeezenet1_1 -d xla -m eager -t eval --profile --profile-detailed
```

Resulting json log files in `logs` directory can be visualized with chrome, tensorboard, or other tools.

To use tensorboard, start tensorboard on container:
```
tensorboard --logdir=logs-xla --bind_all &
```

Then view the interface by pointing a web browser at the machine running the container:
e.g., `http://trlai2.sl.cloud9.ibm.com:6006`


## ongoing todo list
- Prepare code for upstream merge.
- Dive much deeper with profiling etc.


## brief description of scripts
- benchrun.sh - use ../test_bench_xla.py to run benchmarks for selected models using xla
- benchrunbase.sh - use ../test_bench.py to run benchmarks for selected models using cuda and cpu
- loggedpermute.py - a sample python script to create verbose output from pytorch internals
- loggedrun.sh - a runner script to call loggedpermute.py to create verbose output from pytorch internals
- prep.sh - setup script for using xla container to run these benchmarks using xla
- prepbase.sh - setup script for using nightly pytorch container to run these benchmarks using cuda and cpu
- simplerun.sh use - ../run_xla.py to run simple benchmarks for selected models using xla
- simplerunbase.sh - use ../run.py to run simple benchmarks for selected models using cuda and cpu
- logs/tabulate.sh - create table of raw benchrun output for use in excel

## References
- https://github.com/pytorch/benchmark
- https://pytorch.org/xla/master
- https://github.com/pytorch/xla
- https://pytorch.org/tutorials/beginner/profiler.html
- https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm




