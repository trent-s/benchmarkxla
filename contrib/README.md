# PyTorch Benchmarks - xla container version
This is a collection of open source benchmarks used to evaluate PyTorch performance that is tweaked to run in an xla container.


## Suggested installation


This is work in progress. I strongly suggest carefully reviewing output for errors or warnings.
At this point it is safe to assume everything is broken.

```
docker run --gpus all -it  --rm   gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8
cd
mkdir -p git
cd git
git clone https://github.com/trent-s/benchmarkxla.git
sh benchmarkxla/contrib/prep.sh
```

This might be a good point for some sanity checking.
Then try a simple xla  test such as:

```
python benchmarkxla/run_xla.py -d xla -t eval  resnet18
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


## ongoing todo list
- Focus on evaluation tests for BERT_pytorch, vgg16, and resenet18 models.
- Enable test.py and test_bench.py to work with hardcoded xla enablement.
  - Currently, vggt16 and resnet18 seem to work with both test.py and test_bench.py... BERT_pytorch is a work in progress.
- Enable all code to function correctly with and without xla support, so that key xla benchmark support can be merged upstream.
- Look at additional models and additional tests.
- Dive much deeper.

## References
- https://github.com/pytorch/benchmark
- https://pytorch.org/xla/master
- https://github.com/pytorch/xla



