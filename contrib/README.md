# PyTorch Benchmarks - xla container version
This is a collection of open source benchmarks used to evaluate PyTorch performance that is tweaked to run in an xla container.


## Suggested installation


This is work in progress. I strongly suggest carefully reviewing output for errors or warnings.
At this point it is safe to assume everything is broken.

```
docker run --gpus all -it  --rm   gcr.io/tpu-pytorch/xla:nightly_3.8_cuda_11.8
cd
mkdir -P git
cd git
git clone https://github.com/trent-s/benchmarkxla.git
sh benchmarkxla/contrib/prep.sh
```

This might be a good point for some sanity checking.
Then try a simple xla  test such as:

```
python benchmarkxla/runxla.py -d xla -t eval  resnet18
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
work in progress...
```


## ongoing todo list
- focus on evaluation tests for BERT_pytorch, vgg16, and resenet18 models
- enable test.py and test_bench.py to work with hardcoded xla enablement
- enable all code to function correctly with and without xla support, so that key xla benchmark support can be merged upstream.
- look at additional models and additional tests.
- dive much deeper

## References
- https://github.com/pytorch/benchmark
- https://pytorch.org/xla/master/
- https://github.com/pytorch/xla



