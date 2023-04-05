# PyTorch Benchmarks - xla continer version
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
Then try a simple test such as:

```
python ./runxla.py -d xla -t eval  resnet18
```


## ongoing todo list
- focus on evaluation tests for BERT_pytorch, vgg16, and resenet18
- enable test.py and test_bench.py to work with hardcoded xla enablement
- enable all code to function correctly with and without xla support, so that key xla support can be merged upstream.
- look at additional models and additional tests.
- dive deeper

## References
- https://github.com/pytorch/benchmark
- https://pytorch.org/xla/master/
- https://github.com/pytorch/xla



