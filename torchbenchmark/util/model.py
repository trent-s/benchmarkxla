import copy
import importlib
import os
import torch
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    xla_support = 0
from contextlib import contextmanager, ExitStack
import warnings
import inspect
import yaml
from pathlib import Path
from typing import ContextManager, Optional, List, Tuple, Generator
from torch.utils._pytree import tree_map
from torchbenchmark import REPO_PATH
from torchbenchmark.util.extra_args import check_correctness_p, parse_opt_args, apply_opt_args, \
                                           parse_decoration_args, apply_decoration_args, is_staged_train_test, \
                                           TEST_STAGE
from torchbenchmark.util.env_check import set_random_seed, correctness_check, stableness_check, is_hf_model
from torchbenchmark.util.fx_int8 import get_sub_module, prepare_sub_module, convert_sub_module

class PostInitProcessor(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj

@contextmanager
def no_grad(val):
    """Some meta-learning models (e.g. maml) may need to train a target(another) model
    in inference runs
    """
    old_state = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(not val)
        yield
    finally:
        torch.set_grad_enabled(old_state)

@contextmanager
def nested(*contexts):
    """
    Chain and apply a list of contexts
    """
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield contexts

# enable JIT profiling executor
@contextmanager
def enable_profiling_executor():
    try:
        graph_executor = torch._C._get_graph_executor_optimize(True)
        profiling_executor = torch._C._jit_set_profiling_executor(True)
        profiling_mode = torch._C._jit_set_profiling_mode(True)
        yield
    finally:
        torch._C._jit_set_profiling_mode(profiling_mode)
        torch._C._jit_set_profiling_executor(profiling_executor)
        torch._C._get_graph_executor_optimize(graph_executor)

class BenchmarkModel(metaclass=PostInitProcessor):
    DEFAULT_TRAIN_BSIZE: Optional[int] = None
    DEFAULT_EVAL_BSIZE: Optional[int] = None
    # by default, deepcopy the model when checking correctness
    # because some models are stateful (such as moco)
    DEEPCOPY: bool = True

    test: str
    device: str
    jit: bool
    batch_size: int
    extra_args: List[str]
    run_contexts: List[ContextManager]

    """
    A base class for adding models to torch benchmark.
    See [Adding Models](#../models/ADDING_MODELS.md)
    """
    def __init__(self, test: str, device: str, jit: bool=False, batch_size: Optional[int]=None, extra_args: List[str]=[]):
        self.metadata = self.load_metadata()
        self.test = test
        assert self.test == "train" or self.test == "eval", \
            f"Test must be 'train' or 'eval', but get {self.test}. Please submit a bug report."
        self.device = device
        self.jit = jit
        self.determine_batch_size(batch_size)
        self.extra_args = extra_args
        self.opt = None
        # contexts to run in the test function
        if self.test == "train":
            # In train test, there are run contexts that should only be applied for forward/backward/optimizer stage
            # For example, amp only applies for the forward stage
            self.forward_contexts = []
            self.backward_contexts = []
            self.optimizer_contexts = []
        self.run_contexts = [
            enable_profiling_executor  # force JIT profiling executor to be enabled by default
        ]

        # taken from torchdynamo benchmarks, this further controls randomness settings
        def deterministic_torch_manual_seed(*args, **kwargs):
            from torch._C import default_generator

            seed = 1337
            import torch.cuda

            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)

            return default_generator.manual_seed(seed)

        torch.manual_seed = deterministic_torch_manual_seed
        set_random_seed()
        # sanity checks of the options
        assert self.test == "train" or self.test == "eval", f"Test must be 'train' or 'eval', but provided {self.test}."
        # parse the args
        self.dargs, opt_args = parse_decoration_args(self, self.extra_args)
        # if the args contain "--torchdynamo", parse torchdynamo args
        if "--torchdynamo" in opt_args:
            self.dynamo = True
            from torchbenchmark.util.backends.torchdynamo import parse_torchdynamo_args
            self.opt_args, self.extra_args = parse_torchdynamo_args(self, opt_args)
        else:
            self.dynamo = False
            self.opt_args, self.extra_args = parse_opt_args(self, opt_args)

    # Run the post processing for model acceleration
    def __post__init__(self):
        # All arguments should be parsed at this point.
        assert not self.extra_args, f"Expected no unknown args at this point, found {self.extra_args}"
        should_check_correctness = check_correctness_p(self, self.opt_args, self.dargs)
        if should_check_correctness:
            self.eager_output = stableness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY, rounds=1)
            if isinstance(self.eager_output, Tuple):
                self.eager_output = tuple((t.detach() if isinstance(t, torch.Tensor) else t) for t in self.eager_output)
            elif isinstance(self.eager_output, torch.Tensor):
                self.eager_output = self.eager_output.detach()
            if self.test == "train":
                current_optimizer = self.get_optimizer()
                if current_optimizer is not None:
                    self.set_optimizer(None)
                try:
                    if self.DEEPCOPY:
                        copy_model = copy.deepcopy(self)
                    else:
                        copy_model = self
                    copy_model.invoke()
                    self.eager_model_after_one_train_iteration = copy_model.model
                except RuntimeError:
                    warnings.warn(UserWarning("Can't copy the model. Skipping train correctness check."))
                if current_optimizer is not None:
                    self.set_optimizer(current_optimizer)
        # apply decoration args
        apply_decoration_args(self, self.dargs)
        # apply optimization args
        if self.dynamo:
            from torchbenchmark.util.backends.torchdynamo import apply_torchdynamo_args
            apply_torchdynamo_args(self, self.opt_args, self.dargs.precision)
        else:
            apply_opt_args(self, self.opt_args)
        if should_check_correctness:
            # tensorrt or fp16 is known to generate less-accurate results
            # in this case, use more relaxed cosine similarity instead of torch.allclose
            # for correctness testing
            # see: https://github.com/pytorch/torchdynamo/pull/438
            if (
                self.dargs.precision == "fp16"
                or self.dargs.precision == "amp"
                or (self.dynamo and self.opt_args.torchdynamo == "fx2trt")
                or (not self.dynamo and (self.device == "cuda" and self.opt_args.backend == "fx2trt"))
                or (not self.dynamo and self.opt_args.use_cosine_similarity)
                or self.dargs.precision == "fx_int8"
                or self.dargs.precision == "bf16"
                or self.dargs.precision == "amp_fp16"
                or self.dargs.precision == "amp_bf16"
            ):
                self.correctness = correctness_check(self, cos_sim=True, deepcopy=self.DEEPCOPY)
            else:
                # get tolerance of correctness check from os.environ
                atol = float(os.environ.get("TORCHBENCH_ATOL", "1e-4"))
                rtol = float(os.environ.get("TORCHBENCH_RTOL", "1e-4"))
                self.correctness = correctness_check(self, cos_sim=False, deepcopy=self.DEEPCOPY, atol=atol, rtol=rtol)
        # setup distributed trainer
        if self.dargs.distributed:
            if self.dargs.distributed_wrap_fn:
                pos = self.dargs.distributed_wrap_fn.rfind(".")
                module = importlib.import_module(self.dargs.distributed_wrap_fn[:pos])
                apply_trainer = getattr(module, self.dargs.distributed_wrap_fn[(pos+1):])
            else:
                from torchbenchmark.util.distributed.core_model.apply_trainer import apply_trainer
            if is_hf_model(self):
                # DDP requires to use unwrapped model for huggingface
                module, _inputs = self.get_module(wrap_model=False)
            else:
                module, _inputs = self.get_module()
            self.set_module(apply_trainer(module, self.dargs.distributed))
        # Need to clean up the cache because we run deep copy within correceness check
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def determine_batch_size(self, batch_size=None):
        # batch size priority for eval tests: not ALLOW_CUSTOMIZE_BSIZE > user specified > device specified > default
        # batch size priority for train tests: not ALLOW_CUSTOMIZE_BSIZE > user specified > default
        self.batch_size = batch_size
        if not batch_size:
            self.batch_size = self.DEFAULT_TRAIN_BSIZE if self.test == "train" else self.DEFAULT_EVAL_BSIZE
            if self.device == "cuda":
                current_device_name = torch.cuda.get_device_name()
                assert current_device_name, f"torch.cuda.get_device_name() returns None when device is set to cuda, please double check."
            elif self.device == "cpu":
                current_device_name = "cpu"
            elif self.device == "mps":
                current_device_name = "mps"
            elif self.device == "xla":
                current_device_name = xm.xla_device()
            # use the device suggestion on CUDA inference tests, key should be either eval_batch_size or train_batch_size
            device_batch_size_key = f"{self.test}_batch_size"
            if self.metadata and "devices" in self.metadata and current_device_name in self.metadata["devices"] \
                             and device_batch_size_key in self.metadata["devices"][current_device_name]:
                self.batch_size = self.metadata["devices"][current_device_name][device_batch_size_key]
            # If the model doesn't implement test or eval test
            # its DEFAULT_TRAIN_BSIZE or DEFAULT_EVAL_BSIZE will still be None
            if not self.batch_size:
                raise NotImplementedError(f"Test {self.test} is not implemented.")
        else:
            self.batch_size = batch_size
        # Check if specified batch size is supported by the model
        if hasattr(self, "ALLOW_CUSTOMIZE_BSIZE") and (not getattr(self, "ALLOW_CUSTOMIZE_BSIZE")):
            if self.test == "train" and (not self.batch_size == self.DEFAULT_TRAIN_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")
            elif self.test == "eval" and (not self.batch_size == self.DEFAULT_EVAL_BSIZE):
                raise NotImplementedError("Model doesn't support customizing batch size.")

    def load_metadata(self):
        relative_path = self.__class__.__module__.split(".")
        self.name = relative_path[-1]
        metadata_loc = Path(REPO_PATH).joinpath(*relative_path).joinpath("metadata.yaml")
        if not metadata_loc.exists():
            return None
        with open(metadata_loc, "r") as mf:
            metadata = yaml.safe_load(mf)
        return metadata

    def add_context(self, context_fn, stage=TEST_STAGE.ALL):
        ctx = context_fn()
        assert isinstance(ctx, ContextManager), f"Expected adding a ContextManager, get {type(ctx)}. Please report a bug."
        if stage == TEST_STAGE.ALL:
            self.run_contexts.append(context_fn)
        elif stage == TEST_STAGE.FORWARD:
            self.forward_contexts.append(context_fn)
        elif stage == TEST_STAGE.BACKWARD:
            self.backward_contexts.append(context_fn)
        elif stage == TEST_STAGE.OPTIMIZER:
            self.optimizer_contexts.append(context_fn)


    # Common interface for all models extending BenchmarkModel to access the optimizer.
    # Some models have an opt attribute, others have an optimizer attribute; this
    # implementation handles both. This function should not error! Simply return None
    # if there's no optimizer in sight.
    def get_optimizer(self):
        if hasattr(self, "optimizer"):
            return self.optimizer
        if hasattr(self, "opt"):
            return self.opt
        warnings.warn("The optimizer for this model is not stored in self.opt nor self.optimizer. "
                      "Currently returning None! Please override this implementation with your own "
                      "if there is an optimizer this should be returning instead.")
        return None

    # Takes in an optimizer and sets that to be the optimizer used from now on.
    # There are special models like dcgan that would update multiple optimizers at once,
    # so optimizer here is not always strictly a, say, torch.optim.Optimizer.
    def set_optimizer(self, optimizer) -> None:
        if hasattr(self, "optimizer"):
            self.optimizer = optimizer
            return
        if hasattr(self, "opt"):
            self.opt = optimizer
            return
        raise NotImplementedError("The optimizer for this model is not stored in self.opt nor self.optimizer. "
                                  "Please override this implementation with your own.")

    # Default implementation for replacing the model
    def set_module(self, new_model):
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model = new_model
        else:
            raise NotImplementedError("The instance variable 'model' does not exist or is not type 'torch.nn.Module', implement your own `set_module()` function.")

    def gen_inputs(self, num_batches: int=1) -> Tuple[Generator, Optional[int]]:
        """Generate a tuple of (iterator of model input, the size of the iterator).
           If size is None, the input is randomly generated and has infinite size."""
        raise NotImplementedError("Default input generation function is not implemented. "
                                  "Please submit an issue if you need input iterator implementation for the model.")

    def invoke_staged_train_test(self) -> None:
        optimizer = self.get_optimizer()
        if optimizer is not None:
            optimizer.zero_grad()

        with nested(*self.forward_contexts):
            losses = self.forward()

        with nested(*self.backward_contexts):
            self.backward(losses)

        if optimizer is not None:
            with nested(*self.optimizer_contexts):
                self.optimizer_step()

        return None

    def invoke(self) -> Optional[Tuple[torch.Tensor]]:
        out = None
        if self.test == "train" and is_staged_train_test(self):
            self.invoke_staged_train_test()
            return out
        with nested(*self.run_contexts):
            if self.test == "train":
                self.train()
            elif self.test == "eval":
                out = self.eval()
        return out

    def eval_in_nograd(self):
        return True

    def check_opt_vs_noopt_jit(self):
        if not self.jit:
            return

        model_name = inspect.getfile(self.__class__).split(os.sep)[-2]
        print(f"model_name={model_name} , {inspect.getfile(self.__class__)}")
        model_blacklist = [
            'demucs', # set up issue
            'yolov3', # set up issue
            'BERT_pytorch', # set up issue
            'moco', # set up issue
            'Super_SloMo', # results don't match, might be due to the way TE CUDA handles rand?
            'attention_is_all_you_need_pytorch', # results don't match, might be due to the way TE CUDA handles rand?
        ]

        if model_name in model_blacklist:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        # if a model doesn't support `get_module`
        # we should let it throw and then
        # override `check_results` for that model
        try:
            model, inputs = self.get_module()
        except NotImplementedError:
            warnings.warn(UserWarning(f"{model_name}.get_module() doesn't support `check_results` yet!"))
            return

        def bench_allclose(a, b):
            if isinstance(a, torch.Tensor):
                assert(isinstance(b, torch.Tensor))
                assert(a.allclose(b))
            elif isinstance(a, tuple) or isinstance (b, list):
                assert(type(a) == type(b))
                assert(len(a) == len(b))
                for i in range(len(a)):
                    bench_allclose(a[i], b[i])
            else:
                raise RuntimeError("Encountered an supported type.\n" +
                    "Please add the type or override `bench_allclose`")


        try:
            opt = model(*inputs)
        except Exception as e:
            print(e)
            warnings.warn(UserWarning(f"{model_name}.eval() doesn't support `check_results` yet!"))
            return

        # disable optimizations and force a recompilation
        # to a baseline version
        fwd = model._c._get_method("forward")
        fwd._debug_flush_compilation_cache()
        torch._C._set_graph_executor_optimize(False)
        base = model(*inputs)
        torch._C._set_graph_executor_optimize(True)

        bench_allclose(base, opt)

    def enable_channels_last(self):
        model_name = self.name
        try:
            model, _ = self.get_module()
            model = model.to(memory_format=torch.channels_last)
        except RuntimeError:
            warnings.warn(UserWarning(f"{model_name} doesn't support `channels_last` yet!"))
            return
        self.set_module(model)
        def inputs_convert(example_inputs):
            if isinstance(example_inputs, torch.Tensor) and example_inputs.dim()==4:
                return example_inputs.to(memory_format=torch.channels_last)
            elif isinstance(example_inputs, (tuple, list, dict)):
                return tree_map(lambda x: inputs_convert(x), example_inputs)
            else:
                warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `channels_last`!"))
                return example_inputs
        if hasattr(self, 'example_inputs'):
            self.example_inputs = inputs_convert(self.example_inputs)
        else:
            warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `channels_last`!"))

    def enable_fx_int8(self, quant_engine:str='x86'):
        torch.backends.quantized.engine = quant_engine
        try:
            model, _ = self.get_module()
            # Get sub modules
            model, sub_module_list = get_sub_module(model, dict(model.named_modules()), '')
            if not len(sub_module_list):
                warnings.warn(UserWarning(f"{self.name} doesn't have submodule can ben quantized!"))
            model = prepare_sub_module(sub_module_list, model, '', quant_engine)
            self.set_module(model)
            # Calibration
            self.eval()
            model, _ = self.get_module()
            model = convert_sub_module(sub_module_list, model, '')
            self.set_module(model)
        except Exception as e:
            print(e)
            raise RuntimeError(f"{self.name} doesn't support `fx_int8` yet!")

    def enable_bf16(self):
        model_name = self.name
        try:
            model, _ = self.get_module()
            model = model.to(torch.bfloat16)
        except RuntimeError:
            warnings.warn(UserWarning(f"{model_name} doesn't support `to(torch.bfloat16)` yet!"))
            return
        self.set_module(model)
        def inputs_convert(example_inputs):
            if isinstance(example_inputs, torch.Tensor) and example_inputs.dtype == torch.float32:
                return example_inputs.to(torch.bfloat16)
            elif isinstance(example_inputs, (tuple, list, dict)):
                return tree_map(lambda x: inputs_convert(x), example_inputs)
            else:
                warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `torch.bfloat16`!"))
                return example_inputs
        if hasattr(self, 'example_inputs'):
            self.example_inputs = inputs_convert(self.example_inputs)
        else:
            warnings.warn(UserWarning(f"{model_name} example inputs doesn't convert to `torch.bfloat16`!"))

    def enable_amp(self):
        if not self.dynamo and self.opt_args.backend == 'cudagraph':
            return NotImplementedError("AMP not implemented for cudagraphs")
        if not hasattr(self, "amp_context"):
            raise RuntimeError(f"{self.name} doesn't have amp_context support!")
        if self.device == "cpu":
            self.amp_context = lambda: torch.cpu.amp.autocast()
        elif self.device == "cuda":
            self.amp_context = lambda: torch.cuda.amp.autocast()
