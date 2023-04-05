import torch
import torch._dynamo as dynamo
import logging

torch._dynamo.config.verbose=True                # TORCHDYNAMO_VERBOSE=True
torch._dynamo.config.output_code=True
# torch._dynamo.config.output_graph_code=True   # fails with naive usage in xla container
torch._dynamo.config.log_level = logging.DEBUG   # Better to set TORCH_COMPILE_DEBUG?

import torch._functorch.config
torch._functorch.config.debug_partitioner=True   # AOT_PARTITIONER_DEBUG=True
# torch._functorch.config.debug_graphs=True        # AOT_FX_GRAPHS=True # fails with naive usage in xla container
# torch._functorch.config.debug_joint=True         # AOT_FX_GRAPHS_JOINT=True # fails with naive usage in xla container
# torch._functorch.config.log_level=logging.DEBUG  # Automatically set to DEBUG if any of above env is set # fails with naive usage in xla container

import torch._inductor.config
torch._inductor.config.verbose_progress=True
torch._inductor.config.debug=True
torch._inductor.config.trace.enabled=True        # TORCH_COMPILE_DEBUG=1
torch._inductor.config.trace.info_log=True
torch._inductor.config.trace.graph_diagram=True
#torch._inductor.config.pattern_matcher=False
#torch._inductor.config.aggressive_fusion=True
#torch._inductor.config.permute_fusion=True      # TORCHINDUCTOR_PERMUTE_FUSION=1

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    #@dynamo.optimize()
    def forward(self, x):
        x = torch.permute(x, (1,0))
        x = torch.add(x, x[2,:])
        return x

model = MyModel()
#model = torch.compile(model, fullgraph=True)  // an exception occurs when this is executed.
model = torch.compile(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.tensor([[0.,1., 2., 3.],
                  [4.,5., 6., 7.],
                  [8.,9.,10.,11.]], device=device)
y = model(x)
