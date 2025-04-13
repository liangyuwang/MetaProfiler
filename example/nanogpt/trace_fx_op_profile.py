import sys
sys.path.append("../MetaProfiler")

import torch
from model import GPTConfigs, GPT
from metaprofiler import run_fx_op_profiler, print_tabular


model_cfgs = GPTConfigs()
model_cfg = getattr(model_cfgs, "gpt2")
model_cfg.n_layer = 2
with torch.device('meta'):
    model = GPT(model_cfg)

B, T, V = 4, model_cfg.block_size, model_cfg.vocab_size
# profiler requires all args
meta_input = [
    torch.randint(0, V, (B, T)),       # input_ids
    torch.arange(0, T).repeat(B, 1),       # positional encodings
    torch.randint(0, V, (B, T)),   # labels
]
results = run_fx_op_profiler(model, meta_input)
print_tabular(results)