from tqdm.auto import tqdm
import operator
import inspect
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from .utils import *

# Main profiling logic
def run_fx_op_profiler(model, input_sample, device="cuda"):
    traced = symbolic_trace(model)
    meta_info_map = build_meta_info(traced, input_sample)

    results = []
    for node in tqdm(traced.graph.nodes):
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue

        try:
            inputs, kwargs = resolve_inputs(node, meta_info_map, device)
            if not inputs:
                results.append((node.name, "no valid inputs", "", "", 0.0))
                continue

            if node.op == "call_function":
                if node.target == operator.getitem and isinstance(inputs[0], torch.Tensor):
                    avg_time = safe_profile_op(lambda x: x[inputs[1]], inputs[0], device=device)
                else:
                    avg_time = safe_profile_op(node.target, *inputs, **kwargs, device=device)
                results.append((node.name, node.op, str(node.target), [i.shape for i in inputs if isinstance(i, torch.Tensor)], avg_time))

            elif node.op == "call_method":
                method = node.target
                tensor = inputs[0]
                method_args = inputs[1:]
                avg_time = safe_profile_op(getattr(tensor, method), *method_args, **kwargs, device=device)
                results.append((node.name, node.op, method, [t.shape for t in inputs if isinstance(t, torch.Tensor)], avg_time))

            elif node.op == "call_module":
                submod = dict(model.named_modules())[node.target]
                for attr in dir(submod):
                    value = getattr(submod, attr)
                    if isinstance(value, nn.Parameter) and value is not None:
                        setattr(submod, attr, nn.Parameter(torch.randn(value.shape, device=device)))
                submod = submod.eval().cuda()
                avg_time = safe_profile_op(submod, *inputs, **kwargs, device=device)
                results.append((node.name, node.op, str(type(submod)), [i.shape for i in inputs if isinstance(i, torch.Tensor)], avg_time))
                clear_tensors(submod.parameters())
            
            clear_tensors(inputs)

        except Exception as e:
            results.append((node.name, f"\033[31m{str(e)}\033[0m", "", "", 0.0))
            continue

    return results


def safe_profile_op(op_fn, *args, fallback_shape=(1,), device="cuda", **kwargs):
    try:
        # filter
        sig = inspect.signature(op_fn)
        valid_kwargs = {
            k: v for k, v in kwargs.items()
            if k in sig.parameters and sig.parameters[k].kind in [
                inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
            ]
        }
        return profile_op(op_fn, *args, **valid_kwargs)
    except Exception as e:
        try:
            dummy_input = torch.randn(fallback_shape, device=device)
            return profile_op(lambda x: x + 1, dummy_input)
        except:
            return 0.0

def profile_op(op_fn, *args, warmup=2, repeat=10, **kwargs):
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = op_fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeat):
        _ = op_fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / repeat  # ms

# Run shape propagation and collect output shapes of each node
def build_meta_info(traced, input_sample):
    meta_input = to_meta(input_sample)
    ShapeProp(traced).propagate(*meta_input)
    meta_info_map = {}
    for node in traced.graph.nodes:
        meta = node.meta.get("tensor_meta", None)
        if hasattr(meta, "shape"):
            meta_info_map[node.name] = [meta]
        elif isinstance(meta, (list, tuple)):
            meta_info_map[node.name] = [m for m in meta if hasattr(m, "shape")]
        else:
            pass
    return meta_info_map

# Extract the input shapes to a given node based on args/kwargs
def resolve_inputs(node, meta_info_map, device):
    def extract(val):
        if isinstance(val, torch.fx.Node):
            meta = meta_info_map.get(val.name)
            if meta:
                shape = tuple(meta[0].shape)
                dtype = meta[0].dtype
                if dtype in [torch.float, torch.float32, torch.float64]:
                    return torch.randn(shape, dtype=dtype, device=device)
                elif dtype in [torch.int64, torch.int32]:
                    return torch.randint(0, 2, shape, dtype=dtype, device=device)
                elif dtype == torch.bool:
                    return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
                else:
                    return torch.zeros(shape, dtype=dtype, device=device)  # fallbackelse:
            else:
                # fallback: dummy tensor to prevent getitem(None, ...)
                return torch.randn(1, device=device)
        elif isinstance(val, (int, float, tuple, list)):
            return val
        return None

    inputs = []
    for arg in node.args:
        if isinstance(arg, (tuple, list)):
            inputs.append(type(arg)(extract(a) for a in arg))
        else:
            inputs.append(extract(arg))

    kwargs = {}
    for k, v in node.kwargs.items():
        if isinstance(v, torch.fx.Node):
            meta = meta_info_map.get(v.name)
            if meta:
                shape = tuple(meta[0].shape)
                kwargs[k] = torch.randn(shape, device=device)
        elif isinstance(v, (int, float, tuple, list)):
            kwargs[k] = v

    return inputs, kwargs

# Build input tensors based on inferred shapes
def build_fake_inputs(shape_list, device):
    return [torch.randn(shape, device=device) for shape in shape_list]

def print_tabular(results):
    from tabulate import tabulate
    headers = ["name", "op_type", "target", "shape", "time (ms)"]
    table = [
        [name, op_type, target, shape, f"{t:.3f}"] for name, op_type, target, shape, t in results
    ]
    print(tabulate(table, headers=headers, tablefmt="github"))