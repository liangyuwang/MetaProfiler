# MetaProfiler
MetaProfiler is a lightweight, structure-agnostic operator-level profiler for PyTorch models that leverages MetaTensor execution to simulate and benchmark individual ops without loading the full model into GPU memory.

- The table below displays the GPU memory usage for profiling very large-scale models:

|        LLMs        |   1.3B   |   2.7B   |   6.7B   |   13B   |   30B   |    66B    |        175B        |
| :-----------------------: | :------: | :------: | :------: | :------: | :------: | :-------: | :-----------------: |
| **GPU memory (GB)** | `1.05` | `1.23` | `1.78` | `2.15` | `2.88` | `4.45` | **`5.85`** |

## Example Results

| name                               | op_type       | target                                           | input shape                                                                                         |   time (ms) |
|------------------------------------|---------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------|-------------|
| size                               | call_method   | size                                             | [torch.Size([4, 1024])]                                                                       |       0.01  |
| getitem                            | call_function | <built-in function getitem>                      | [torch.Size([1])]                                                                             |       0.003 |
| getitem_1                          | call_function | <built-in function getitem>                      | [torch.Size([1])]                                                                             |       0.009 |
| transformer_wpe                    | call_module   | <class 'torch.nn.modules.sparse.Embedding'>      | [torch.Size([4, 1024])]                                                                       |       0.038 |
| transformer_wte                    | call_module   | <class 'torch.nn.modules.sparse.Embedding'>      | [torch.Size([4, 1024])]                                                                       |       0.037 |
| add                                | call_function | <built-in function add>                          | [torch.Size([4, 1024, 768]), torch.Size([4, 1024, 768])]                                      |       0.048 |
...