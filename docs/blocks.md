# Polymind Blocks Plugin Architecture

Polymind supports external transformer/attention blocks via a simple registry.

- Implement a block class with signature `(d_model: int, n_heads: int)`
- Register it with `@register_block("my_block_v1")`
- Publish your package independently and import it before model creation

Example:
```python
from polymind.core.registry import register_block
import torch.nn as nn

@register_block("my_block_v1")
class MyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        # ... your attention + MLP ...
    def forward(self, x):
        return x
```

Use it by setting in YAML:
```yaml
model:
  name: polymind
  block: my_block_v1
```
