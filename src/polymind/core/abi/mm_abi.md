# Polymind Multi-Modal ABI (MM-ABI) v1.1

Polymind is our multi-modal model. This ABI defines metadata for projector/encoder components and adopts TinyLLaVA-style deltas.

Version: 1.1 (backward compatible with v1.0)

What's new in v1.1:
- projector metadata includes:
  - recipe_hint: enum {base, share}
  - encoder_trainable: enum {none, norms, last_k_blocks}
  - t_cap: int (canonical). Deprecated alias: T_cap
- Sequence policy note: partial vision-encoder finetuning allowed when core ≤ 3B.

Migration from v1.0:
- `T_cap` remains accepted but will be normalized to `t_cap` with a deprecation warning. Update manifests to use `t_cap`.

Required/Optional Fields (projector):
- t_cap (int) [required if capacity limiting is used]
- d_model (int) [required]
- recipe_hint (string enum: base|share) [optional]
- encoder_trainable (string enum: none|norms|last_k_blocks) [optional]
