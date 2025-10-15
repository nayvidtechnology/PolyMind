# MM-ABI v1.1

This document defines the Multi-Modal ABI (Application Binary Interface) metadata for projector/encoder components.

Version: 1.1 (backward compatible with v1.0)

What's new in v1.1:
- projector metadata now includes:
  - recipe_hint: enum {base, share}
  - encoder_trainable: enum {none, norms, last_k_blocks}
  - t_cap: int (canonical). Deprecated alias: T_cap
- Sequence policy note: partial vision-encoder finetuning allowed when core ≤ 3B.

Migration from v1.0:
- If your manifests used `T_cap`, it remains accepted but will be normalized to `t_cap` with a deprecation warning. Update your manifests to use `t_cap`.

Required/Optional Fields (projector):
- t_cap (int) [required if capacity limiting is used]
- d_model (int) [required]
- recipe_hint (string enum: base|share) [optional]
- encoder_trainable (string enum: none|norms|last_k_blocks) [optional]

Notes:
- For cores ≤ 3B parameters, partial finetuning of the vision encoder is allowed as per `encoder_trainable`.
