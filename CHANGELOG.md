# Changelog

All notable changes to this project will be documented in this file.

## 1.1.0 - 2025-10-15

- feat(abi): MM‑ABI v1.1 — projector knobs & unfreeze policy
  - Added projector fields: `recipe_hint`, `encoder_trainable`, and canonical `t_cap` (deprecated alias `T_cap` still accepted)
  - Added Python validators that normalize `T_cap` to `t_cap` with a deprecation warning
  - JSON schema updated to validate both v1.0 and v1.1 metadata
