# PolyMind
Poly for “many modalities” and “many minds” — emphasizes modular design, flexibility, and reasoning diversity

# Modular Multi‑Modal LLM — Architecture, Design & Delivery Playbook

**Author:** Divyang (Solution Architect)  
**Version:** 1.0 (MM‑ABI v1.0)  
**Scope:** Laptop & Mobile on‑device multi‑modal LLM with cloud‑optional training; modular components that teams can train independently and assemble via strict interfaces; publishable artifacts for local and app embedding.

---

## 0) Executive Summary
A compact, modular, multi‑modal LLM stack built around a strict **MM‑ABI** (Application Brain Interface). Teams produce swappable **Encoders/Projectors/Decoders/Skills** that plug into a small **Core LLM** (2–4B). An Assembly Graph describes a build (Perception‑first, Math‑first, Audio‑first). The stack ships consistent runtimes (GGUF, ONNX, CoreML, ExecuTorch) and a cloud‑agnostic training pipeline (AWS/GCP/Azure) with experiment tracking, checkpoints, and CI gates.

Targets:  
- Laptop (NVIDIA/AMD/Intel GPU, NPU)  
- Android (NNAPI/Qualcomm HTP) / iOS (ANE)  
- Optional server inference (Triton, OpenVINO, vLLM)

Latency budgets: text <300 ms first token; image caption <900 ms; chart‑QA <1.8 s; short ASR <1.0 s for 5 s audio.

---

## 1) System Architecture

### 1.1 Layered Architecture
- **Layer A — Core LLM** (2–4B): causal decoder with LongRoPE, GQA, paged KV cache, speculative decoding (0.5–0.8B draft). Quant: INT4/INT8 (AWQ/GPTQ) and KV INT8/FP8.
- **Layer B — Modal Adapters**: Encoders (Vision/Audio/Video) → **Projectors** (LDP‑style) → LLM tokens; Decoders (Image/Video/TTS) for generative outputs.
- **Layer C — Skills Runtime**: Router + sandboxed Tools (Reasoning/Perception/Math/Science/Safety). Tool‑calls are JSON blocks emitted as special tokens.
- **Layer D — Runtimes**: llama.cpp (GGUF), ONNX Runtime/OpenVINO/DirectML, CoreML, ExecuTorch, plus mobile bindings.

### 1.2 Data Flow (text+image example)
1) Image → Vision Encoder → Patch Tokens `[T_enc,D_enc]`  
2) Projector → LLM Tokens `[T_cap,d_model]`  
3) Core LLM consumes `{text + <image> tokens}` and emits text / `<tool_call>` JSON  
4) Router executes tool, returns result to LLM, generation continues  
5) Optional: Image/Video/Audio Decoders for generative modalities

```
[Image] → [Vision Enc] → [Projector] →   ┐
                                          ├→ [Core LLM] → [Router/Skills] → [Outputs]
[Audio] → [Audio Enc ] → [Projector] →   ┘
```

---

## 2) MM‑ABI v1.0 (Contracts)

### 2.1 Token & Sequence Contracts
- `vocab_size`: 32000 (shared tokenizer)  
- `special_tokens`: `<bos> <eos> <pad> <image> </image> <audio> </audio> <video> </video> <tool_call> </tool_call> <scratch>`  
- `max_seq`: 4096 (includes modality tokens)  
- `d_model`: 2048 (example; configurable)  
- KV dtype: FP8/INT8 (flag)  
- RoPE scaling: LongRoPE enabled

### 2.2 Projector Contract
- **Input:** `[B, T_enc, D_enc]`  
- **Output:** `[B, T_llm ≤ T_cap, d_model]`  
- **Caps:** `T_cap` per modality (Image 64; Audio 96; Video 128 default)  
- Pooling: Adaptive 1D or learned downsampler; optional temporal block for video

### 2.3 Tool‑Call JSON Contract
Text stream embeds blocks between `<tool_call>` and `</tool_call>` containing compact JSON:
```json
{"tool":"chartqa","args":{"image_ref":"#img0","ops":["extract_table","calc_slope"]}}
```
Router must reply with JSON (no newlines) via `feed_tool_result`.

---

## 3) Module Types & Manifests

Each module ships a **`module.yaml`** manifest.

### 3.1 Encoder (Vision) — Example Manifest
```yaml
name: vision-enc
version: 1.3.0
type: encoder
modality: image
abi: mm-abi-1.0
inputs:
  - name: image
    shape: [H,W,3]
    dtype: uint8
outputs:
  - name: patch_tokens
    shape: [T_enc, D_enc]
    dtype: fp16
caps:
  patch_stride: 16
  max_res: [1024, 1024]
export:
  onnx: vision_enc.onnx
quant:
  supported: [int8, int4]
```

### 3.2 Projector (Vision) — Example Manifest
```yaml
name: vision-proj
version: 1.2.0
type: projector
modality: image
abi: mm-abi-1.0
inputs:
  - name: patch_tokens
    shape: [T_enc, D_enc]
    dtype: fp16
outputs:
  - name: llm_tokens
    shape: [T_cap, d_model]
    dtype: fp16
params:
  T_cap: 64
  d_model: 2048
export:
  onnx: vision_proj.onnx
```

### 3.3 Decoder (Image) — Example Manifest
```yaml
name: img-dec
version: 0.8.0
type: decoder
modality: image
abi: mm-abi-1.0
inputs:
  - name: latent_tokens
    shape: [T_dec, d_model]
    dtype: fp16
params:
  size: 256
  steps: 6
export:
  onnx: img_dec.onnx
```

### 3.4 Skill/Tool — Example Manifest
```yaml
name: chartqa
version: 0.6.0
type: skill
abi: mm-abi-1.0
inputs:
  - name: image_ref
    type: uri
  - name: ops
    type: list[str]
outputs:
  - name: result
    type: json
runtime:
  container: ghcr.io/yourorg/chartqa:0.6.0
  limits:
    cpu: "1"
    mem: "512Mi"
    timeout_ms: 400
```

---

## 4) Assembly Graphs (Build Recipes)
A **`build.graph.yaml`** declares the composition and exports.

```yaml
core:
  llm: {ref: core-llm@3.1.0, d_model: 2048, kv_dtype: fp8}

modalities:
  - enc: {ref: vision-enc@1.3.0}
    proj: {ref: vision-proj@1.2.0, T_cap: 64, out_dim: 2048}
  - enc: {ref: audio-enc@1.1.0}
    proj: {ref: audio-proj@1.0.0, T_cap: 96, out_dim: 2048}

decoders:
  - {ref: img-dec@0.8.0, size: 256, steps: 6}
  - {ref: tts-dec@0.7.0, codec: encodec-16khz}

skills:
  - {ref: ocr-latex@0.4.0}
  - {ref: chartqa@0.6.0}
  - {ref: sympy-lite@0.5.0}

runtime:
  router: {ref: router@1.2.0, tool_call_tokens: ["<tool_call>", "</tool_call>"]}
  sandbox: {ref: sandbox@0.3.0, timeout_ms: 300, mem_mb: 64}

exports:
  laptop:
    gguf: {quant: int4, kv: fp8}
    onnx: [vision-enc, vision-proj, img-dec, audio-enc]
  mobile-android:
    executorch: {core: int4}
    nnapi: [vision-enc, audio-enc]
  mobile-ios:
    coreml: {core: int4}
```

---

## 5) Repository & Project Structure

```
multimod/
  core/
    llm/                  # 2–4B core, rope, kv cache
    draft_llm/            # 0.5–0.8B speculative model
    adapters/             # LoRA/DoRA/QLoRA
  modalities/
    vision/encoder/
    vision/projector/
    video/
    audio/encoder/
    tts_codec/
    decoders/{image,video,audio}/
  runtime/
    router/               # tool-call dispatcher
    skills/{reasoning,perception,math_science,safety_budget}/
    sandbox/              # micropython/pyodide shim
  quant/{awq,gptq,gguf}/
  inference/runners/{llama_cpp,onnxrt,openvino,coreml,executorch}/
  eval/suites/{VQA,TextCaps,ChartQA,DocVQA,GSM8K,MATH,ASR}/
  data/{manifests,samplers,privacy}/
  iac/                    # Terraform modules per cloud
  pipelines/              # training/inference CI/CD
  scripts/{export_onnx.py,convert_coreml.py,pack_gguf.sh}
  builds/                 # build.graph.yaml files + locks
```

- **Monorepo** for interfaces + shared tooling; **sub‑repos** allowed for modules with their own CI.
- **Versioning:** SemVer; compatibility matrix stored in `builds/compat.csv`.

---

## 6) CI/CD & Checks (GitHub Actions/Azure DevOps)

### 6.1 Per‑Module CI
- **ABI tests:** validate shapes/dtypes/caps vs manifest
- **Golden I/O tests:** small canonical samples (post‑quant as well)
- **Budget tests:** latency, VRAM, power targets (laptop & mobile emu)
- **Security tests:** sandbox tools; forbid net/disk unless allowed
- **Export tests:** ONNX/CoreML/ExecuTorch conversions

### 6.2 Build CI
- Validate `build.graph.yaml` compatibility; resolve refs/versions
- Compose & export runners (GGUF/ONNX/CoreML) with hashes
- Smoke tests per target (captioning, ChartQA, ASR, math PoT)
- Publish artifacts to OCI registry + model hub (license‑aware)

---

## 7) Data & Training Pipelines (Cloud‑agnostic)

### 7.1 Stages
1) **Stage‑0 Align (freeze encoders; train projectors):** contrastive & VQA/ASR; small instruction mix  
2) **Stage‑1 I‑Tuning (LLM small updates):** multi‑modal SFT with CoT/PoT  
3) **Stage‑2 Tool Distill:** generate tool‑calls; DPO for correct routing  
4) **Stage‑3 Efficiency:** AWQ/GPTQ; KV INT8/FP8; QLoRA adapters if needed

### 7.2 Orchestration
- **Trainer:** PyTorch + FSDP/DeepSpeed ZeRO‑2; bfloat16/float8 hybrid  
- **Tracking:** MLflow or Weights & Biases; experiment lineage per module  
- **Datasets:** declarative manifests (Parquet/JSONL), filters, PII scrub  
- **Checkpoints:** modular: `s3://…/core/…`, `gs://…/vision-enc/…`, `azure://…/img-dec/…`

### 7.3 Cloud Blueprints (Terraform + YAML samples)

**AWS**  
- Compute: p3/p4 (training), g5/l4 (finetune); EKS or SageMaker Training  
- Storage: S3 (raw + curated + checkpoints); DynamoDB for manifests  
- Queue: SQS for data shards; EventBridge for pipeline triggers  
- Security: KMS keys per bucket; VPC endpoints; IAM OIDC for runners

**GCP**  
- Compute: A2/H100 or L4; Vertex AI Training & Pipelines  
- Storage: GCS buckets; BQ for data catalog; Pub/Sub for orchestration  
- Security: CMEK on buckets; Workload Identity; Artifact Registry for containers

**Azure**  
- Compute: AML Compute (ND/NC), AKS;  
- Storage: ADLS Gen2; Key Vault for secrets; Event Grid for triggers  
- Security: Managed Identity; Private Links; Purview for catalog

### 7.4 Checkpoint Policy
- **Frequency:** per‑epoch and best‑val; delta checkpoints for adapters  
- **Format:** safetensors for weights; ONNX for exports; GGUF packs  
- **Promotion:** `staging → candidate → prod` with signed manifests  
- **Repro:** hash of data manifests + code revision pinned in MLflow

---

## 8) Quantization & Export

- **LLM:** AWQ/GPTQ INT4; KV INT8/FP8; GGUF export for llama.cpp  
- **Encoders/Decoders:** ONNX INT8 (per‑channel preferred) via ORT/OpenVINO  
- **Mobile:** CoreML (weight‑only INT4/8 + ANE ops), ExecuTorch/NNAPI builds  
- **Validation:** quality deltas <2% absolute on task suites vs fp16

---

## 9) Runtimes & Integration

- **Laptop:** llama.cpp (GGUF) for core; ORT/OpenVINO for enc/dec; DirectML path on Windows, ROCm on AMD  
- **Server (optional):** NVIDIA Triton or vLLM for text; separate microservices for enc/dec  
- **Mobile:** ExecuTorch + NNAPI (Android), CoreML + BNNS/ANE (iOS)

### Tooling Integrations (Dev Assistants)
- **ChatGPT/Assistants & IDE agents:** expose a local HTTP API with schema‑first endpoints (`/assemble`, `/infer`, `/evaluate`), so coding agents can run builds, tests, and quick evals.  
- **Google‑style agent builders (Vertex/Agent Builder):** optional connectors call the same HTTP API; all privacy toggles must default to on‑device only.

---

## 10) Security, Privacy, and Safety
- **Default offline:** no outbound network from skills; allowlist per tool  
- **Data governance:** PII scrubbers in data loaders; audit logs to cloud log sinks  
- **Model safety prompts:** instruction prefixes; safety skill for blocklists  
- **Signature:** cryptographic signatures on exported artifacts & manifests

---

## 11) Evaluation & Dashboards
- **Perception:** VQA, TextCaps, ChartQA, DocVQA‑lite; EM/F1 + latency  
- **Math/Science:** GSM8K‑lite, MATH‑lite; EM, tool‑call accuracy; unit‑check pass rate  
- **Audio:** WER (mini); TTS MOS‑proxy; ms/s gen  
- **System:** first‑token latency, tok/s, VRAM, watts (laptop/mobile)

A single dashboard aggregates module versions and builds for product decisions.

---

## 12) Reference Builds

### P‑Lite (Perception‑first Laptop)
- `core-llm@3.1.0` (3B INT4), `vision-enc@1.3`, `vision-proj@1.2`, `img-dec@0.8`  
- Skills: `ocr-latex`, `chartqa`  
- Targets: caption ≤900 ms; chart slope ≤1.8 s

### M‑Lite (Math‑first Laptop/Mobile)
- `core-llm@3.1.0` (3B INT4), `ocr-latex@0.4`, `sympy-lite@0.5`  
- Speculative decoding; GSM8K‑class PoT; ≤300 ms first‑token

---

## 13) Developer Quickstarts

### 13.1 Assembly CLI
```bash
mm assemble builds/p-lite.graph.yaml \
  --export gguf=int4,kv=fp8 \
  --export onnx=vision-enc,vision-proj,img-dec \
  --out dist/p-lite/
```

### 13.2 Inference (Python)
```python
req = {
  "inputs": [
    {"type":"text","text":"Explain this plot and compute the slope."},
    {"type":"image","path":"plot.png"}
  ],
  "preferences":{"cot":true,"max_tokens":256,"budget_ms":1800},
  "tools_allowed":["chartqa","sympy"],
  "outputs":["text"]
}
for token in multimod.run(req):
    print(token, end="")
```

---

## 14) Roadmap (90 days)
- **Weeks 1–2:** finalize MM‑ABI; vision encoder + projector wired; INT4 LLM baseline  
- **Weeks 3–5:** multi‑modal SFT with CoT/PoT; OCR‑LaTeX; chart‑QA  
- **Weeks 6–8:** tool‑DPO; PoT with micropython sandbox; quant regressions  
- **Weeks 9–10:** image decoder + video keyframe+interpolate; safety budget  
- **Weeks 11–12:** mobile builds; thermal and battery gates

---

## 15) Publishing
- **Artifacts:** GGUF, ONNX, CoreML, manifests, eval cards  
- **Licensing:** per‑module SPDX IDs; third‑party model notices  
- **Distribution:** OCI registry (containers), model hub, signed checksums  
- **Docs:** quickstarts, API schemas, privacy posture

---

## 16) Appendices
- **A. `module.yaml` templates** for all module types  
- **B. Terraform module inputs/outputs for AWS/GCP/Azure**  
- **C. CI workflows (GitHub Actions) for ABI/golden/budget checks**  
- **D. Data manifest spec with PII policy**  
- **E. Safety prompt baselines and evaluation rubric**

---

## 17) Repo Scaffold Quickstart (Local)

This repository now includes a minimal runnable scaffold for agents, training configs, and multi-cloud storage adapters.

1) Create venv and install deps
  - In VS Code, run the task: "venv: install deps" (Tasks: Run Task)

2) Set a provider API key (example for OpenAI)
  - Create a `.env` file with: `OPENAI_API_KEY=...`

3) Run a sample chat locally
  - Launch config: "Run sample chat (local)" or run the CLI: `src/runtime/cli/chat.py --provider openai --message "Hello!"`

4) Dry-run training pipeline
  - Launch config: "Run training dry-run"; uses `configs/training/default.yaml`

Folders of interest:
- `configs/` — app, provider, and training configs
- `src/agents/` — provider adapters (OpenAI included)
- `src/runtime/cli/` — sample CLIs (`chat.py`, `train.py`)
- `src/storage/` — local/S3/GCS/Azure Blob adapters
- `infra/terraform/{aws,azure,gcp}` — placeholders for IaC

Note: Cloud SDK packages are optional. Install only those you need.

