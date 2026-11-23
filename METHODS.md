# Methods and reproducibility notes

## Environment

On a Tesla P40 GPU (compute capability 6.1), use an **isolated virtual environment** with `torch==2.6.0+cu124`, `transformers==4.51.3`, `datasets==3.4.1`, `accelerate==1.6.0`, NumPy 2.2.4 and GPUtil 1.4.0. The SamsungLabs/LittleBit upstream commit was `42d658b`. Ensure PyTorch's wheel includes compatible Pascal CUDA kernels. Verify CUDA tensor arithmetic on the selected GPU before training.

## Procedure

1. Load each pretrained model and tokenizer. Construct a frozen FP32 teacher and an FP32 student from the same pretrained checkpoint.
2. Apply upstream `quantization.utils.apply_littlebit_patch` to the student, with `quant_func=SmoothSign`, `quant_mod=LittleBitLinear`, `eff_bit=0.55`, `residual=False`, `min_split_dim=8`, `kv_factor=1.0`, `use_itq=True`, and `itq_n_iter=2`.
3. Freeze non-LittleBit parameters; optimize converted module parameters with AdamW (LR 2e-5), using KL divergence of student and teacher logits plus mean hidden-state MSE. This was a **custom smoke-test training loop**, not Samsung's full default `main.py` path.
4. Tokenize four fixed generic English sentences; run two optimizer steps with a single short training example and padded sequence length 48 per step. This dataset is not a realistic training corpus or held-out evaluation set.
5. Save a PyTorch packed checkpoint and read it back using `torch.load(weights_only=True)`; check tensor/key counts.
6. For OPT-125M, export the packed state to safetensors and reconstruct the model on CPU using its config, patch the quantized modules for inference, unpack weights with upstream `_load_and_process_state_dict`, load with `assign=True`, assert no missing/unexpected weights, then run a GPU forward pass and confirm finite next-token logits.

## Adaptations and caveats

Samsung's default `main.py` uses `bf16=True` and DeepSpeed; the P40 does not support native BF16, so this smoke test used FP32 and a small PyTorch loop. Samsung's stock checkpoint loader hit a meta-tensor import/initialization error in this environment; an experimental CPU-first loader succeeded instead. These adaptations are not patches claimed to be accepted upstream.

The user-facing training loss values are from different models and are not cross-model-comparable. The output of `torch.cuda.memory_reserved` after a step was 102 MiB (tiny model) and 1,666 MiB (OPT-125M); these are **observed reserved-memory readings**, not measured peak consumption across all processes. Total GPU VRAM usage was shared with a separate production inference service.

## Follow-up validation needed

- Quality: held-out perplexity, benchmark accuracy, and fixed prompt evaluations against each base model.
- Compression: actual exported storage, tensor-level realized effective BPW, tied weights, and runtime footprint.
- Reliability: repeated runs, timeouts, memory stability, checkpoint restore and longer training.
- Compatibility: independent serving adapter, API integration and native backend support.

**Do not treat these snapshots as a trained assistant or publish compression claims based on their checkpoint sizes.**
## Released reproduction tools (2026-11-23)

The independent Pascal-compatible training and inference-verification scripts now appear under [scripts/](scripts/). The training script was copied from the executed, successful Thing2 experiment, and both staged scripts passed Python syntax checks and `--help` import checks. They have **not** been rerun end-to-end from a fresh public clone; that remains an independent reproduction milestone. The inference script accepts a checkpoint directory instead of the prior host-specific absolute path. See [REPRODUCE.md](REPRODUCE.md).

## Subsequent engineering measurements (not model-quality milestones)

A storage-only optimization of the existing OPT-125M packed checkpoint reduced the recorded file size from 159.993 MiB to 86.352 MiB, a ratio of 0.53973. The packaging change used FP16 embedding storage and tied weights. The recorded CPU loader test produced exactly matching logits (max absolute difference 0). The experiment did **not** increase model parameters or establish faster model execution; see [packing-fp16-embedding-2026-11-23.json](packing-fp16-embedding-2026-11-23.json).

On November 23, a CPU-only re-execution of the actual LittleBit latent-factor QAT resume test succeeded: after a one-step synthetic optimizer snapshot, restored logits matched the prior model exactly (max absolute difference 0), and optimizer state counts matched. The existing resume-test implementation is currently tied to a local research workspace and is **not yet included** among the portable public scripts. The 32-step WikiText-2 public QAT pilot is staged locally but not verified as completed.

## Quality evaluation already published

The independent 4,064-token WikiText-2 validation block-based perplexity evaluation (two repeat runs) reported **67.12** for original OPT-125M and **17,325.74** for the two-step LittleBit checkpoint. These are non-overlapping 128-token blocks, not conventional sliding-window test perplexity. The result demonstrates catastrophic quality degradation of the minimally trained checkpoint and should prevent misinterpreting the earlier training or storage successes as a deployable model. See [heldout-validation-2026-11-21.json](heldout-validation-2026-11-21.json).
