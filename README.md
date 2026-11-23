# LittleBit-2 on an NVIDIA Tesla P40: early experimental results

**Status:** Two quantization-aware training smoke tests completed on November 21, 2026, including packed checkpoint verification and a separate reload/inference test for OPT-125M. **These are engineering feasibility tests, not quality or performance benchmarks.**

This independent research dataset records experiments using [Samsung Research LittleBit](https://github.com/SamsungLabs/LittleBit) at upstream commit `42d658b`. Samsung's repository is subject to **CC BY-NC 4.0**. We do not redistribute their source code or pretrained/derived weights here.

## Summary

| Base model | Converted linear layers | QAT steps | Training loss (step 1 → 2) | Packed checkpoint bytes | Reload + finite-logit GPU inference |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny random OPT | 30 | 2 | 0.112128 → 0.111123 | 3,353,150 | Not separately tested |
| Facebook OPT-125M | 72 | 2 | 103.080460 → 66.014114 | 167,764,818 | **Passed** |

The 0.55 bits-per-weight **target applies to converted linear layers**, not to the entire model. The checkpoints include other, less-compressed tensors, and file sizes are not total model-size reduction ratios. A declining loss over two toy batches is not evidence that the resulting model is useful.

## Held-out comparison: first actual language-quality check

On November 21, 2026, two independent GPU evaluations compared the unmodified **OPT-125M** with the existing **two-step synthetic LittleBit-2 checkpoint** on exactly the same 4,064 **WikiText-2 validation tokens** (128-token independent blocks).

| Model | Block-based perplexity (lower is better) |
| --- | ---: |
| Original OPT-125M | **67.12** |
| Two-step LittleBit-2 smoke checkpoint | **17,325.74** |

The second run reproduced the perplexity values exactly. The quantized checkpoint's held-out quality is **severely degraded**, which is unsurprising after only two synthetic optimizer steps. These are *not standard published WikiText-2 sliding-window perplexities*, and **do not measure LittleBit-2's potential after proper training**. The WikiText-2 test split remains untouched. No new public-corpus QAT training or model serving integration was performed.

See [heldout-validation-2026-11-21.json](heldout-validation-2026-11-21.json) for the structured method, measurements and caveats. This file contains independently measured numbers only, no vendor code, checkpoint weights or private system information.

## November 22–23: storage and reproducibility follow-up

The experimental Pascal GPU source scripts have now been shared separately from Samsung's upstream implementation:

- [scripts/train_pascal_smoke.py](scripts/train_pascal_smoke.py) runs the FP32, two-step synthetic QAT smoke experiment.
- [scripts/verify_pascal_checkpoint.py](scripts/verify_pascal_checkpoint.py) loads the locally generated OPT-125M packed checkpoint with the CPU-first workaround and validates finite CUDA logits.
- [packing-fp16-embedding-2026-11-23.json](packing-fp16-embedding-2026-11-23.json) records a **storage-only** tied-embedding/FP16 packaging experiment.

| Disk packaging metric | Recorded result |
| --- | ---: |
| Initial packed PyTorch checkpoint | 159.99 MiB |
| Optimized FP16 embedding checkpoint | 86.35 MiB |
| Disk reduction versus initial checkpoint | **46.03%** |
| CPU loader-logit comparison | **Exact match in recorded test** |
| Increased model parameter count | **No** |
| Native Strata/Ollama loading verified | **No** |

**The checkpoint packaging result does not establish a 2×-capacity model or better inference quality.** The 125M nominal parameter count is unchanged, and the optimized checkpoint is not distributed here. CPU equivalence does not establish GPU serving performance.

A subsequent *CPU-only* test of the actual LittleBit quantized training-factor and optimizer-state resume mechanism passed: restored logits matched exactly after a one-step synthetic training snapshot (maximum absolute difference 0.0). The public-text **32-step quality-recovery QAT pilot has been staged, not launched or completed**. No meaningful post-QAT quality-recovery claim is made.

### Reproduce the original smoke test

Install the official Samsung upstream source at the pinned commit in a **separate** directory and install the version-pinned requirements in an isolated virtual environment. Follow [REPRODUCE.md](REPRODUCE.md) before running anything on a GPU. Example, from the Samsung repo root with its local Python environment activated:

```bash
PYTHONPATH=. python ../littlebit2-p40-experiments/scripts/train_pascal_smoke.py \
  --model-id facebook/opt-125m \
  --output ./outputs/opt125m-itq-smoke \
  --steps 2 --eff-bit 0.55 --itq-iterations 2 \
  --memory-fraction 0.22
```

The example assumes the research repository is a sibling of the Samsung checkout; adjust paths as necessary. The script downloads public model files unless already cached. It is **not** safe to use on a shared production GPU without confirming available VRAM and exclusive authorization to run training.

## Hardware and software

- NVIDIA Tesla P40 (Pascal, compute capability 6.1), NVIDIA driver 580.178.04.
- Python 3.10.12; PyTorch 2.6.0+cu124; Transformers 4.51.3; Datasets 3.4.1.
- PyTorch job process memory-fraction cap: 22% of a 24 GiB GPU, with production workloads on the same card.
- No BF16 or DeepSpeed used in these smoke tests. FP32 model tensors were used.
- LittleBit-2 Joint-ITQ initialized with two ITQ iterations, SmoothSign, residual disabled, target 0.55 BPW.

## What's in this repository

- `results.json`: Structured experimental environment, methods, measurements and caveats.
- `heldout-validation-2026-11-21.json`: First repeatable public-text validation comparison of the original OPT-125M and the two-step smoke checkpoint.
- `results.csv`: Compact tabular results for analysis.
- `METHODS.md`: Experiment procedure and verification steps.
- `LICENSE-NOTES.md`: Attribution and redistribution/usage limitations.
- `REPRODUCE.md`: Minimal isolated setup, preflight and commands.
- `scripts/`: Standalone Pascal-compatible smoke training and checkpoint-reload scripts.
- `packing-fp16-embedding-2026-11-23.json`: Storage optimization and CPU-parity report.

## What the results actually show

- The official LittleBit quantization components can be imported and exercised with CUDA/PyTorch on Pascal architecture in an isolated environment.
- Two forward/backward/optimizer steps completed for each model; packed tensors were serialized and read back.
- A separately exported OPT-125M packed checkpoint was reloaded using a CPU-first workaround; its output logits were finite and had vocabulary dimension 50,272.
- Native model serving via Strata/Ollama, generalization after proper QAT, effective whole-model compression, and sustained performance **remain unverified**. The two-step smoke checkpoint's first measured held-out quality is very poor.

## Source and scope

Samsung Research / SamsungLabs: https://github.com/SamsungLabs/LittleBit. This independent benchmark report is **not an official Samsung benchmark**, and results are from the stated setup only. No private hostnames, IP addresses, credentials, prompts from household members, or trained weight files are included.

For an honest comparison, the next phase should train on a specified public corpus, evaluate an untouched base model and its quantized counterpart on held-out text, measure achieved effective bits/weight with a consistent definition, and report batch size, tokens/s, peak VRAM and repeated-run variance.
