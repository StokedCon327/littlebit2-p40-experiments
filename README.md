# LittleBit-2 on an NVIDIA Tesla P40: early experimental results

**Status:** Two quantization-aware training smoke tests completed on November 21, 2026, including packed checkpoint verification and a separate reload/inference test for OPT-125M. **These are engineering feasibility tests, not quality or performance benchmarks.**

This independent research dataset records experiments using [Samsung Research LittleBit](https://github.com/SamsungLabs/LittleBit) at upstream commit `42d658b`. Samsung's repository is subject to **CC BY-NC 4.0**. We do not redistribute their source code or pretrained/derived weights here.

## Summary

| Base model | Converted linear layers | QAT steps | Training loss (step 1 → 2) | Packed checkpoint bytes | Reload + finite-logit GPU inference |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny random OPT | 30 | 2 | 0.112128 → 0.111123 | 3,353,150 | Not separately tested |
| Facebook OPT-125M | 72 | 2 | 103.080460 → 66.014114 | 167,764,818 | **Passed** |

The 0.55 bits-per-weight **target applies to converted linear layers**, not to the entire model. The checkpoints include other, less-compressed tensors, and file sizes are not total model-size reduction ratios. A declining loss over two toy batches is not evidence that the resulting model is useful.

## Hardware and software

- NVIDIA Tesla P40 (Pascal, compute capability 6.1), NVIDIA driver 580.178.04.
- Python 3.10.12; PyTorch 2.6.0+cu124; Transformers 4.51.3; Datasets 3.4.1.
- PyTorch job process memory-fraction cap: 22% of a 24 GiB GPU, with production workloads on the same card.
- No BF16 or DeepSpeed used in these smoke tests. FP32 model tensors were used.
- LittleBit-2 Joint-ITQ initialized with two ITQ iterations, SmoothSign, residual disabled, target 0.55 BPW.

## What's in this repository

- `results.json`: Structured experimental environment, methods, measurements and caveats.
- `results.csv`: Compact tabular results for analysis.
- `METHODS.md`: Experiment procedure and verification steps.
- `LICENSE-NOTES.md`: Attribution and redistribution/usage limitations.

## What the results actually show

- The official LittleBit quantization components can be imported and exercised with CUDA/PyTorch on Pascal architecture in an isolated environment.
- Two forward/backward/optimizer steps completed for each model; packed tensors were serialized and read back.
- A separately exported OPT-125M packed checkpoint was reloaded using a CPU-first workaround; its output logits were finite and had vocabulary dimension 50,272.
- Native model serving via Strata/Ollama, end-to-end model quality, generalization, compression ratios, sustained training, and throughput **remain unverified**.

## Source and scope

Samsung Research / SamsungLabs: https://github.com/SamsungLabs/LittleBit. This independent benchmark report is **not an official Samsung benchmark**, and results are from the stated setup only. No private hostnames, IP addresses, credentials, prompts from household members, or trained weight files are included.

For an honest comparison, the next phase should train on a specified public corpus, evaluate an untouched base model and its quantized counterpart on held-out text, measure achieved effective bits/weight with a consistent definition, and report batch size, tokens/s, peak VRAM and repeated-run variance.
