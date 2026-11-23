# Reproduce the Pascal/P40 engineering smoke tests

This repository contains independently authored *experimental glue scripts* for Samsung Research's LittleBit-2. It does not distribute Samsung source code, pretrained weights, data corpora, or checkpoints. The original vendor implementation carries a **CC BY-NC 4.0** license; follow that license and the base-model license.

## Tested platform

- NVIDIA Tesla P40, compute capability 6.1 and working driver 580.178.04.
- Python 3.10.12, `torch==2.6.0+cu124`, `transformers==4.51.3`, `datasets==3.4.1`.
- GPU available VRAM at least **7 GiB**, preferably more. Check other running services first. A PyTorch per-process allocation limit does **not** reserve capacity or provide complete protection against resource contention.
- FP32 computation. P40 does not support native BF16.

## Checkout and environment

Choose an isolated directory on a machine with adequate disk space; do not modify production model environments. The following layout places the two repositories beside each other:

```bash
mkdir -p littlebit-rnd
cd littlebit-rnd
git clone https://github.com/SamsungLabs/LittleBit.git
git -C LittleBit checkout 42d658b0c79f76450b34b6a3547462c7cdc3e1a0
git clone https://github.com/StokedCon327/littlebit2-p40-experiments.git

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
python -m pip install 'numpy==2.2.4' 'transformers==4.51.3' \
  'datasets==3.4.1' 'accelerate==1.6.0' 'GPUtil==1.4.0' \
  'sentencepiece' 'tensorboardX'
```

If your Ubuntu installation lacks `python3.10-venv`, install it through your normal administrator process, or create an environment with an approved user-local virtualenv tool. Avoid replacing system Python.

The reproduction scripts import quantization modules from the pinned Samsung repo; run them from the Samsung repo root:

```bash
cd LittleBit

# Read-only preflight before using the GPU
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print("free VRAM bytes", torch.cuda.mem_get_info(0)[0])
PY

# Very small random-model pipeline test
PYTHONPATH=. python ../littlebit2-p40-experiments/scripts/train_pascal_smoke.py \
  --model-id hf-internal-testing/tiny-random-OPTForCausalLM \
  --output ./outputs/tiny-opt-itq-smoke \
  --steps 2 --eff-bit 0.55 --itq-iterations 2 \
  --memory-fraction 0.22

# Pretrained OPT-125M pipeline test; only if enough GPU memory is available
PYTHONPATH=. python ../littlebit2-p40-experiments/scripts/train_pascal_smoke.py \
  --model-id facebook/opt-125m \
  --output ./outputs/opt125m-itq-smoke \
  --steps 2 --eff-bit 0.55 --itq-iterations 2 \
  --memory-fraction 0.22
```

These are **short synthetic demonstrations**, not full QAT training, and may download public models from Hugging Face. Do not trust retrieved model code or substitute unreviewed remote code; the scripts set `trust_remote_code=False`.

## Verify the packed checkpoint

The training script saves a packed `checkpoint.pt`, configuration and tokenizer; the inference-verification script requires an additional `model.safetensors` export because it uses Samsung's safetensors packed-weight loader. Produce that file by exporting the `weights_only=True` loaded checkpoint:

```bash
python - <<'PY'
from pathlib import Path
import torch
from safetensors.torch import save_file
folder=Path('outputs/opt125m-itq-smoke')
state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
save_file({k: v.detach().contiguous().clone() for k,v in state.items()}, folder/'model.safetensors')
print('exported', folder/'model.safetensors')
PY

PYTHONPATH=. python ../littlebit2-p40-experiments/scripts/verify_pascal_checkpoint.py \
  --checkpoint-dir ./outputs/opt125m-itq-smoke --memory-fraction 0.22
```

The duplicate-storage issue in straightforward safetensors export may produce a file **larger** than the packed PyTorch checkpoint. This is not optimized model compression. The separate packaging experiment is documented in `packing-fp16-embedding-2026-11-23.json`.

## Integrity and limitations

- Confirm both models report completed two-step checkpoints before comparison.
- Do not interpret the `0.55` bit target as whole-model effective BPW.
- Reproduce held-out quality evaluation on a distinct validation set, not the training sentences. Current two-step quantized checkpoint had very poor held-out quality compared with the unmodified model.
- Do not publish checkpoint weights or model downloads under this repository without checking distribution licenses and privacy.
- Native Strata/Ollama serving and long-run GPU stability remain unverified.
- All scripts are research code. Run them only in isolated environments, with model downloads and GPU workloads explicitly authorized.
