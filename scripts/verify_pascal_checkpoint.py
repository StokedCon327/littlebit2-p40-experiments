#!/usr/bin/env python3
"""Pascal-safe LittleBit-2 packed-checkpoint inference validation."""
import argparse, json, torch
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from quantization.utils import apply_littlebit_patch
from quantization.utils.quant_util import _load_and_process_state_dict
from quantization.modules import LittleBitLinear

cli = argparse.ArgumentParser(description="Verify a saved LittleBit-2 OPT-125M checkpoint")
cli.add_argument("--checkpoint-dir", type=Path, required=True,
                 help="Directory with model.safetensors, config.json and tokenizer")
cli.add_argument("--memory-fraction", type=float, default=0.22)
args_cli = cli.parse_args()
if not (0 < args_cli.memory_fraction <= 0.25):
    cli.error("Memory fraction must be between 0 and 0.25")
root = args_cli.checkpoint_dir.expanduser().resolve()
if not root.is_dir():
    cli.error("Checkpoint directory not found")
torch.set_num_threads(4)
assert torch.cuda.mem_get_info()[0] > 7*1024**3, 'Insufficient free GPU memory'
torch.cuda.set_per_process_memory_fraction(args_cli.memory_fraction)
args=SimpleNamespace(model_id='facebook/opt-125m', quant_func='SmoothSign',
 quant_mod='LittleBitLinear', residual=False, eff_bit=0.55, split_dim=1024,
 kv_factor=1.0, min_split_dim=8, use_itq=True, itq_n_iter=2)
cfg=AutoConfig.from_pretrained(str(root), trust_remote_code=False)
# Instantiate OPT on CPU, not under global torch.device("meta"), avoiding
# torch._dynamo's meta-tensor import initialization problem.
model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=False)
model=apply_littlebit_patch(model,args,do_train=False)
weights,waspacked=_load_and_process_state_dict(str(root),torch.float32)
missing,unexpected=model.load_state_dict(weights,strict=False,assign=True)
assert not missing, f'Missing checkpoint weights: {missing[:8]}'
assert not unexpected, f'Unexpected checkpoint keys: {unexpected[:8]}'
model.tie_weights()
for layer in model.modules():
 if isinstance(layer,LittleBitLinear):
  layer._binarized=waspacked
for name,param in model.named_parameters():
 assert not param.is_meta,f'Uninitialized meta tensor: {name}'
model.eval().to('cuda')
tokenizer=AutoTokenizer.from_pretrained(str(root),trust_remote_code=False)
query='The technician verified the'
with torch.no_grad():
    logits=model(**tokenizer(query,return_tensors='pt').to('cuda')).logits[0,-1]
assert torch.isfinite(logits).all(),'Non-finite output'
next_token=tokenizer.decode([int(logits.argmax())])
payload={'status':'INFERENCE_RELOAD_PASS','model':'facebook/opt-125m',
         'packed_reloaded':bool(waspacked),'layers':sum(isinstance(x,LittleBitLinear) for x in model.modules()),
         'logit_shape':list(logits.shape),'example_next_token':next_token}
(root/'inference_result.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload),flush=True)

