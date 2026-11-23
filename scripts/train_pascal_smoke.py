#!/usr/bin/env python3
"""Isolated Pascal GPU LittleBit-2 QAT smoke test. Not a production checkpoint."""
import argparse, json, os, time
from pathlib import Path
from types import SimpleNamespace
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantization.utils import apply_littlebit_patch
from quantization.modules import LittleBitLinear

TEXTS = [
    "The apprentice checked the fire alarm device and documented the zone.",
    "A dependable assistant confirms the action and reports the observed result.",
    "Computer systems require backups, monitoring, and careful deployment.",
    "Engineers record errors and verify a repair before declaring success.",
]
def event(name, **kwargs):
    print(json.dumps({"event": name, "at": round(time.time(), 3), **kwargs}), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="hf-internal-testing/tiny-random-OPTForCausalLM")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--eff-bit", type=float, default=0.55)
    p.add_argument("--itq-iterations", type=int, default=2)
    p.add_argument("--memory-fraction", type=float, default=0.22)
    p.add_argument("--max-length", type=int, default=48)
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "CUDA not available"
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
    free, total = torch.cuda.mem_get_info(0)
    event("preflight", gpu=torch.cuda.get_device_name(0), free_mib=round(free/1048576),
          cap_mib=round(total*a.memory_fraction/1048576), torch=torch.__version__)
    if free < 7*1024**3:
        raise RuntimeError("Under 7 GiB GPU memory available: preserving existing Ollama service")
    event("loading_tokenizer", model=a.model_id)
    tokenizer = AutoTokenizer.from_pretrained(a.model_id, trust_remote_code=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(TEXTS, return_tensors="pt", truncation=True, padding="max_length",
                       max_length=a.max_length)
    batch = {key: val[:1].to("cuda") for key, val in inputs.items() if key != "token_type_ids"}
    event("loading_teacher")
    teacher = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.float32,
                                                   trust_remote_code=False).eval().to("cuda")
    for param in teacher.parameters(): param.requires_grad_(False)
    event("loading_student")
    student = AutoModelForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.float32,
                                                   trust_remote_code=False)
    student.config.use_cache = False
    qa = SimpleNamespace(model_id=a.model_id, quant_func="SmoothSign", quant_mod="LittleBitLinear",
                         residual=False, split_dim=1024, eff_bit=a.eff_bit, kv_factor=1.0,
                         min_split_dim=8, use_itq=True, itq_n_iter=a.itq_iterations)
    event("initializing_littlebit2", itq_iters=a.itq_iterations)
    student = apply_littlebit_patch(student, qa, do_train=True)
    converted = sum(isinstance(m, LittleBitLinear) for m in student.modules())
    assert converted > 0, "No modules converted to LittleBitLinear"
    for param in student.parameters(): param.requires_grad_(False)
    for mod in student.modules():
        if isinstance(mod, LittleBitLinear):
            for param in mod.parameters(): param.requires_grad_(True)
    student.to("cuda")
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-5)
    event("initialized", converted=converted,
          trainable=sum(p.numel() for p in student.parameters() if p.requires_grad))
    with torch.no_grad():
        target = teacher(**batch, output_hidden_states=True)
    student.train()
    losses=[]
    for step in range(a.steps):
        optimizer.zero_grad(set_to_none=True)
        result = student(**batch, output_hidden_states=True)
        kd = F.kl_div(F.log_softmax(result.logits.float(), dim=-1),
                      F.softmax(target.logits.float(), dim=-1), reduction="batchmean")
        align = torch.stack([F.mse_loss(x.float(), y.float()) for x, y in
                              zip(result.hidden_states[1:], target.hidden_states[1:])]).mean()
        loss = kd + align
        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        event("step_complete", step=step+1, loss=round(losses[-1], 6),
              reserved_gpu_mib=round(torch.cuda.memory_reserved()/1048576))
        del result, kd, align, loss
    student.eval().cpu()
    del optimizer, teacher, target
    torch.cuda.empty_cache()
    state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    packed = sum(k.endswith("_packed") for k in state)
    assert packed, "No packed quantization tensors found"
    checkpoint = output/"checkpoint.pt"
    torch.save(state, checkpoint)
    student.config.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert len(reloaded)==len(state), "Checkpoint read-back failed"
    meta = {"result":"QAT_SMOKE_PASS", "model":a.model_id, "steps":a.steps,
            "eff_bit_target":a.eff_bit, "itq_iterations":a.itq_iterations,
            "quantized_layers":converted, "losses":losses,
            "packed_tensor_keys":packed, "checkpoint_bytes":checkpoint.stat().st_size,
            "note":"Short synthetic data QAT test, not production model or Strata format."}
    (output/"result.json").write_text(json.dumps(meta,indent=2))
    event("checkpoint_verified", path=str(checkpoint), bytes=checkpoint.stat().st_size,
          packed_tensors=packed, losses=losses)

if __name__=="__main__":
    main()

