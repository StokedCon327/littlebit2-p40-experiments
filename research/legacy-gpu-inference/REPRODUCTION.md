# Reproduction notes

This is a sanitized method record, not a certified clean-clone recipe. The research code used local experimental builds; precise upstream source pins and redistributable patches for every engine need separate validation. Do not install on a production GPU. Use an isolated environment, a GPU idle check, a loopback-only endpoint, timeouts, memory limits and a verified cleanup path. Never drop shared system caches or unload unrelated workloads for these tests without authorization.

## Checkpoints

- Qwen: Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated.Q4_K_M.gguf, 18556689600 bytes; SHA256 `78e46edeaa74d5557bcbb8a5da2d134fb448c4a5dcbee5ce12022484bde0705d`.
- DeepSeek: [Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF), revision `73fa239749dfc09e847ea5e1149626d064224d92`, five UD-IQ1_S shards, total 232979608448 bytes. All five checksums verified locally. Metadata architecture deepseek41, 40 layers, 384 experts/layer, 6 routed; size label 748B-A16B. Obtain and verify the original files yourself; no weights included.

## Hardware

Tesla P40 SM 6.1 nominal 24 GB VRAM, 32 GB RAM inference host, SSD ext4 (USB connection); Linux CUDA 12-compatible runtime libraries, private compatibility loader for differing host glibc versions. The compilation host had more RAM. Two-host RPC test used private SSH forwarding on gigabit Ethernet; no network addresses or keys are included.

## Build notes

InferSSD: CUDA enabled, CMAKE_CUDA_ARCHITECTURES=61, LLAMA_SSD_STORE=ON, GGML_CUDA_SSD_VRAM_TIER=ON for the successful DeepSeek runtime. RPC repair adjusted the operation-count guard 97 to 102, protocol patch 4.0.1 to 4.0.2 and client negotiation to require exact major/minor/patch. Rebuild matching binaries at both endpoints; this is a local compatibility patch, not a general interoperability guarantee.

ik_llama.cpp: add appropriate cstddef/std::size_t declarations and validate pre-SM80 BF16 fallback by bit-widening into FP32. No original upstream source or patch redistribution in this report; license and exact revision review pending.

## Benchmark settings

Qwen matched tests: `llama-bench -m MODEL -ngl 99 -t 8 -b 512 -ub 512 -fa 0 -p 512 -n 128 -r 3` (verify syntax on pinned executable). Longer comparison uses 4096/256 and combined 4096,256. Record executable version, exact command, output and sampled GPU peak.

Colibri DeepSeek: `SERVE=1 CTX=512 COLI_CUDA=1 COLI_CUDA_DENSE_GB=8 COLI_VULKAN=0 OMP_NUM_THREADS=8 COLI_NO_OMP_TUNE=1 V41_DSPARK=0 SEED=1 V41_STATS=1 COLI_VK_TIER_HOST_GB=12`. Set SNAP to the local imported checkpoint; use eight expert slots per layer as positional argument. Set COLI_USAGE to a separate copy of the identical routing history for A/B experiments. The initial history file is not published; exact warm-cache replication requires rebuilding that history. Inspect the framed protocol rather than assuming newline-only READY. Request payload is UTF-8 with byte length, not character count.

InferSSD single-GPU: device CUDA0, split mode none, 99 GPU layers, 512 context, 128 batch/microbatch, eight threads, flash attention off, reasoning off, prompt cache disabled, one slot, mmap, no-repack, no-warmup, defer-experts, SSD source JSON array containing five local shard paths, 12 GiB VRAM expert budget, 1024 MiB VRAM reserve, 4096 MiB host headroom. Linux cgroup cap 25 GiB RAM, swap limit zero. Do not pass the Darwin/Windows-only host-budget option on Linux. Use a local loopback server port and the raw completion endpoint, temperature 0, max 32 generated tokens, cache_prompt=false, ignore_eos=false.

Raw non-thinking prompt format: begin-of-sentence marker, User marker, task, Assistant marker, closing think marker. Verify exact tokenizer control strings for the pinned model. First test: "Write a Python function named square that returns the square of its argument. Output only Python code." Correct semantic answer is `def square(x): return x * x`. Persistent cases: see [BENCHMARKS.md](BENCHMARKS.md) and [persistent-cases.json](persistent-cases.json).

Report prompt and decode time separately; aggregate by summing tokens and times, not averaging per-request rates. Record actual responses and score correctness, including failures. Record cold/warm definition, cache history, host and GPU peak, startup time, cleanup and failure reasons. Repeat on an independent host before claiming general performance.
