# Measurements (2026-11-24/25)

## Definitions

Decode = generated tokens / generation-only elapsed time. Prompt-inclusive output throughput = generated tokens / (prefill + decode). Model loading and cache warm-up are reported separately. MiB and GiB are binary units; GB and MB are decimal unless explicitly stated. Short fixed-generation benchmarks and natural-EOS request tests are different methods and must not be combined as a ranked speed leaderboard. Results rounded for readability.

## Qwen3-Coder 30B-A3B Q4_K_M (same 18.56 GB GGUF)

Matched llama-bench settings: 512 prompt tokens, 128 generation tokens, three repeats, 99 GPU layers, batch and microbatch 512, eight threads, flash attention off.

| Engine | Prompt t/s | Decode t/s | Peak MiB |
|---|---:|---:|---:|
| llama.cpp | 1184.35 | 62.8289 | 18115 |
| InferSSD | 1001.72 | 69.52948 | 18061 |

Longer matched test: 4096 prompt tokens, 256 generated tokens, three repeats.

| Engine | Prompt t/s | Decode t/s | Combined benchmark t/s | Peak MiB |
|---|---:|---:|---:|---:|
| llama.cpp | 837.379 | 64.2718 | 320.7583 | 18459 |
| InferSSD | 737.588 | 68.9313 | 310.0871 | 18601 |
| ik_llama.cpp | 785.287 | 66.7089 | 279.8354 | 18455 |

These are fully GPU-resident tests, not evidence of faster SSD streaming. Combined benchmark metric follows the benchmark program, not generated-only throughput.

Other bounded Qwen tests: BigMoe fully resident 63.192 t/s, peak 18011 MiB (256 tokens); storage-streamed 12.839 t/s, peak 9589 MiB (64 tokens), approximately 46.8% lower peak VRAM. Different generation lengths confound direct comparison. Colibri expert budgets 17/20 GB: 46.79/50.46 t/s. ParaFlow loopback RPC 59.70 t/s. InferSSD loopback RPC 58.40 t/s, two-host layer split 28.87 t/s (128 tokens, 26-token prompt, 99.88 s load, local peak 9659 MiB and remote peak 9217 MiB). The direct benchmark uses a different prompt; do not attribute the full gap solely to networking.

## DeepSeek Colibri request tests

Checkpoint size 232.98 GB, metadata size label 748B-A16B (not independently counted). Context 512, cap eight experts per layer, eight CPU threads, CUDA dense allowance 8 GB, temperature zero, DSpark off. Same square-function prompt, 14 generated tokens. Saved routing history differs between original and later tests; 8.5/12 GiB tests used identical initial history.

| Test | Host pool GiB | Output t/s incl. prefill | Request s | SSD MB | Misses | GPU expert route share | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8.5 | 0.213 | 65.591 | 17011.2 | 2237 | 53.1% | 23289 |
| Matched cache A | 8.5 | 0.674 | 20.760 | 3601.7 | 427 | 84.6% | 23289 |
| Matched cache B | 12 | 0.938 | 14.924 | 836.2 | 99 | 84.6% | 23297 |

Cache B: warm start 86.2 s, whole run 126.05 s; cache A: warm start 75.0 s, whole run 120.04 s. A to B request throughput improved about 39%; warm-up increased. The original-to-B SSD read reduction is 95.1%, not a cache-size-only causal estimate.

## DeepSeek InferSSD CUDA expert tier

Standalone: load 76.01 s, prefill 23 tokens in 35.713 s, generate 15 tokens in 24.985 s, decode 0.60035 t/s, prompt-inclusive 0.2471 t/s, peak 23347 MiB, correct square function.

Persistent session: no prompt-cache tokens reused; first square request, four different tasks, then repeated square. No independent repeat sets. Reasoning suppressed with the non-thinking prompt format. All six requests returned responses.

| Task | Expected | Actual | Generated | Request s | Decode t/s | Correct |
|---|---|---|---:|---:|---:|---|
| Square | square function | correct function | 15 | 93.431 | 0.2924 | yes |
| Python alias | 44 | 50 | 2 | 53.248 | 0.7966 | no |
| SQL count | 3,2,4 | 3,2,4 | 6 | 44.250 | 1.1774 | yes |
| Shortest path | 9 | 7 | 2 | 46.318 | 3.2657 | no |
| Python closure | [2,2,2] | [0,1,2] | 10 | 51.402 | 0.6797 | no |
| Square repeat | square function | correct function | 15 | 26.340 | 3.8082 | yes |

50 generated tokens, 314.989 s total request time, aggregate prompt-inclusive throughput 0.159 t/s, aggregate decode 0.640 t/s (total tokens divided by sum of decode times). Accuracy 3/6 requests, 2/5 distinct tasks. Peak 24411 MiB, load 96.00 s, whole run 416.91 s, GPU memory released at exit. The repeat improved decode 13x and request time 3.55x; this is not sustained unfamiliar-prompt performance. Tiny outputs make individual decode rates noisy.

The matching persistent Colibri test did not submit any requests because the harness did not recognize a control-character-framed READY message. No matched persistent engine ranking is available.
