# Engine comparison

| Engine | Local evidence | Remaining uncertainty |
|---|---|---|
| llama.cpp | Fully resident Qwen matched baseline | Oversized DeepSeek not demonstrated |
| InferSSD | Qwen benchmarks; DeepSeek SSD-backed single-GPU generation; persistent cache; Qwen network RPC | Quality, repeatability, warm unfamiliar workloads; no distributed GPU expert tier proof |
| Colibri | Qwen GPU expert cache; DeepSeek correct square and arithmetic tests; host-cache A/B | Persistent harness repair; held-out accuracy; DSpark weights |
| BigMoeOnEdge | Qwen GPU-resident and SSD-streamed modes | Exact DeepSeek architecture and multi-host GPU support not demonstrated |
| ik_llama.cpp | Pascal build repaired; Qwen matched benchmark | Additional architectures and quality |
| ParaFlow | Qwen local RPC loopback | Exact DeepSeek V4.1 architecture not demonstrated |
| Strata | Existing working Qwen serving; same-host multi-GPU interface | Exact large-checkpoint runtime not demonstrated; no tested multi-host support |
| M40 / Maxwell | Strata SM 5.2 compilation succeeded | No M40 hardware run; do not infer speed or compatibility from compilation alone |

The broader survey considered MoE-Infinity, FastLLM, DwarfStar, NeutronStar, AirLLM, KTransformers and Prima.cpp. No positive measured performance claim is made here for those unvalidated candidates.

Pascal work included a local ik_llama.cpp size_t declaration repair and pre-SM80 BF16-to-FP32 fallback, a Colibri CUDA compatibility build, and InferSSD RPC protocol and CUDA expert-tier configuration changes. The Colibri DP4A microbenchmark was correct but did not outperform the scalar comparator (0.18743 vs 0.18336 ms; 32,768 dots); microbenchmarks are not model throughput.

No LittleBit-2 compression is credited for any of these measurements. No claim of a faster model from a modified GPU firmware is made.
