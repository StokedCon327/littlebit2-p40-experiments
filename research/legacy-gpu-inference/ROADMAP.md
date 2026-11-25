# Research roadmap: intelligence at usable speed

The objective is not simply to fit ever-larger checkpoints. It is high-quality local coding and reasoning with affordable hardware, reproducible measurements and reliable operation.

| Workload | Proposed minimum decode target | Preferred |
|---|---:|---:|
| Background research | 5 t/s | 15+ t/s |
| Complex reasoning | 10 t/s | 25+ t/s |
| Interactive chat | 20 t/s | 40+ t/s |
| Autonomous coding | 30 t/s | 60+ t/s |

Targets are design goals, not measured predictions. At 3 t/s, 2000 output tokens take more than 11 minutes before prompt processing or tools. Current DeepSeek results are not fit for a primary coding agent.

Research priorities:
1. Repair persistent Colibri readiness detection; run identical cold/warm conditions, multiple repetitions and unfamiliar tasks; compare non-thinking and thinking prompts fairly.
2. Expand accuracy evaluation: executable unit tests in a sandbox, repository repair, tool-call correctness, reasoning and structured output. Use held-out tests, avoid contamination. Compare smaller Qwen-class models on the same tasks. Require meaningful quality improvement before accepting additional latency.
3. Optimize expert residency and storage reads; characterize 32 vs 64 GB RAM and USB SSD vs NVMe when hardware is available. Measure effective workload I/O rather than advertised sequential speed.
4. Investigate compatible speculative decoding, verify draft acceptance and output quality, measure full generation time. Do not claim DSpark speedups without its required weights and tests.
5. Investigate better quantization and expert compression, including LittleBit-2 as a separate research track with proper training and held-out evaluation. Avoid extrapolation from two-step smoke QAT.
6. Evaluate Pascal-specific kernels and same-host dual GPUs only with controlled tests. Maxwell hardware remains unvalidated; do not buy additional cards solely from a compile result.

Report quality, total latency, sustained tokens/s, context, memory, power when available, reproducibility and cost. Keep experiments isolated from deployed services. Publish negative results as well as positive findings.
