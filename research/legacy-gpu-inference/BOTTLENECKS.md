# Bottlenecks and practical interpretation

1. **Storage:** the existing USB-connected SSD delivered about 406 MB/s in a sequential direct-read check and about 380–390 MB/s in observed expert reads. Uncached routed experts incur large transfers; warm-cache optimization reduced reads significantly on a repeated prompt. Random access, queueing and I/O overlap matter more than advertised peak bandwidth.
2. **Host RAM:** a 32 GB host constrains the resident expert set. Increasing the configured host cache from 8.5 to 12 GiB improved matched-history request throughput but lengthened warm-up. A 64 GB host and larger NVMe are hypotheses for future testing, not measured gains.
3. **VRAM:** peak use approached the entire available P40 capacity in persistent inference (24411 MiB sampled). Additional VRAM may improve residency; it does not automatically improve latency or intelligence.
4. **Compute:** CPU-side expert work, GPU transfers, attention, and quantized kernel throughput remain after disk misses are reduced. A zero-disk-miss extrapolation is not a promise of speed.
5. **Interconnect:** two-host Qwen layer splitting works, but network RPC adds substantial overhead. Same-host dual GPU and faster interconnect performance remain untested for this checkpoint.
6. **Accuracy:** three of five unique tasks failed with the heavily quantized non-thinking checkpoint configuration. Quantization, prompting, implementation and model limitations are possible contributors; the experiment does not isolate the cause.

The next hardware decision should follow controlled experiments rather than assuming additional retired cards or faster storage will resolve the quality and latency gap.
