# Legacy GPU inference: beyond fitting the model

**Independent engineering study, November 24–25, 2026.** This is a companion workstream to the LittleBit-2 training study, not a LittleBit-2 inference result. No new LittleBit-2-trained checkpoint was used in these experiments.

## Research objective

Maximize useful local model intelligence, sustained inference speed, and reliability per dollar on inexpensive retired enterprise GPUs. Fitting a checkpoint is a feasibility milestone, not proof of practical performance or quality.

## Findings

- A single NVIDIA Tesla P40 (Pascal SM 6.1, nominal 24 GB VRAM), with 32 GB host RAM and a USB-connected SSD, generated correct Python code using a 232,979,608,448-byte (233 GB decimal) DeepSeek V4.1 Flash UD-IQ1_S checkpoint in both Colibri and InferSSD.
- Fully GPU-resident Qwen3-Coder 30B-A3B Q4_K_M reached 69.52948 generated tokens/s in the short InferSSD benchmark; conventional llama.cpp reached 62.8289 tokens/s on the matched test.
- Colibri repeated-prompt cache experiments reduced observed SSD reads from 17.01 GB to 0.84 GB; a controlled 8.5 vs 12 GiB host-cache comparison increased prompt-inclusive output throughput from 0.674 to 0.938 tokens/s. The older baseline had less developed expert routing history.
- Persistent InferSSD reached 3.808 tokens/s decode on a repeated request, but only 0.640 tokens/s aggregate decode over six requests and 3/6 correct responses (2/5 unique tasks). Prompt processing was additional. The three failed answers were retained.
- Two-host P40 layer splitting worked for Qwen but reduced decode speed substantially relative to local execution over the tested gigabit Ethernet path. DeepSeek distributed expert streaming was not demonstrated.
- BigMoeOnEdge reduced peak VRAM from 18,011 to 9,589 MiB in a storage-streaming experiment, with a throughput trade-off and different generation lengths.

**Conclusion:** a 233 GB checkpoint can execute on this hardware, but its current speed and accuracy do not justify making it the primary coding agent. Fully resident smaller models are presently the more practical option.

## Read the study

- [BENCHMARKS.md](BENCHMARKS.md): measured results and metric definitions.
- [ENGINE-COMPARISON.md](ENGINE-COMPARISON.md): engine-specific outcomes and compatibility.
- [BOTTLENECKS.md](BOTTLENECKS.md): memory, storage, and interconnect interpretation.
- [FAILED-EXPERIMENTS.md](FAILED-EXPERIMENTS.md): blocked runs and quality failures.
- [REPRODUCTION.md](REPRODUCTION.md): model identity, settings, and replication limits.
- [ROADMAP.md](ROADMAP.md): practical performance and quality targets.
- [benchmark-data.csv](benchmark-data.csv): sanitized measured values.

Raw household system logs, credentials, network topology, model weights and proprietary data are not distributed. These results are local measurements, not independently replicated vendor benchmarks. Source revision and clean-clone replication coverage are incomplete; no claim of a fully reproducible release is made.
