# Negative and incomplete results

- **InferSSD distributed DeepSeek attempt 1:** rejected a host-budget setting supported only on Darwin and Windows. Linux retry removed that setting and retained a cgroup memory limit.
- **Attempt 2:** dense tensors split successfully across local and RPC GPU buffers, but context initialization failed because nonzero GPU expert-cache budget required an injected provider. The build had `GGML_CUDA_SSD_VRAM_TIER=OFF`. Building with it enabled subsequently passed single-GPU inference. The inspected SSD context code supports one GPU expert executor per context; no claim is made that the distributed failure is fixed.
- **Colibri persistent run:** loaded and warmed 3866 experts but benchmark script waited for plain READY followed by newline. Actual readiness was framed by control characters. No requests sent; deadline terminated cleanly. Repair and re-run pending at this reporting cutoff.
- **DeepSeek persistent InferSSD quality:** wrong Python alias answer 50 (expected 44), shortest path 7 (expected 9), closure [0,1,2] (expected [2,2,2]). Non-thinking prompt, heavy quantization, short test. Failures retained; no causal attribution yet.
- **Speculative decoding:** current checkpoint metadata inspection found no DSpark/MTP draft head. Compatible weights and actual accepted-draft throughput must be validated before a speed claim.
- **Pascal microbenchmark:** DP4A did not improve the tested dot-product time; numerical checks passed.
- **Maxwell:** no actual M40 test. Compilation is not proof of practical inference.
- **LittleBit-2 integration:** separate study; no demonstrated integration of a trained LittleBit-2 checkpoint with these inference engines.

All reported completed lab runs released their GPU allocations. No production routing changes were made. Failures and blocked attempts are not counted as successful generated responses.
