## LiteSpec Optimization 10/14 (Test on H800 GPUs)

1. Reuse RNN/head logits during the DLite transition stage

   - KL and supervised loss share a single student head computation, reducing the number of projections from 2 to 1.
   - Gradient equivalence has been verified.

2. Reuse draft logits for DFlash/DFlash2 logging

   - Metrics use the objective's logits directly, eliminating the extra vocabulary projection on logging steps.
   - Forward passes with metrics require 1 projection per chunk instead of 2.

3. Skip logging callbacks during checkpoint recomputation

   - The backward pass no longer repeats projections and metrics used for logging, including teacher projections needed only for logging.
   - Recomputation required by the training objective is preserved.

4. Reuse distribution diagnostics in DSpark

   - Acceptance probabilities, L1 distances, and teacher/student top-1 probabilities reuse existing results, reducing duplicate softmax operations and reductions.

5. Copy FP32 master parameters directly into BF16 parameters

   - Parameter updates and checkpoint restoration use `copy_(master)`, eliminating the intermediate full-size BF16 tensor previously created before copying.
   - This reduces temporary allocations and memory transfers.

6. Compile and fuse operations after the L1 softmax

   - Preserve native softmax while fusing subtraction, absolute value, weighting, and reduction; CE now remains native.
   - Latest observations from stable training:
     - Loss phase: 24.68 → 22.86 ms, a 7.4% reduction in time.
     - Backward pass: 71.71 → 70.09 ms, a 2.25% reduction in time.
     - Peak allocated memory: 28.70 → 26.67 GiB, a reduction of 2.03 GiB.

7. Compile KL and optimize the teacher computation boundary

   - Compute and detach teacher probabilities in advance, preventing AOTAutograd from recomputing teacher normalization during the backward pass.
   - Historical operator microbenchmarks with exclusive GPU access, shape `[512, 151936]`:
     - FP32: 2.661 → 1.643 ms, a 38.3% reduction in time.
     - BF16: 3.297 → 1.705 ms, a 48.3% reduction in time.

8. Compile BCE with a fallback for small inputs

   - Eligible BCE operations with FP32 targets can be compiled; CPU and small inputs retain native execution in `auto` mode.
   - Shared loss functions use an input-size threshold to avoid unnecessary compilation for small workloads.
