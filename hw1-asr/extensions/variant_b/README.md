# Alternative CUDA task-source snapshot

These five CUDA task-source files are **alternative implementations retained for comparison**, not the default `hw1-asr/csrc/` task files. The original independent test notes report 33/33 clean test cases passing on an RTX 2080 Ti and a same-node PyTorch/custom profile, with negative full-CUDA end-to-end latency results.

To select an alternative implementation, review one task at a time, replace the official task path in a *separate working copy*, and rerun the official build/tests on the same hardware. Do not claim this archived variant has been tested within the currently assembled checkout. No executable symlinks are created here.
