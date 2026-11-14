# Dwayne's independently validated CUDA task-source snapshot

These five CUDA file blobs were copied byte-for-byte from `MLS-Dwayne` commit `45215acf37713659536adcda491556024d5058ef`. They are **alternatives retained for provenance**, not the default `hw1-asr/csrc/` task files in this draft. Dwayne's project notes report 33/33 clean test cases passing on an RTX 2080 Ti and the same-node PyTorch/custom profile, with negative full-CUDA end-to-end latency results.

To select a different submitted implementation, review one task at a time, replace the official task path in a *separate working copy*, and rerun the official build/tests on the same hardware. Do not claim this archived variant has been tested within the newly assembled checkout. No executable symlinks are created here.
