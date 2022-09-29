# General Matrix Multiply

This is a quick-and-dirty benchmark of some methods for matrix multiplication. I wrote it to play around with vector instructions. 

I only pushed it to github because I don't understand some of the results. The `mult_transpose` function is significantly slower when compiled with fused multiply-add instructions enabled. Either with `-mfma` or `-march=native` on my cpu.

This is by no means a good benchmark. But the slow down is too significant and consistent between runs to be an artefact.

The CPU is a AMD Ryzen 7 3700X.

With just `-mavx`:
```
Running build/gemm
Run on (16 X 4426.17 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 16384 KiB (x2)
Load Average: 0.60, 0.31, 0.22
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-------------------------------------------------------
Benchmark             Time             CPU   Iterations
-------------------------------------------------------
bm_naive     2432083111 ns   2426607064 ns            1
bm_transpose  725917844 ns    724115971 ns            1
bm_avx        307623767 ns    306742756 ns            2
```

With `-mfma`:
```
Running build/gemm
Run on (16 X 4426.17 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 512 KiB (x8)
  L3 Unified 16384 KiB (x2)
Load Average: 0.32, 0.28, 0.21
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-------------------------------------------------------
Benchmark             Time             CPU   Iterations
-------------------------------------------------------
bm_naive     2504957466 ns   2499343469 ns            1
bm_transpose 1209504004 ns   1206285226 ns            1
bm_avx        314533624 ns    313548474 ns            2
```
