# Local SM120 cubin set (vLLM-Moet moe_w2)

The shipped upstream set (`vllm-moet-upstream/kernels/cubins-sm120`) plus the
DeepSeek-V4.1-Flash contractions, generated here on 2026-10-14.

| K | used by | NWARP | variants |
|---|---|---|---|
| 5120 | V4.1 gate-up (hidden 5120), any TP | 8 | w2, w2 mc4, w2 mc4afrag, w4 |
| 2304 | V4.1 down, TP1 (intermediate 2304) | 6 | same |
| 1152 | V4.1 down, TP2 | 3 | same |

## How they were built
- **SASS:** `vllm-moet-upstream/kernels/gen/gen_moe_w2.py <out> <K>`; `MOEW2_MC=4` gives mc4 and adding `MOEW2_AFRAG=1` gives mc4afrag. `gen_moe_w4.py <out> <K>` gives w4. Sources are in `src/`.
- **Assembler:** cubit `c139df8` (the "Initial public release", 2026-06-25), built in `cubit-0625`. Upstream's pin `5912400` is not in the public history. `c139df8` reproduces the shipped `moe_w2_mm_k4096_a32.cubin` byte-for-byte; cubit HEAD fails to encode the QMMA.SF `.ROW` form, so it must not be used for these kernels.
- **Command:** `cubit asm <sass> -o <cubin> --kernel moe_w2_mm|moe_w4_mm --mercury-stub vllm-moet-upstream/kernels/sass/qmma_e4m3.merc.stub -t cubit-0625/tables/sm120.json`

## Validation
Run on an RTX PRO 6000 (GPU 1) with upstream's op checkers at V4.1 shapes (E=5, M=4):
- `moe_w2_check.py` / `moe_w4_check.py` give worst_rel 2.6e-3 to 3.5e-3 (distinct=1, i.e. deterministic);
- `moe_w2_afrag_check.py` shows AFRAG bit-exact against mc4 (0/15 mismatched pairs);
- the shapes were K=5120 with N=4608 and N=2304, and K=2304/1152 with N=5120.

## Using them
Serving needs `VLLM_MOE_W2_CUBIT_DIR=<install prefix>/share/tandem/cubins` and `VLLM_MOE_W2_KS=7168,6144,5120,4096,2304,2048,1152,1024,512`. The loader only opens the listed K values.
