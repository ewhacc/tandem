# DeepGEMM `_C` for DeepSeek-V4.1 on SM120

`_C.cpython-312-x86_64-linux-gnu.so` replaces the one in the vLLM 0.31.0 wheel
(`vllm/third_party/deep_gemm/`). It is DeepGEMM from https://github.com/vllm-project/DeepGEMM at commit
`e1f418c2` with `deepgemm-e1f418c2-sm120.patch` applied (vLLM-Moet's SM120 fix: 128-row indexer KV pages and
per-group BLOCK_M for the SM120 paged MQA logits kernel), built with vLLM's `tools/build_deepgemm_C.py` for
CUDA 13 / Python 3.12. DeepGEMM is MIT licensed (LICENSE here).

To rebuild: clone DeepGEMM, `git checkout e1f418c2`, `git apply deepgemm-e1f418c2-sm120.patch`, then run vLLM's
`tools/build_deepgemm_C.py` against that tree (it needs the libdw development headers).
