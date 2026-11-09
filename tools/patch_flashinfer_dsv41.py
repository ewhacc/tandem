#!/usr/bin/env python3
"""DeepSeek-V4.1 SM120 sparse-MLA support for an installed FlashInfer 0.7.0.post1.

vLLM 0.31 serves V4.1 with a 32-token SWA cache page (DeepseekV4SWACache) and
compressed caches paged at 64 (compress_ratio=2 layers) or 128 (ratio=1).
FlashInfer 0.7.0.post1 instantiates its SM120 DSV4 decode and prefill kernels
for 64-token pages only, so the first V4.1 decode fails with "SM120 sparse-MLA
has no decode kernel for this shape: ... page_block_size=32".

Same approach as vllm-moet-upstream tools/dsv41_sm120/patch_flashinfer.py
(written against FlashInfer 0.6.18, whose anchors no longer exist): the
kernels are templated on the page sizes and use them only for addressing, so
this adds instantiations and lets the dispatchers and the planner reach them.
0.7.0.post1 already rejects (instead of mis-addressing) a non-64 page, and its
top-k is a runtime argument, so no TOPK instantiations are needed.

  decode   DSV4 at page 32: num_heads 16/32/64 dedicated + runtime-H fallback
  prefill  single cache, page 32: num_heads 16/32/64, BF16 (topk<=256) / FP8
           dual cache, page 32 x extra page {64, 128}: full-tile + general
  planner  DSV4 accepts page 32 (prefill only for the instantiated head counts)

Head counts are those of V4.1's 64 attention heads at TP 4/2/1 (this box: TP
1-2); add more to DSV41_HEADS if needed (each costs JIT compile time).

Idempotent, anchored (asserts). Originals are kept as <file>.orig once.
  python3 tools/patch_flashinfer_dsv41.py [--flashinfer-dir DIR]
Then move ~/.cache/flashinfer/0.7.0.post1/120f/cached_ops/sparse_mla_sm120
aside so the module is rebuilt from the patched sources.
"""
import argparse
import ast
import pathlib
import shutil

DSV41_HEADS = (16, 32, 64)
MARK = "DSV4.1 (moe-platform)"


def patch(path: pathlib.Path, old: str, new: str) -> None:
    s = path.read_text()
    if new in s:
        return
    assert s.count(old) == 1, f"anchor x{s.count(old)} in {path.name}:\n{old[:300]}"
    orig = path.with_name(path.name + ".orig")
    if not orig.exists():
        shutil.copyfile(path, orig)
    path.write_text(s.replace(old, new))
    print(f"patched {path.name}: {old.strip().splitlines()[0][:70]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flashinfer-dir", default=None,
                    help="flashinfer package dir (default: the one this python imports)")
    d = ap.parse_args().flashinfer_dir
    if d is None:   # locate without importing (importing needs a GPU/CUDA setup)
        import importlib.util
        spec = importlib.util.find_spec("flashinfer")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("flashinfer not found: pass --flashinfer-dir")
        d = list(spec.submodule_search_locations)[0]
    fi = pathlib.Path(d)
    csrc = fi / "data/csrc"
    ver = (fi / "version.txt").read_text().strip() if (fi / "version.txt").exists() else "?"
    print(f"flashinfer {ver} at {fi}")

    # ---------------------------------------------------------------- decode
    dec = csrc / "sparse_mla_sm120_decode_dsv4.cu"
    patch(dec,
          "  if (page_block_size != 64) return false;\n  if (num_splits <= 0) return false;",
          f"  // {MARK}: the SWA cache page is 32; dispatched below.\n"
          "  if (page_block_size != 64 && !(mt == ModelType::DSV4 && page_block_size == 32))\n"
          "    return false;\n  if (num_splits <= 0) return false;")
    args = ("num_heads, topk, Q, KV_cache, indices, mid_out, mid_lse, topk_length, output, out_lse, "
            "attn_sink, extra_KV_cache, extra_indices, extra_topk_length, extra_topk, pbs_extra, "
            "stride_extra_kv_block, num_tokens, num_splits, chunks_per_block_override, sm_scale, "
            "stride_kv_block, stride_indices_token, stride_extra_indices_token, stride_out_lse, "
            "stream")
    lines = [f"  // {MARK}: DSV4 at SWA page 32 (must precede the page-64 dispatch,",
             "  // which does not look at page_block_size).",
             "  if (mt == ModelType::DSV4 && page_block_size == 32) {"]
    for h in DSV41_HEADS:
        lines.append(f"    if (num_heads == {h})")
        lines.append(f"      return launch_decode_dsv4_impl<ModelType::DSV4, {h}, 32>({args});")
    lines.append(f"    return launch_decode_dsv4_impl<ModelType::DSV4, 0, 32>({args});")
    lines.append("  }")
    patch(dec,
          "#define DSV4_DISPATCH(H) DECODE_DISPATCH(ModelType::DSV4, (H))\n",
          "\n".join(lines) + "\n#define DSV4_DISPATCH(H) DECODE_DISPATCH(ModelType::DSV4, (H))\n")

    # --------------------------------------------------------- prefill single
    pre = csrc / "sparse_mla_sm120_prefill.cu"
    single_old = """                                 size_t stride_kv_block, size_t stride_out_lse,
                                 const int* topk_length_ptr, cudaStream_t stream) {
  if (page_block_size != 64) return false;"""
    sw = "\n".join(
        f"      case {h}:\n        DSV41_MG({h}, {1 if h <= 16 else 2});\n        return true;"
        for h in DSV41_HEADS)
    single_new = f"""                                 size_t stride_kv_block, size_t stride_out_lse,
                                 const int* topk_length_ptr, cudaStream_t stream) {{
  // {MARK}: SWA cache page 32 (instantiated head counts only).
  if (page_block_size == 32) {{
#define DSV41_MG(NH, NHG)                                                                      \\
  if (topk <= 256)                                                                             \\
    launch_prefill_mg<ModelType::DSV4, ComputeMode::BF16, NH, 32, NHG>(                        \\
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, stride_kv_block, \\
        stride_out_lse, topk_length_ptr, stream);                                              \\
  else                                                                                         \\
    launch_prefill_mg<ModelType::DSV4, ComputeMode::FP8, NH, 32, NHG>(                         \\
        Q, KV, indices, attn_sink, output, out_lse, sm_scale, num_tokens, topk, stride_kv_block, \\
        stride_out_lse, topk_length_ptr, stream)
    switch (num_heads) {{
{sw}
      default:
        return false;
    }}
#undef DSV41_MG
  }}
  if (page_block_size != 64) return false;"""
    patch(pre, single_old, single_new)

    # ----------------------------------------------------------- prefill dual
    dual_old = """                               const int* topk_length_extra_ptr, cudaStream_t stream) {
  if (page_block_size != 64) return false;"""
    swd = lambda m: "\n".join(  # noqa: E731
        f"        case {h}:\n          {m}({h}, PBSX, {1 if h <= 16 else 2});\n          return true;"
        for h in DSV41_HEADS)
    dual_new = f"""                               const int* topk_length_extra_ptr, cudaStream_t stream) {{
  // {MARK}: SWA page 32 with compressed pages 64 (compress_ratio=2 layers)
  // or 128 (ratio=1 layers); instantiated head counts only.
  if (page_block_size == 32) {{
#define DSV41_FT(NH, PBSX, NHG)                                                                  \\
  launch_prefill_mg_dual_fulltile<ModelType::DSV4, NH, 32, PBSX, NHG>(                           \\
      Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens, topk, \\
      topk_extra, stride_kv_block, stride_kv_block_extra, stride_out_lse, stream)
#define DSV41_DUAL(NH, PBSX, NHG)                                                                \\
  launch_prefill_mg_dual<ModelType::DSV4, ComputeMode::BF16, NH, 32, PBSX, NHG>(                 \\
      Q, KV, indices, KV_extra, idx_extra, attn_sink, output, out_lse, sm_scale, num_tokens, topk, \\
      topk_extra, stride_kv_block, stride_kv_block_extra, stride_out_lse, topk_length_ptr,       \\
      topk_length_extra_ptr, stream)
#define DSV41_BY_PBSX(PBSX)                                                                      \\
  do {{                                                                                           \\
    if (full) {{                                                                                  \\
      switch (num_heads) {{                                                                       \\
{swd("DSV41_FT").replace(chr(10), " " * 8 + chr(92) + chr(10))}                                   \\
        default:                                                                                  \\
          return false;                                                                           \\
      }}                                                                                          \\
    }}                                                                                            \\
    switch (num_heads) {{                                                                         \\
{swd("DSV41_DUAL").replace(chr(10), " " * 8 + chr(92) + chr(10))}                                 \\
      default:                                                                                    \\
        return false;                                                                             \\
    }}                                                                                            \\
  }} while (0)
    const bool full = topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
                      topk_extra % BI == 0;
    if (extra_page_block_size == 64) DSV41_BY_PBSX(64);
    if (extra_page_block_size == 128) DSV41_BY_PBSX(128);
    return false;
#undef DSV41_BY_PBSX
#undef DSV41_DUAL
#undef DSV41_FT
  }}
  if (page_block_size != 64) return false;"""
    patch(pre, dual_old, dual_new)

    # --------------------------------------------------------------- planner
    plan = fi / "mla/_sparse_mla_sm120_plan.py"
    patch(plan, "_PAGE_BLOCK_SIZE = 64\n",
          f"_PAGE_BLOCK_SIZE = 64\n# {MARK}: DSV4 SWA page 32 (decode: any head count via the\n"
          "# runtime-H fallback; prefill: the instantiated head counts).\n"
          "_DSV41_PAGE_BLOCK_SIZE = 32\n"
          f"_DSV41_PREFILL_HEADS = frozenset({{{', '.join(map(str, DSV41_HEADS))}}})\n")
    patch(plan,
          "    if num_tokens > _DECODE_MAX_TOKENS or page_block_size != _PAGE_BLOCK_SIZE:\n"
          "        return False\n    if model_type == _MODEL_TYPE_DSV4:",
          "    if num_tokens > _DECODE_MAX_TOKENS:\n        return False\n"
          "    if page_block_size == _DSV41_PAGE_BLOCK_SIZE and model_type == _MODEL_TYPE_DSV4:\n"
          "        return (num_heads, topk) in _DECODE_DSV4_DISPATCH\n"
          "    if page_block_size != _PAGE_BLOCK_SIZE:\n        return False\n"
          "    if model_type == _MODEL_TYPE_DSV4:")
    patch(plan,
          "    if has_extra or page_block_size != _PAGE_BLOCK_SIZE or not _prefill_topk_ok(topk):\n"
          "        return False\n",
          "    if (page_block_size == _DSV41_PAGE_BLOCK_SIZE and model_type == _MODEL_TYPE_DSV4\n"
          "            and not has_extra):\n"
          "        return _prefill_topk_ok(topk) and num_heads in _DSV41_PREFILL_HEADS\n"
          "    if has_extra or page_block_size != _PAGE_BLOCK_SIZE or not _prefill_topk_ok(topk):\n"
          "        return False\n")
    patch(plan,
          "        and has_extra\n        and page_block_size == _PAGE_BLOCK_SIZE\n"
          "        and _prefill_topk_ok(topk)\n        and num_heads in _PREFILL_DSV4_HEADS\n    )",
          "        and has_extra\n        and _prefill_topk_ok(topk)\n"
          "        and (\n            (page_block_size == _PAGE_BLOCK_SIZE\n"
          "             and num_heads in _PREFILL_DSV4_HEADS)\n"
          "            or (page_block_size == _DSV41_PAGE_BLOCK_SIZE\n"
          "                and num_heads in _DSV41_PREFILL_HEADS)\n        )\n    )")
    ast.parse(plan.read_text())
    print("PATCH-FLASHINFER-DSV41-DONE")


if __name__ == "__main__":
    main()
