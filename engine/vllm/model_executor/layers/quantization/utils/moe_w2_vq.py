# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vector-code decode kernel for the 2-bit w2 tier (tools/e2e/vq_pipe.py codes).

A weight group is 8 consecutive k of one row: a pattern index into a learned
table C [P, 8] (e2m1 units, stored as e4m3) plus sign bits (7 stored, the
8th makes the number of negatives even). P = 512: 9 + 7 = 16 bits per group,
the same 2 bits per weight as the scalar planes. Scales are super-block
scales: per row and 256 k one exponent T (half octaves), per 32 k a 2-bit
step j, block scale 2^((T - j) / 2).

Built on moe_w2_mma's mma.sync e4m3 m16n8k32 decode kernel, same descriptor
ABI ({a, as, b, bs, c, m_rows}, one pair per blockIdx.y; b = code plane,
bs = scale plane) plus the table as a second kernel argument. In one 32-k
block a lane holds, per row, the A bytes of k {4t..4t+3} (a0) and
{16+4t..16+4t+3} (a2); here the lane's 8 bytes are the group k {8t..8t+7}
instead, and the activations are loaded in the same order, so the mma sums
the same products. The group decodes as one 8-byte table read from shared
memory with the sign bits XORed in. Both scales fold into the f32
accumulator per 32-k block, so the half-octave steps are exact.

Layouts (pack_vq): code plane [N/16][K/64][32 lanes][4 x u16], lane
(g, t) = (lane / 4, lane % 4), u16 i = (k32 h = i / 2, row g + 8 (i % 2)),
code = idx << 7 | signs 0-6. Scale plane [N/16][K/256][48 B]: T + 128 of
rows 0-15 (16 B), then per g one u32 of 2-bit steps, bit 2 (2 c + r) for
k32 c of the super-block (0-7) and row g + 8 r.
"""

import ctypes

import torch

from vllm.model_executor.layers.quantization.utils import moe_w2_mma

_SRC = r"""
// sign bits s (8) -> e4m3 sign masks of bytes 0-3 (lo) / 4-7 (hi)
__device__ __forceinline__ u32 sgn4(u32 s) {
  const u32 m = ((s & 0xfu) * 0x01010101u) & 0x08040201u;
  return (m + 0x7f7f7f7fu) & 0x80808080u;
}

__device__ __forceinline__ float half_oct(int h) {
  // 2^(h / 2), h integer half octaves
  const float r = __uint_as_float((u32)((h >> 1) + 127) << 23);
  return (h & 1) ? r * 1.41421356237f : r;
}

template <int K, int NB, int WK, int NT, int P>
__device__ __forceinline__ void wvq(const u64 *descs, const u64 *tab) {
  __shared__ u64 C[P];
  for (int i = threadIdx.x; i < P; i += blockDim.x) C[i] = tab[i];
  __syncthreads();
  const u64 *d = descs + blockIdx.y * 6;
  const int m = (int)d[5];
  if (m <= 0) return;
  const u8 *A = (const u8 *)d[0];
  const float *AS = (const float *)d[1];
  const u8 *W = (const u8 *)d[2];
  const u8 *WS = (const u8 *)d[3];
  u16 *Cout = (u16 *)d[4];
  const int n_rows = gridDim.x * NB * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tid = lane & 3;
  const int nb0 = blockIdx.x * NB;
  constexpr int KB = K / 64, G = K / 32, BB = 256;
  bool act[NT], v0[NT], v1[NT];
  const u8 *ap[NT];
  const float *as0[NT], *as1[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    act[n] = gid + 8 * n < m;
    const int t0 = 8 * n + tid * 2, t1 = t0 + 1;
    v0[n] = t0 < m;
    v1[n] = t1 < m;
    ap[n] = A + (size_t)(act[n] ? gid + 8 * n : 0) * K + tid * 8;
    as0[n] = AS + (v0[n] ? t0 : 0) * G;
    as1[n] = AS + (v1[n] ? t1 : 0) * G;
  }
  float acc[NT][NB][4];
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int j = 0; j < NB; ++j)
      acc[n][j][0] = acc[n][j][1] = acc[n][j][2] = acc[n][j][3] = 0.f;
  const u8 *wp[NB];
  const u8 *sp[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) {
    wp[j] = W + (size_t)(nb0 + j) * KB * BB + lane * 8;
    sp[j] = WS + (size_t)(nb0 + j) * (K / 256) * 48;
  }
  for (int kb = warp; kb < KB; kb += WK) {
    uint2 w[NB];
    float r[NB][2][2];     // [k32 h][row g / g+8]
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      w[j] = __ldg((const uint2 *)(wp[j] + (size_t)kb * BB));
      const u8 *s = sp[j] + (kb >> 2) * 48;
      const int T0 = (int)__ldg(s + gid) - 128, T8 = (int)__ldg(s + gid + 8) - 128;
      const u32 js = __ldg((const u32 *)(s + 16) + gid);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = (kb & 3) * 2 + h;
        r[j][h][0] = half_oct(T0 - (int)((js >> (4 * c)) & 3u));
        r[j][h][1] = half_oct(T8 - (int)((js >> (4 * c + 2)) & 3u));
      }
    }
    u32 b[NT][4];
    float sa[NT][2][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const u8 *a = ap[n] + kb * 64;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint2 x = act[n] ? __ldg((const uint2 *)(a + 32 * h)) : make_uint2(0u, 0u);
        b[n][2 * h] = x.x;
        b[n][2 * h + 1] = x.y;
        sa[n][h][0] = v0[n] ? __ldg(as0[n] + kb * 2 + h) : 0.f;
        sa[n][h][1] = v1[n] ? __ldg(as1[n] + kb * 2 + h) : 0.f;
      }
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      // a[h][row t][lo/hi]
      u32 a[2][2][2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const u32 code = ((i < 2 ? w[j].x : w[j].y) >> (16 * (i & 1))) & 0xffffu;
        u32 sg = code & 0x7fu;
        sg |= (__popc(sg) & 1u) << 7;
        const u64 v = C[code >> 7];
        a[i >> 1][i & 1][0] = (u32)v ^ sgn4(sg);
        a[i >> 1][i & 1][1] = (u32)(v >> 32) ^ sgn4(sg >> 4);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float r0 = r[j][h][0], r8 = r[j][h][1];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          float p[4] = {0.f, 0.f, 0.f, 0.f};
          mma(p, a[h][0][0], a[h][1][0], a[h][0][1], a[h][1][1], b[n][2 * h],
              b[n][2 * h + 1]);
          acc[n][j][0] = fmaf(p[0], sa[n][h][0] * r0, acc[n][j][0]);
          acc[n][j][1] = fmaf(p[1], sa[n][h][1] * r0, acc[n][j][1]);
          acc[n][j][2] = fmaf(p[2], sa[n][h][0] * r8, acc[n][j][2]);
          acc[n][j][3] = fmaf(p[3], sa[n][h][1] * r8, acc[n][j][3]);
        }
      }
    }
  }

  if (WK > 1) {
    __shared__ float red[WK][NT * NB * 4][32];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          red[warp][(n * NB + j) * 4 + i][lane] = acc[n][j][i];
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int j = 0; j < NB; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float s = red[0][(n * NB + j) * 4 + i][lane];
#pragma unroll
          for (int w2 = 1; w2 < WK; ++w2) s += red[w2][(n * NB + j) * 4 + i][lane];
          acc[n][j][i] = s;
        }
  }
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int t0 = 8 * n + tid * 2, t1 = t0 + 1;
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int row = (nb0 + j) * 16 + gid;
      if (v0[n]) {
        Cout[(size_t)t0 * n_rows + row] = bf16(acc[n][j][0]);
        Cout[(size_t)t0 * n_rows + row + 8] = bf16(acc[n][j][2]);
      }
      if (v1[n]) {
        Cout[(size_t)t1 * n_rows + row] = bf16(acc[n][j][1]);
        Cout[(size_t)t1 * n_rows + row + 8] = bf16(acc[n][j][3]);
      }
    }
  }
}
"""

_ENTRY = r"""
extern "C" __global__ void __launch_bounds__({threads}, {minb})
{name}(const u64 *descs, const u64 *tab) {{ wvq<{K}, {NB}, {WK}, {NT}, {P}>(descs, tab); }}
"""

# tier -> NT (token tiles of 8)
_TIERS = {"vq": 1, "vqmc4": 2}


def make_launchers(ks, P=512, cfg=None, tiers=("vq", "vqmc4")) -> dict:
    """{(tier, K): fn(desc, table, n_rows, pairs, stream=None)}; table: the
    [P, 8] e4m3 pattern table as a uint8 [P, 8] (or float8) CUDA tensor."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    cfg = cfg or moe_w2_mma._CFG
    nb, wk = cfg[:2]
    minb = cfg[3] if len(cfg) > 3 else 1
    torch.zeros(1, device="cuda")
    cap = torch.cuda.get_device_capability()
    src = moe_w2_mma._SRC + _SRC
    names = {}
    for tier in tiers:
        for k in ks:
            if k % 256:
                continue
            name = f"moe_{tier}_p{P}_k{k}_nb{nb}_wk{wk}_b{minb}"
            names[(tier, k)] = name
            src += _ENTRY.format(threads=32 * wk, name=name, K=k, NB=nb, WK=wk,
                                 NT=_TIERS[tier], P=P, minb=minb)
    img = moe_w2_mma._compile(src, f"sm_{cap[0]}{cap[1]}")
    cu = moe_w2_cubit._driver()
    mod = ctypes.c_void_p()
    moe_w2_cubit._ck(cu.cuModuleLoadData(ctypes.byref(mod), img),
                     "cuModuleLoadData moe_w2_vq")
    moe_w2_mma._modules.append((mod, img))
    fns = {}
    for key, name in names.items():
        fn = ctypes.c_void_p()
        moe_w2_cubit._ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                                name.encode()),
                         f"cuModuleGetFunction {name}")
        fns[key] = _launcher(fn, key[1], nb, wk, key[0])
    return fns


def _launcher(fn, K, nb, wk, tier):
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    cu = moe_w2_cubit._driver()
    rows = 16 * nb

    def launch(desc, table, n_rows, pairs, stream=None):
        if n_rows % rows:
            raise ValueError(f"moe_w2_vq: n_rows={n_rows} not a multiple of {rows}")
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        elif not isinstance(stream, int):
            stream = stream.value or 0
        args = [ctypes.c_uint64(desc.data_ptr()), ctypes.c_uint64(table.data_ptr())]
        argv = (ctypes.c_void_p * 2)(*[ctypes.cast(ctypes.byref(x), ctypes.c_void_p)
                                       for x in args])
        moe_w2_cubit._ck(cu.cuLaunchKernel(fn, n_rows // rows, pairs, 1, 32 * wk, 1, 1,
                                           0, stream, argv, None), "launch moe_w2_vq")

    launch.tier, launch.K = tier, K
    return launch


def table_e4m3(C: torch.Tensor) -> torch.Tensor:
    """float [P, 8] (e2m1 units) -> uint8 [P, 8] e4m3 bytes."""
    return C.float().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def pack_vq(idx: torch.Tensor, sg: torch.Tensor, s: torch.Tensor,
            q: torch.Tensor | None = None):
    """One matrix's codes (VQExperts layout: idx [N, K/8] pattern index, sg
    [N, K/8] sign bits, s [N, K/32] UE8M0-style exponent byte, q [N, K/32]
    quarter-octave steps, even) -> (code plane uint8, scale plane uint8)."""
    N, K8 = idx.shape
    K = K8 * 8
    assert N % 16 == 0 and K % 256 == 0
    code = (idx.int() << 7) | (sg.int() & 0x7F)                      # [N, K/8]
    # [N/16, 16 rows, K/64, k32 h, tid] -> [N/16, K/64, g, tid, (h, row t)]
    c = code.view(N // 16, 2, 8, K // 64, 2, 4)                      # nb, t, g, kb, h, tid
    c = c.permute(0, 3, 2, 5, 4, 1).contiguous()                     # nb, kb, g, tid, h, t
    plane = c.to(torch.int32).to(torch.int16).view(torch.uint8).reshape(N // 16, K // 64, 256)
    H = 2 * (s.int() - 127)
    if q is not None:
        assert bool((q.int() % 2 == 0).all()), "super-block steps are half octaves"
        H = H + q.int() // 2
    Hs = H.view(N, K // 256, 8)
    T = Hs.max(-1).values                                            # [N, K/256]
    j = T[..., None] - Hs                                            # [N, K/256, 8]
    assert int(j.max()) <= 3, f"step range {int(j.max())} > 3 half octaves"
    assert int(T.min()) >= -128 and int(T.max()) < 128
    Tb = (T + 128).to(torch.uint8).view(N // 16, 16, K // 256).permute(0, 2, 1)   # nb, sb, row
    jj = j.view(N // 16, 2, 8, K // 256, 8)                          # nb, r, g, sb, c
    sh = (2 * (2 * torch.arange(8, device=j.device)[:, None]
               + torch.arange(2, device=j.device)[None, :])).view(1, 1, 1, 8, 2)  # c, r
    js = (jj.permute(0, 3, 2, 4, 1).long() << sh).sum((-1, -2))     # nb, sb, g
    js = js.to(torch.int64).to(torch.int32).view(torch.uint8).view(N // 16, K // 256, 32)
    sp = torch.cat((Tb.contiguous(), js), -1).contiguous()           # [N/16, K/256, 48]
    return plane.contiguous(), sp


def ref_weight(idx, sg, s, q, C):
    """bf16 [N, K] reference decode (vq_pipe.VQExperts.get_one_ref) with an
    e4m3-rounded table."""
    Ce = C.float().to(torch.float8_e4m3fn).float()
    bits = torch.tensor([1 << j for j in range(8)], dtype=torch.int32, device=idx.device)
    sgb = (sg.int()[..., None] & bits) != 0
    sgb[..., 7] = sgb[..., :7].sum(-1) % 2 == 1
    v = Ce[idx.long()] * (1 - 2 * sgb.float())
    N = v.shape[0]
    e = s.float() - 127 + (q.float() * 0.25 if q is not None else 0)
    return v.view(N, -1) * torch.exp2(e).repeat_interleave(32, 1)
