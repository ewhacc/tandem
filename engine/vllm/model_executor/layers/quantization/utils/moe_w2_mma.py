# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native mma.sync decode kernel for the 2-bit w2 tier (SM89 and newer).

The SM120 path runs hand-written SASS (QMMA.SF block-scaled MMA). Other
GPUs have FP8 tensor cores but no block-scaled MMA, and the Triton
emulation (moe_w2_triton) is latency/ALU bound at decode: Triton's tile
layouts do not match the fragment-major 2-bit planes, so most of its
instructions are address arithmetic and layout conversion. This kernel
reads the planes in their native order instead:

  * weights are the mma A operand (16 rows = one n-block); each lane's
    8 plane bytes of a 64-k block ARE its A fragments for two k32 groups
    (pack_fragment_major was built for this), decoded 4 codes per PRMT
    through the e4m3 LUT {-4, -1, 1, 4}
  * tokens are the mma B operand (n8: m <= 8 per pair; decode uses m <= 4)
  * mma.sync.m16n8k32 e4m3 x e4m3 -> f32 per k32 group, then both scales
    are folded into an f32 accumulator (exact for UE8M0 scales)
  * split-K across the warps of a CTA, reduced through shared memory in a
    fixed order: deterministic, so the replay contract holds

Same 6-field descriptor ABI and grid shape family as the cubins
({a, as, b, bs, c, m_rows}; one pair per blockIdx.y), compiled at boot by
NVRTC for the device's own sm_XY (no prebuilt artifacts per arch).

Fragment decode and the A/B role assignment follow the Ada decode
prototype in vllm-moet-sm89-w2 (kernels/cuda/moe_w2_sm89_decode.cu,
Apache-2.0); generalized to any K, multiple n-blocks per CTA, split-K
warps and loads kept in flight.
"""

import ctypes
import glob
import os

import torch

_SRC = r"""
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

__device__ __forceinline__ u32 dec4(u32 b) {
  // 4 2-bit codes (low byte) -> 4 e4m3 bytes: 0 -> -4, 1 -> -1, 2 -> 1, 3 -> 4
  u32 sel = (b & 0x3u) | ((b & 0xcu) << 2) | ((b & 0x30u) << 4) |
            ((b & 0xc0u) << 6);
  u32 d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(0x4838b8c8u), "r"(0u),
      "r"(sel));
  return d;
}

// two code bytes (bits 0-7, 8-15) -> e4m3 words for byte 0 (*d0), byte 1 (*d1)
__device__ __forceinline__ void dec8(u32 x, u32 &d0, u32 &d1) {
  u32 t = (x | (x << 8)) & 0x00ff00ffu;
  t = (t | (t << 4)) & 0x0f0f0f0fu;
  t = (t | (t << 2)) & 0x33333333u;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d0) : "r"(0x4838b8c8u), "r"(0u),
      "r"(t));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d1) : "r"(0x4838b8c8u), "r"(0u),
      "r"(t >> 16));
}

__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 sel) {
  u32 d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}

// f16 path: two code bytes P (bits 0-7) and Q (bits 16-23) -> f16x2 words
// o = {P(c0,c1), P(c2,c3), Q(c0,c1), Q(c2,c3)}; {-4,-1,1,4} are exact in
// f16 (hi bytes C4 BC 3C 44, lo byte 0)
__device__ __forceinline__ void dec2_f16(u32 x, u32 (&o)[4]) {
  const u32 v = (x | (x << 6)) & 0x33333333u;      // nibbles c0 c2 c1 c3
  const u32 sa = ((v << 4) & 0xf0f0f0f0u) | 0x04040404u;
  const u32 sb = (v & 0xf0f0f0f0u) | 0x04040404u;
  o[0] = prmt(0x443cbcc4u, 0u, sa);
  o[1] = prmt(0x443cbcc4u, 0u, sb);
  o[2] = prmt(0x443cbcc4u, 0u, sa >> 16);
  o[3] = prmt(0x443cbcc4u, 0u, sb >> 16);
}

// 4 e4m3 bytes -> f16x2 (e0,e1), (e2,e3); exact
__device__ __forceinline__ void e4m3x4_f16(u32 x, u32 &b0, u32 &b1) {
#if __CUDA_ARCH__ >= 890
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2;"
      " cvt.rn.f16x2.e4m3x2 %0, l; cvt.rn.f16x2.e4m3x2 %1, h; }"
      : "=r"(b0), "=r"(b1) : "r"(x));
#else
  // e4m3 bits << 7 is an f16 with exponent bias 15 instead of 7: x 2^8
  const u32 y0 = prmt(x, 0u, 0x4140u), y1 = prmt(x, 0u, 0x4342u);
  const u32 f0 = ((y0 << 7) & 0x3f803f80u) | ((y0 << 8) & 0x80008000u);
  const u32 f1 = ((y1 << 7) & 0x3f803f80u) | ((y1 << 8) & 0x80008000u);
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(b0) : "r"(f0), "r"(0x5c005c00u));
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(b1) : "r"(f1), "r"(0x5c005c00u));
#endif
}

__device__ __forceinline__ void mma16(float (&c)[4], u32 a0, u32 a1, u32 a2,
                                      u32 a3, u32 b0, u32 b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ue8m0(u32 s) {
  // UE8M0 0 (2^-127) maps to 0.0: an all-but-zero block, no visible change
  return __uint_as_float(s << 23);
}

__device__ __forceinline__ void mma(float (&c)[4], u32 a0, u32 a1, u32 a2,
                                    u32 a3, u32 b0, u32 b1) {
  c[0] = c[1] = c[2] = c[3] = 0.f;
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ u16 bf16(float v) {
  u16 r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v));
  return r;
}

// e2m1 nibbles (16 bits: n0..n3, k4 order) -> 4 e4m3 bytes in k4 order:
// magnitude by an 8-byte PRMT LUT, sign bits by PRMT sign replication
__device__ __forceinline__ u32 dec4n_e4m3(u32 v) {
  const u32 mag = prmt(0x3c383000u, 0x4c484440u, v & 0x7777u);
  const u32 sgn = prmt(v, v << 4, 0x9d8cu);
  return mag | (sgn & 0x80808080u);
}

// e2m1 nibbles -> f16x2 words A = (n1, n3), B = (n0, n2) (k is permuted
// to match: activations use e4m3x4_f16 of bytes (e1, e3, e0, e2))
__device__ __forceinline__ void dec4n_f16(u32 v, u32 &a, u32 &b) {
  const u32 y = v << 4;
  a = prmt(0x3e3c3800u, 0x46444240u, v & 0x7070u) |
      (prmt(v, 0u, 0x9484u) & 0x80008000u);
  b = prmt(0x3e3c3800u, 0x46444240u, y & 0x7070u) |
      (prmt(y, 0u, 0x9484u) & 0x80008000u);
}

// One CTA: NB n-blocks (16 rows each) of one pair; WK warps split the K/64
// blocks (warp w takes kb = w, w + WK, ...). W4: e2m1 nibble plane (16 B
// per lane per 64-k block) instead of 2-bit codes (8 B). NT token tiles of
// 8 (m <= 8 * NT).
template <int K, int NB, int WK, int F16, int W4, int NT>
__device__ __forceinline__ void wdec(const u64 *descs) {
  const u64 *d = descs + blockIdx.y * 6;
  const int m = (int)d[5];
  if (m <= 0) return;
  const u8 *A = (const u8 *)d[0];
  const float *AS = (const float *)d[1];
  const u8 *W = (const u8 *)d[2];
  const u8 *WS = (const u8 *)d[3];
  u16 *C = (u16 *)d[4];
  const int n_rows = gridDim.x * NB * 16;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tid = lane & 3;
  const int nb0 = blockIdx.x * NB;
  constexpr int KB = K / 64, G = K / 32, BB = W4 ? 512 : 256;
  bool act[NT], v0[NT], v1[NT];
  const u8 *ap[NT];
  const float *as0[NT], *as1[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    act[n] = gid + 8 * n < m;
    const int t0 = 8 * n + tid * 2, t1 = t0 + 1;
    v0[n] = t0 < m;
    v1[n] = t1 < m;
    ap[n] = A + (size_t)(act[n] ? gid + 8 * n : 0) * K + tid * 4;
    as0[n] = AS + (v0[n] ? t0 : 0) * G;
    as1[n] = AS + (v1[n] ? t1 : 0) * G;
  }

  float acc[NT][NB][4];
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int j = 0; j < NB; ++j)
      acc[n][j][0] = acc[n][j][1] = acc[n][j][2] = acc[n][j][3] = 0.f;

  // per n-block base pointers (lane / row offsets folded in once)
  const u8 *wp[NB];
  const u8 *sp[NB];
#pragma unroll
  for (int j = 0; j < NB; ++j) {
    wp[j] = W + (size_t)(nb0 + j) * KB * BB + lane * (BB / 32);
    sp[j] = WS + (size_t)(nb0 + j) * G * 16 + gid;
  }
  for (int kb = warp; kb < KB; kb += WK) {
    // lane words: W2 2 x u32 (tile0 | tile1, bytes k32*2 + half);
    // W4 4 x u32 (tile*2 + k32, bytes half*2 + k4/2)
    u32 w[NB][W4 ? 4 : 2];
    u32 sc[NB][2];         // weight scale bytes: k32 x (row g | row g+8)
    u32 b[NT][4];          // activations, (k32, half) = q / 2, q % 2
    float sa[NT][2][2];    // [tile][k32][token t0/t1]
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      if (W4) {
        const uint4 x = __ldg((const uint4 *)(wp[j] + (size_t)kb * BB));
        w[j][0] = x.x; w[j][1] = x.y;
        w[j][W4 ? 2 : 0] = x.z; w[j][W4 ? 3 : 1] = x.w;
      } else {
        const uint2 x = __ldg((const uint2 *)(wp[j] + (size_t)kb * BB));
        w[j][0] = x.x; w[j][1] = x.y;
      }
      const u8 *s = sp[j] + kb * 32;
      sc[j][0] = (u32)__ldg(s) | ((u32)__ldg(s + 8) << 8);
      sc[j][1] = (u32)__ldg(s + 16) | ((u32)__ldg(s + 24) << 8);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const u8 *a = ap[n] + kb * 64;
#pragma unroll
      for (int q = 0; q < 4; ++q)
        b[n][q] = act[n] ? __ldg((const u32 *)(a + 16 * q)) : 0u;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        sa[n][h][0] = v0[n] ? __ldg(as0[n] + kb * 2 + h) : 0.f;
        sa[n][h][1] = v1[n] ? __ldg(as1[n] + kb * 2 + h) : 0.f;
      }
    }
    // f16 path: B fragments as f16x2, in the k order the A decode produces
    u32 bh[NT][4][2];
    if (F16) {
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int q = 0; q < 4; ++q)
          e4m3x4_f16(W4 ? prmt(b[n][q], 0u, 0x2031u) : b[n][q], bh[n][q][0],
                     bh[n][q][1]);
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      // A fragments: [k32 h][half][tile = row g / g+8][f16 word] (e4m3: word 0)
      u32 a[2][2][2][2];
      if (W4) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int f = 0; f < 2; ++f)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
              const u32 v = (w[j][t * 2 + h] >> (16 * f)) & 0xffffu;
              if (F16) dec4n_f16(v, a[h][f][t][0], a[h][f][t][1]);
              else a[h][f][t][0] = dec4n_e4m3(v);
            }
      } else if (F16) {
#pragma unroll
        for (int f = 0; f < 2; ++f)
#pragma unroll
          for (int t = 0; t < 2; ++t) {
            // bytes k32*2 + half: P / Q = k32 0 / 1 of half f
            u32 o[4];
            dec2_f16((w[j][t] >> (8 * f)) & 0x00ff00ffu, o);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              a[h][f][t][0] = o[2 * h];
              a[h][f][t][1] = o[2 * h + 1];
            }
          }
      } else {
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int t = 0; t < 2; ++t)
            dec8((w[j][t] >> (16 * h)) & 0xffffu, a[h][0][t][0], a[h][1][t][0]);
      }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float r0 = ue8m0(sc[j][h] & 0xffu);
        const float r8 = ue8m0(sc[j][h] >> 8);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          float p[4] = {0.f, 0.f, 0.f, 0.f};
          if (F16) {
            mma16(p, a[h][0][0][0], a[h][0][1][0], a[h][0][0][1],
                  a[h][0][1][1], bh[n][2 * h][0], bh[n][2 * h][1]);
            mma16(p, a[h][1][0][0], a[h][1][1][0], a[h][1][0][1],
                  a[h][1][1][1], bh[n][2 * h + 1][0], bh[n][2 * h + 1][1]);
          } else {
            mma(p, a[h][0][0][0], a[h][0][1][0], a[h][1][0][0],
                a[h][1][1][0], b[n][2 * h], b[n][2 * h + 1]);
          }
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
        C[(size_t)t0 * n_rows + row] = bf16(acc[n][j][0]);
        C[(size_t)t0 * n_rows + row + 8] = bf16(acc[n][j][2]);
      }
      if (v1[n]) {
        C[(size_t)t1 * n_rows + row] = bf16(acc[n][j][1]);
        C[(size_t)t1 * n_rows + row + 8] = bf16(acc[n][j][3]);
      }
    }
  }
}
"""

_ENTRY = r"""
extern "C" __global__ void __launch_bounds__({threads}, {minb})
{name}(const u64 *descs) {{ wdec<{K}, {NB}, {WK}, {F16}, {W4}, {NT}>(descs); }}
"""

# default decode config (n-blocks per CTA, split-K warps, k-blocks in flight)
# (n-blocks per CTA, split-K warps, k-blocks per step, min CTAs per SM);
# H200 sweep (K 2048/4096, 8-64 pairs): 4,4,1,4 best or within 3%
_CFG = tuple(int(x) for x in os.getenv("VLLM_MOE_W2_MMA_CFG",
                                       "4,4,1,4").split(","))

_nvrtc = None
_modules = []                # keep loaded modules alive


def _lib_nvrtc():
    global _nvrtc
    if _nvrtc is None:
        cands = []
        try:
            import nvidia
            for root in nvidia.__path__:
                cands += sorted(glob.glob(os.path.join(root, "*", "lib",
                                                       "libnvrtc.so*")))
        except ImportError:
            pass
        cands += ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
        err = None
        for c in cands:
            if "builtins" in c or ".alt." in c:
                continue
            try:
                _nvrtc = ctypes.CDLL(c)
                break
            except OSError as e:
                err = e
        if _nvrtc is None:
            raise RuntimeError(f"moe_w2_mma: libnvrtc not found ({err})")
    return _nvrtc


def _compile(src: str, arch: str) -> bytes:
    nv = _lib_nvrtc()
    prog = ctypes.c_void_p()
    r = nv.nvrtcCreateProgram(ctypes.byref(prog), src.encode(),
                              b"moe_w2_mma.cu", 0, None, None)
    if r:
        raise RuntimeError(f"nvrtcCreateProgram failed ({r})")
    opts = [f"--gpu-architecture={arch}".encode(), b"--std=c++17",
            b"-default-device"]
    argv = (ctypes.c_char_p * len(opts))(*opts)
    r = nv.nvrtcCompileProgram(prog, len(opts), argv)
    if r:
        n = ctypes.c_size_t()
        nv.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        nv.nvrtcGetProgramLog(prog, log)
        raise RuntimeError("moe_w2_mma: NVRTC compile failed:\n"
                           + log.value.decode(errors="replace"))
    n = ctypes.c_size_t()
    nv.nvrtcGetCUBINSize(prog, ctypes.byref(n))
    img = ctypes.create_string_buffer(n.value)
    nv.nvrtcGetCUBIN(prog, img)
    nv.nvrtcDestroyProgram(ctypes.byref(prog))
    return img.raw


def supported() -> bool:
    return torch.cuda.get_device_capability() >= (8, 0)


def dot_mode() -> str:
    """"e4m3": native mma.sync e4m3 (SM89, SM120). "f16": codes decoded to
    f16 + native f16 mma (SM80/86: no fp8 MMA; SM90/SM100: fp8 only via
    wgmma/tcgen05, mma.sync e4m3 is emulated with extra conversions)."""
    m = os.getenv("VLLM_MOE_W2_MMA_DOT", "auto").lower()
    if m in ("e4m3", "f16"):
        return m
    cap = torch.cuda.get_device_capability()
    return "e4m3" if cap == (8, 9) or cap[0] == 12 else "f16"


# tier -> (W4, NT): w2 decode (m <= 8), w2mc4 prefill (m <= 16), w4 FP4
# recovery tier (m <= 4: prefill dispatches it as M<=4 sub-entries)
_TIERS = {"w2": (0, 1), "w2mc4": (0, 2), "w4": (1, 1)}


def make_launchers(ks, cfg=None, dot=None, fallback=None,
                   tiers=("w2", "w2mc4", "w4")) -> dict:
    """{(tier, K): fn(desc, n_rows, pairs, stream=None)} for every tier and
    K, from one NVRTC build for this device."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    nb, wk = (cfg or _CFG)[:2]
    minb = (cfg or _CFG)[3] if len(cfg or _CFG) > 3 else 1
    dot = dot or dot_mode()
    torch.zeros(1, device="cuda")      # primary context current on this thread
    cap = torch.cuda.get_device_capability()
    # VLLM_MOE_W2_MMA_ARCH=sm_80: build an older same-major target (tests the
    # pre-SM89 software e4m3 conversion on an SM89 GPU)
    arch = os.getenv("VLLM_MOE_W2_MMA_ARCH") or f"sm_{cap[0]}{cap[1]}"
    src = _SRC
    names = {}
    for tier in tiers:
        w4, nt = _TIERS[tier]
        for k in ks:
            if k % 64:
                continue
            name = f"moe_{tier}_mma_k{k}_nb{nb}_wk{wk}_b{minb}_{dot}"
            names[(tier, k)] = name
            src += _ENTRY.format(threads=32 * wk, name=name, K=k, NB=nb,
                                 WK=wk, F16=int(dot == "f16"), W4=w4, NT=nt,
                                 minb=minb)
    img = _compile(src, arch)
    cu = moe_w2_cubit._driver()
    mod = ctypes.c_void_p()
    moe_w2_cubit._ck(cu.cuModuleLoadData(ctypes.byref(mod), img),
                     "cuModuleLoadData moe_w2_mma")
    _modules.append((mod, img))
    fns = {}
    for key, name in names.items():
        fn = ctypes.c_void_p()
        moe_w2_cubit._ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                                name.encode()),
                         f"cuModuleGetFunction {name}")
        fns[key] = _launcher(fn, key[1], nb, wk, (fallback or {}).get(key),
                             dot, key[0])
    return fns


def _launcher(fn, K, nb, wk, fallback=None, dot="f16", tier="w2"):
    """fallback: launcher for n_rows that are not a multiple of 16 * nb."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_cubit
    cu = moe_w2_cubit._driver()
    rows = 16 * nb

    def launch(desc, n_rows, pairs, stream=None):
        if n_rows % rows:
            if fallback is None:
                raise ValueError(f"moe_w2_mma: n_rows={n_rows} not a "
                                 f"multiple of {rows}")
            fallback(desc, n_rows, pairs, stream)
            return
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        elif not isinstance(stream, int):
            stream = stream.value or 0     # ctypes.c_void_p from moe_w2_cubit
        arg = ctypes.c_uint64(desc.data_ptr())
        argv = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(arg),
                                                 ctypes.c_void_p))
        moe_w2_cubit._ck(cu.cuLaunchKernel(fn, n_rows // rows, pairs, 1,
                                           32 * wk, 1, 1, 0, stream, argv,
                                           None), "launch moe_w2_mma")

    launch.tier, launch.K, launch.fp8, launch.native = tier, K, True, True
    launch.dot = dot
    return launch


def self_test(fns: dict, ks, n_rows: int = 256) -> dict:
    """Boot parity gate for the native launchers (each tier at its m range),
    same reference and threshold as moe_w2_triton.self_test."""
    from vllm.model_executor.layers.quantization.utils import moe_w2_triton
    ms = {"w2": (1, 4, 8), "w2mc4": (9, 16), "w4": (1, 4)}
    out = {}
    for (tier, k), fn in fns.items():
        if k not in ks:
            continue
        for m in ms[tier]:
            desc, bufs, ref = moe_w2_triton.op_case(
                "w4" if tier == "w4" else "w2", k, n_rows, m)
            fn(desc, n_rows, 1)
            torch.cuda.synchronize()
            w = moe_w2_triton.worst_rel(bufs[4], ref)
            out[(tier, k, m)] = w
            if not w < moe_w2_triton._GATE:
                raise RuntimeError(
                    f"moe_w2_mma parity FAILED: {tier} K={k} m={m} worst_rel="
                    f"{w:.3e} (gate {moe_w2_triton._GATE}), dot {fn.dot}")
    return out
