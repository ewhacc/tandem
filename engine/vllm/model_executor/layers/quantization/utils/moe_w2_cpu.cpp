// moe_w2 CPU expert engine (E2): one routed expert from the
// fragment-major 2-bit host row [codes13 | sc13 | codes2 | sc2], the same
// bytes the GPU reads (moe_w2_planes.pack_fragment_major / pack_scales).
//
// Plane block (nb, kb) = 256 bytes = 16 rows x 64 k; byte
//   bi = (((g*4 + t)*2 + tile)*2 + k32)*2 + half,  row = tile*8 + g,
//   k  = kb*64 + k32*32 + half*16 + t*4 + k4  (k4 = 2-bit field of the byte).
// A byte is one VNNI lane (4 consecutive k): expand -> u8 (w + 4), dot with
// int8 activations pre-permuted to the same lane order, subtract 4*sum(a).
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
namespace {

#ifndef PF_KB
#define PF_KB 4
#endif
// the multi-pass kernels read 32 or 64 of every 256 plane bytes per pass, a
// stride the hardware prefetchers follow poorly; the first pass pulls whole
// kb chunks PF_KB ahead (past the block end it warms the next block)
inline void prefetch_kb(const uint8_t* p) {
  for (int l = 0; l < 4; l++) _mm_prefetch((const char*)p + 64 * l, _MM_HINT_T0);
}

#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__)
#define W2E_ISA "avx512"
struct Consts {
  __m512i perm[4];      // source bytes 16q + 2w + {0,1} -> qword w bytes 0,1
  __m512i idx[4][4];    // lane -> scale index k32*16 + row
  __m512i idxq[2];      // activation byte for (p, lane, k4)
  __m512i msctrl, lut, ones;
  __mmask64 keep01;
  Consts() {
    alignas(64) uint8_t b[64];
    for (int q = 0; q < 4; q++) {
      for (int i = 0; i < 64; i++) b[i] = (uint8_t)(16 * q + 2 * (i >> 3) + (i & 1));
      perm[q] = _mm512_loadu_si512(b);
    }
    alignas(64) int32_t ix[16];
    for (int j = 0; j < 4; j++)
      for (int q = 0; q < 4; q++) {
        for (int l = 0; l < 16; l++)
          ix[l] = ((l >> 1) & 1) * 16 + ((l >> 2) & 1) * 8 + 2 * j + (q >> 1);
        idx[j][q] = _mm512_loadu_si512(ix);
      }
    for (int p = 0; p < 2; p++) {
      for (int l = 0; l < 16; l++)
        for (int k4 = 0; k4 < 4; k4++)
          b[l * 4 + k4] = (uint8_t)(((l >> 1) & 1) * 32 + (l & 1) * 16 +
                                    (p * 2 + (l >> 3)) * 4 + k4);
      idxq[p] = _mm512_loadu_si512(b);
    }
    // byte i of qword w: field at bit 2*(i&3) + 8*(i>>2) -> code k4 = i&3 of
    // source byte 2w + (i>>2)
    for (int i = 0; i < 64; i++) b[i] = (uint8_t)(2 * (i & 3) + 8 * ((i & 7) >> 2));
    msctrl = _mm512_loadu_si512(b);
    static const uint8_t val[4] = {0, 3, 5, 8};   // code -> w + 4
    for (int i = 0; i < 64; i++) b[i] = val[i & 3];
    lut = _mm512_loadu_si512(b);
    ones = _mm512_set1_epi8(1);
    uint64_t m = 0;
    for (int i = 0; i < 64; i++) if ((i & 7) < 2) m |= 1ull << i;
    keep01 = (__mmask64)m;
  }
};
const Consts& C() { static Consts c; return c; }

struct ActQ {                 // one activation vector, int8 per 32-group
  int8_t* aq = nullptr;       // [KB][2][64] lane order
  int32_t* ac = nullptr;      // [KB][2][16] 4 * lane sums
  float* asv = nullptr;       // [KB][16] group scale per lane
  int cap = 0;
  void reserve(int K) {
    if (cap >= K) return;
    free(aq); free(ac); free(asv);
    aq = (int8_t*)aligned_alloc(64, (size_t)K * 2);
    ac = (int32_t*)aligned_alloc(64, (size_t)K / 2 * 4);
    asv = (float*)aligned_alloc(64, (size_t)K / 4 * 4);
    cap = K;
  }
};

inline __m512i quant16(__m512 v, __m512 inv) {
  __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v, inv));
  q = _mm512_max_epi32(_mm512_min_epi32(q, _mm512_set1_epi32(127)),
                       _mm512_set1_epi32(-127));
  return q;
}

// 64 values of chunk kb -> lane-order int8, lane sums, lane scales
inline void prep_kb(const float* x, ActQ& o, int kb) {
  const Consts& c = C();
  __m512 v[4];
  for (int i = 0; i < 4; i++) v[i] = _mm512_loadu_ps(x + 16 * i);
  float m0 = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v[0]), _mm512_abs_ps(v[1])));
  float m1 = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v[2]), _mm512_abs_ps(v[3])));
  float s0 = m0 > 0.f ? m0 / 127.f : 1.f, s1 = m1 > 0.f ? m1 / 127.f : 1.f;
  __m512 i0 = _mm512_set1_ps(1.f / s0), i1 = _mm512_set1_ps(1.f / s1);
  __m128i b0 = _mm512_cvtepi32_epi8(quant16(v[0], i0));
  __m128i b1 = _mm512_cvtepi32_epi8(quant16(v[1], i0));
  __m128i b2 = _mm512_cvtepi32_epi8(quant16(v[2], i1));
  __m128i b3 = _mm512_cvtepi32_epi8(quant16(v[3], i1));
  __m512i q = _mm512_inserti32x4(_mm512_castsi128_si512(b0), b1, 1);
  q = _mm512_inserti32x4(q, b2, 2);
  q = _mm512_inserti32x4(q, b3, 3);
  for (int p = 0; p < 2; p++) {
    __m512i a = _mm512_permutexvar_epi8(c.idxq[p], q);
    _mm512_store_si512(o.aq + (size_t)kb * 128 + 64 * p, a);
    __m512i sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c.ones, a);
    _mm512_store_si512(o.ac + (size_t)kb * 32 + 16 * p, _mm512_slli_epi32(sum, 2));
  }
  _mm512_store_ps(o.asv + (size_t)kb * 16,
                  _mm512_mask_blend_ps((__mmask16)0xCCCC, _mm512_set1_ps(s0),
                                       _mm512_set1_ps(s1)));
}

inline __m512 exp16(__m512 x) {
  x = _mm512_max_ps(_mm512_min_ps(x, _mm512_set1_ps(88.f)), _mm512_set1_ps(-88.f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693145752f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(1.42860677e-06f), r);
  __m512 p = _mm512_set1_ps(1.f / 720);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 120));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 24));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 6));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));
  return _mm512_scalef_ps(p, n);
}

// g = silu(min(gate, lim)) * clamp(up, +-lim) for 64 values, then prep_kb
// (vq: the natural-order layout of prep_kb_vq)
inline void prep_kb_vq(const float* x, ActQ& o, int kb);
inline void act_kb(const float* gate, const float* up, float lim, ActQ& o, int kb,
                   bool vq = false) {
  alignas(64) float g[64];
  __m512 L = _mm512_set1_ps(lim), NL = _mm512_set1_ps(-lim), one = _mm512_set1_ps(1.f);
  for (int i = 0; i < 4; i++) {
    __m512 gt = _mm512_min_ps(_mm512_loadu_ps(gate + 16 * i), L);
    __m512 u = _mm512_max_ps(_mm512_min_ps(_mm512_loadu_ps(up + 16 * i), L), NL);
    __m512 e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), gt));
    _mm512_store_ps(g + 16 * i, _mm512_mul_ps(_mm512_div_ps(gt, _mm512_add_ps(one, e)), u));
  }
  if (vq) prep_kb_vq(g, o, kb);
  else prep_kb(g, o, kb);
}

// rows [nb*16, nb*16 + 16) of y = W x
inline void gemv_nb(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                    const ActQ& a, float* y) {
  const Consts& c = C();
  const int KB = K / 64, KS = K / 32;
  __m512 acc[4][4];
  for (int j = 0; j < 4; j++)
    for (int q = 0; q < 4; q++) acc[j][q] = _mm512_setzero_ps();
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  const uint8_t* sb = sc + (size_t)nb * KS * 16;
  for (int kb = 0; kb < KB; kb++) {
    __m512i s32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sb + kb * 32)));
    __m512 F0 = _mm512_castsi512_ps(_mm512_slli_epi32(s32, 23));
    s32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sb + kb * 32 + 16)));
    __m512 F1 = _mm512_castsi512_ps(_mm512_slli_epi32(s32, 23));
    __m512 asv = _mm512_load_ps(a.asv + (size_t)kb * 16);
    __m512i A[2] = {_mm512_load_si512(a.aq + (size_t)kb * 128),
                    _mm512_load_si512(a.aq + (size_t)kb * 128 + 64)};
    __m512i AC[2] = {_mm512_load_si512(a.ac + (size_t)kb * 32),
                     _mm512_load_si512(a.ac + (size_t)kb * 32 + 16)};
    for (int j = 0; j < 4; j++) {
      __m512i src = _mm512_loadu_si512(pb + kb * 256 + 64 * j);
      for (int q = 0; q < 4; q++) {
        __m512i t = _mm512_maskz_permutexvar_epi8(c.keep01, c.perm[q], src);
        __m512i f = _mm512_multishift_epi64_epi8(c.msctrl, t);
        __m512i w = _mm512_permutexvar_epi8(f, c.lut);
        __m512i d = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, A[q & 1]);
        d = _mm512_sub_epi32(d, AC[q & 1]);
        __m512 s = _mm512_mul_ps(_mm512_permutex2var_ps(F0, c.idx[j][q], F1), asv);
        acc[j][q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), s, acc[j][q]);
      }
    }
  }
  for (int j = 0; j < 4; j++)
    for (int h = 0; h < 2; h++) {
      __m512 v = _mm512_add_ps(acc[j][2 * h], acc[j][2 * h + 1]);
      int g = 2 * j + h;
      y[g] = _mm512_mask_reduce_add_ps((__mmask16)0x0F0F, v);
      y[8 + g] = _mm512_mask_reduce_add_ps((__mmask16)0xF0F0, v);
    }
}

// rows [nb*16, nb*16 + 16) of y_t = W x_t for NT tokens of one expert: each
// 64-byte weight chunk is expanded once and dotted with every token
// (j-outer, so the nb block's later passes hit L1)
template <int NT>
inline void gemv_nb_multi(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                          const ActQ* const* a, float* const* y) {
  const Consts& c = C();
  const int KB = K / 64, KS = K / 32;
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  const uint8_t* sb = sc + (size_t)nb * KS * 16;
  for (int j = 0; j < 4; j++) {
    __m512 acc[NT][4];
    for (int t = 0; t < NT; t++)
      for (int q = 0; q < 4; q++) acc[t][q] = _mm512_setzero_ps();
    for (int kb = 0; kb < KB; kb++) {
      if (j == 0) prefetch_kb(pb + (size_t)(kb + PF_KB) * 256);
      __m512i s32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sb + kb * 32)));
      __m512 F0 = _mm512_castsi512_ps(_mm512_slli_epi32(s32, 23));
      s32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(sb + kb * 32 + 16)));
      __m512 F1 = _mm512_castsi512_ps(_mm512_slli_epi32(s32, 23));
      __m512i src = _mm512_loadu_si512(pb + kb * 256 + 64 * j);
      __m512i w[4];
      __m512 ws[4];
      for (int q = 0; q < 4; q++) {
        __m512i t = _mm512_maskz_permutexvar_epi8(c.keep01, c.perm[q], src);
        w[q] = _mm512_permutexvar_epi8(_mm512_multishift_epi64_epi8(c.msctrl, t), c.lut);
        ws[q] = _mm512_permutex2var_ps(F0, c.idx[j][q], F1);
      }
      for (int t = 0; t < NT; t++) {
        const ActQ& A = *a[t];
        __m512 asv = _mm512_load_ps(A.asv + (size_t)kb * 16);
        for (int q = 0; q < 4; q++) {
          int p = q & 1;
          __m512i d = _mm512_dpbusd_epi32(
              _mm512_setzero_si512(), w[q],
              _mm512_load_si512(A.aq + (size_t)kb * 128 + 64 * p));
          d = _mm512_sub_epi32(d, _mm512_load_si512(A.ac + (size_t)kb * 32 + 16 * p));
          acc[t][q] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d), _mm512_mul_ps(ws[q], asv),
                                      acc[t][q]);
        }
      }
    }
    for (int t = 0; t < NT; t++)
      for (int h = 0; h < 2; h++) {
        __m512 v = _mm512_add_ps(acc[t][2 * h], acc[t][2 * h + 1]);
        int g = 2 * j + h;
        y[t][g] = _mm512_mask_reduce_add_ps((__mmask16)0x0F0F, v);
        y[t][8 + g] = _mm512_mask_reduce_add_ps((__mmask16)0xF0F0, v);
      }
  }
}

// 64 bf16 -> f32
inline void bf16_64(const uint16_t* s, float* xf) {
  for (int i = 0; i < 4; i++) {
    __m512i v = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)(s + 16 * i)));
    _mm512_store_ps(xf + 16 * i, _mm512_castsi512_ps(_mm512_slli_epi32(v, 16)));
  }
}

// ---- vector codes (VLLM_MOE_W2_VQ, moe_w2_vq.pack_vq). A u16 code =
// pattern << 7 | signs 0-6 of 8 consecutive k of one row (the 8th sign
// makes the negatives even). Block (nb, kb) = 256 bytes: lane (g, t) at
// (g*4 + t)*8, u16 i = (k32 h = i/2, row g + 8 (i%2)), group k = kb*64 +
// 32h + 8t. The pattern table here is u8 [P][8] = int8 (w * 16) ^ 0x80;
// one gather of 8 groups (one row, (h, t) in order) gives 64 bytes in
// natural k order, the signs negate them mod 256 (-(w+128) = -w+128), and
// dpbusd with int8 activations in natural order minus 128 * lane sums is
// the int dot. Scale plane [N/16][K/256][48]: T + 128 of rows 0-15, then
// per g a u32 of 2-bit steps j, bit 2 (2c + r) for k32 c and row g + 8r;
// block scale 2^((T - j) / 2).
#define W2E_VQ 1
struct VQConsts {
  __m512i perm;         // 32 codes [gg][t][h][r] -> [gg][r][h][t]
  __m512i rowmod8, rhi; // lane row -> row & 7, 2 * (row >> 3)
  __m512i rsel[16];     // R[kb] (h 0 | h 1) -> lanes 0-7 row, 8-15 16 + row
  VQConsts() {
    for (int row = 0; row < 16; row++) {
      alignas(64) int32_t q[16];
      for (int l = 0; l < 16; l++) q[l] = (l < 8 ? 0 : 16) + row;
      rsel[row] = _mm512_load_si512(q);
    }
    alignas(64) uint16_t p[32];
    for (int gg = 0; gg < 2; gg++)
      for (int r = 0; r < 2; r++)
        for (int h = 0; h < 2; h++)
          for (int t = 0; t < 4; t++)
            p[gg * 16 + r * 8 + h * 4 + t] = (uint16_t)(gg * 16 + t * 4 + h * 2 + r);
    perm = _mm512_load_si512(p);
    alignas(64) int32_t a[16], b[16];
    for (int l = 0; l < 16; l++) { a[l] = l & 7; b[l] = 2 * (l >> 3); }
    rowmod8 = _mm512_load_si512(a);
    rhi = _mm512_load_si512(b);
  }
};
const VQConsts& VC() { static VQConsts c; return c; }

// bit 0 = parity of each u16 (s < 128)
inline __m512i parity16(__m512i s, __m512i one16) {
#ifdef __AVX512BITALG__
  return _mm512_and_si512(_mm512_popcnt_epi16(s), one16);
#else
  s = _mm512_xor_si512(s, _mm512_srli_epi16(s, 4));
  s = _mm512_xor_si512(s, _mm512_srli_epi16(s, 2));
  s = _mm512_xor_si512(s, _mm512_srli_epi16(s, 1));
  return _mm512_and_si512(s, one16);
#endif
}
constexpr int VQ_KMAX = 8192;
#ifndef VQ_CP1
#define VQ_CP1 4
#endif
#ifndef VQ_CP2
#define VQ_CP2 2
#endif

// 64 values of chunk kb -> int8 in natural order (aq + kb*64), -128 * lane
// sums (ac + kb*16, the dpbusd start value), lane scales (asv + kb*16:
// lanes 0-7 k32 0, 8-15 k32 1)
inline void prep_kb_vq(const float* x, ActQ& o, int kb) {
  __m512 v[4];
  for (int i = 0; i < 4; i++) v[i] = _mm512_loadu_ps(x + 16 * i);
  float m0 = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v[0]), _mm512_abs_ps(v[1])));
  float m1 = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v[2]), _mm512_abs_ps(v[3])));
  float s0 = m0 > 0.f ? m0 / 127.f : 1.f, s1 = m1 > 0.f ? m1 / 127.f : 1.f;
  __m512 i0 = _mm512_set1_ps(1.f / s0), i1 = _mm512_set1_ps(1.f / s1);
  __m128i b0 = _mm512_cvtepi32_epi8(quant16(v[0], i0));
  __m128i b1 = _mm512_cvtepi32_epi8(quant16(v[1], i0));
  __m128i b2 = _mm512_cvtepi32_epi8(quant16(v[2], i1));
  __m128i b3 = _mm512_cvtepi32_epi8(quant16(v[3], i1));
  __m512i q = _mm512_inserti32x4(_mm512_castsi128_si512(b0), b1, 1);
  q = _mm512_inserti32x4(q, b2, 2);
  q = _mm512_inserti32x4(q, b3, 3);
  _mm512_store_si512(o.aq + (size_t)kb * 64, q);
  __m512i sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_set1_epi8(1), q);
  _mm512_store_si512(o.ac + (size_t)kb * 16,
                     _mm512_sub_epi32(_mm512_setzero_si512(), _mm512_slli_epi32(sum, 7)));
  _mm512_store_ps(o.asv + (size_t)kb * 16,
                  _mm512_mask_blend_ps((__mmask16)0xFF00, _mm512_set1_ps(s0),
                                       _mm512_set1_ps(s1)));
}

// the nb block's row scales R[kb][h][row] (times the table scale cs)
inline void vq_scales(const uint8_t* sp, int KB, float cs, float* R) {
  const VQConsts& v = VC();
  const __m512 rt2 = _mm512_set1_ps(1.41421356237f);
  for (int sb = 0; sb < KB / 4; sb++, sp += 48) {
    __m512i T = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)sp)),
                                 _mm512_set1_epi32(128));
    __m512i js = _mm512_permutexvar_epi32(
        v.rowmod8, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(sp + 16))));
    for (int c = 0; c < 8; c++) {
      __m512i j = _mm512_and_si512(
          _mm512_srlv_epi32(js, _mm512_add_epi32(v.rhi, _mm512_set1_epi32(4 * c))),
          _mm512_set1_epi32(3));
      __m512i h = _mm512_sub_epi32(T, j);
      __m512 f = _mm512_castsi512_ps(_mm512_slli_epi32(
          _mm512_add_epi32(_mm512_srai_epi32(h, 1), _mm512_set1_epi32(127)), 23));
      f = _mm512_mask_mul_ps(f, _mm512_test_epi32_mask(h, _mm512_set1_epi32(1)), f, rt2);
      // c = (kb & 3) * 2 + h
      _mm512_store_ps(R + ((size_t)(sb * 4 + (c >> 1)) * 2 + (c & 1)) * 16,
                      _mm512_mul_ps(f, _mm512_set1_ps(cs)));
    }
  }
}

// rows [nb*16, nb*16 + 16) of y_t = W x_t for NT tokens: 4 / CP passes over
// the block, CP 64-byte code chunks (chunk p: rows 2p, 2p+8, 2p+1, 2p+9)
// each, so every group is decoded once for all NT tokens
template <int NT, int CP>
inline void gemv_vq(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                    const ActQ* const* a, float* const* y, const uint8_t* tab,
                    float cs) {
  const VQConsts& v = VC();
  const int KB = K / 64;
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  alignas(64) float R[VQ_KMAX / 64 * 32];
  vq_scales(sc + (size_t)nb * (K / 256) * 48, KB, cs, R);
  const __m512i m7f = _mm512_set1_epi16(0x7f), one16 = _mm512_set1_epi16(1);
  for (int p0 = 0; p0 < 4; p0 += CP) {
    __m512 acc[NT][CP * 4];
    for (int t = 0; t < NT; t++)
      for (int r = 0; r < CP * 4; r++) acc[t][r] = _mm512_setzero_ps();
    for (int kb = 0; kb < KB; kb++) {
      if (p0 == 0) prefetch_kb(pb + (size_t)(kb + PF_KB) * 256);
      // per token: the row scales of this kb times its activation scales
      const __m512 R0 = _mm512_load_ps(R + (size_t)kb * 32),
                   R1 = _mm512_load_ps(R + (size_t)kb * 32 + 16);
      __m512 Rt[NT][2];
      __m512i At[NT], Ct[NT];
      for (int t = 0; t < NT; t++) {
        const ActQ& A = *a[t];
        Rt[t][0] = _mm512_mul_ps(R0, _mm512_set1_ps(A.asv[(size_t)kb * 16]));
        Rt[t][1] = _mm512_mul_ps(R1, _mm512_set1_ps(A.asv[(size_t)kb * 16 + 8]));
        At[t] = _mm512_load_si512(A.aq + (size_t)kb * 64);
        Ct[t] = _mm512_load_si512(A.ac + (size_t)kb * 16);
      }
      for (int c = 0; c < CP; c++) {
        const int p = p0 + c;
        const int rows[4] = {2 * p, 2 * p + 8, 2 * p + 1, 2 * p + 9};
        __m512i P = _mm512_permutexvar_epi16(v.perm,
                                             _mm512_loadu_si512(pb + kb * 256 + 64 * p));
        __m512i s7 = _mm512_and_si512(P, m7f);
        __m512i sg = _mm512_or_si512(s7, _mm512_slli_epi16(parity16(s7, one16), 7));
        alignas(32) uint64_t sm[4];
        _mm256_store_si256((__m256i*)sm, _mm512_cvtepi16_epi8(sg));
        __m512i ix = _mm512_srli_epi16(P, 7);
        __m512i ix32[2] = {_mm512_cvtepu16_epi32(_mm512_castsi512_si256(ix)),
                           _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(ix, 1))};
        for (int r = 0; r < 4; r++) {
          __m256i iv = (r & 1) ? _mm512_extracti64x4_epi64(ix32[r >> 1], 1)
                               : _mm512_castsi512_si256(ix32[r >> 1]);
          __m512i w = _mm512_i32gather_epi64(iv, (const void*)tab, 8);
          w = _mm512_mask_sub_epi8(w, (__mmask64)sm[r], _mm512_setzero_si512(), w);
          const __m512i sel = v.rsel[rows[r]];
          for (int t = 0; t < NT; t++) {
            __m512i d = _mm512_dpbusd_epi32(Ct[t], w, At[t]);
            acc[t][c * 4 + r] = _mm512_fmadd_ps(
                _mm512_cvtepi32_ps(d), _mm512_permutex2var_ps(Rt[t][0], sel, Rt[t][1]),
                acc[t][c * 4 + r]);
          }
        }
      }
    }
    for (int c = 0; c < CP; c++) {
      const int p = p0 + c;
      const int rows[4] = {2 * p, 2 * p + 8, 2 * p + 1, 2 * p + 9};
      for (int t = 0; t < NT; t++)
        for (int r = 0; r < 4; r++) y[t][rows[r]] = _mm512_reduce_add_ps(acc[t][c * 4 + r]);
    }
  }
}

inline void gemv_tokens_vq(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                           const ActQ* const* a, float* const* y, int n,
                           const uint8_t* tab, float cs) {
  for (int i = 0; i < n; i += 4) {
    switch (n - i < 4 ? n - i : 4) {
      case 1: gemv_vq<1, VQ_CP1>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      case 2: gemv_vq<2, VQ_CP2>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      case 3: gemv_vq<3, 1>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      default: gemv_vq<4, 1>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
    }
  }
}
#elif defined(__AVX2__) && defined(__FMA__)
// AVX2 (Zen2/Zen3): the 32 bytes of (nb, kb, g) are c = t*8 + p with
// p = half + 2*k32 + 4*tile. unpack{lo,hi}_epi64 of the low / high nibbles
// give V_v whose lane h holds the 8 bytes of t = 2h + v; X_{v,s} (s = half)
// puts in lane h, dword j = k32 + 2*tile, the 4 codes of byte p = s + 2j as
// u8 (w + 4). The four X share the dword -> (k32, tile) map, so their int16
// products add before one madd; activations are pre-permuted to match.
#define W2E_ISA "avx2"
struct Consts {
  __m256i ctl[2];       // V byte for (s, lane, k4)
  __m256i fmask, lut, nib, ones16;
  __m256i sca[8], scb[8];   // scale byte (k32, tile, row g) -> dword j
  uint8_t aidx[128];        // act byte of X_i lane h dword j k4
  Consts() {
    alignas(32) uint8_t b[32];
    for (int s = 0; s < 2; s++) {
      for (int i = 0; i < 32; i++) {
        int j = (i & 15) >> 2, k4 = i & 3;
        b[i] = (uint8_t)((k4 >= 2 ? 8 : 0) + s + 2 * j);
      }
      ctl[s] = _mm256_load_si256((const __m256i*)b);
    }
    for (int i = 0; i < 32; i++) b[i] = (i & 1) ? 0x0C : 0x03;
    fmask = _mm256_load_si256((const __m256i*)b);
    // field c (k4 even) or 4c (k4 odd) -> w + 4 for code c
    static const uint8_t val[4] = {0, 3, 5, 8};
    memset(b, 0, 32);
    for (int h = 0; h < 2; h++)
      for (int c = 0; c < 4; c++) { b[16 * h + c] = val[c]; b[16 * h + 4 * c] = val[c]; }
    lut = _mm256_load_si256((const __m256i*)b);
    nib = _mm256_set1_epi8(15);
    ones16 = _mm256_set1_epi16(1);
    // Y = the kb's 32 scale bytes (k32*16 + tile*8 + row), W = Y lanes swapped
    for (int g = 0; g < 8; g++) {
      alignas(32) uint8_t a[32], c[32];
      memset(a, 0x80, 32); memset(c, 0x80, 32);
      for (int h = 0; h < 2; h++)
        for (int j = 0; j < 4; j++) {
          int k32 = j & 1, off = (j >> 1) * 8 + g;
          (k32 == h ? a : c)[16 * h + 4 * j] = (uint8_t)off;
        }
      sca[g] = _mm256_load_si256((const __m256i*)a);
      scb[g] = _mm256_load_si256((const __m256i*)c);
    }
    for (int i = 0; i < 4; i++)
      for (int h = 0; h < 2; h++)
        for (int j = 0; j < 4; j++)
          for (int k4 = 0; k4 < 4; k4++)
            aidx[i * 32 + h * 16 + j * 4 + k4] = (uint8_t)(
                (j & 1) * 32 + (i & 1) * 16 + (2 * h + (i >> 1)) * 4 + k4);
  }
};
const Consts& C() { static Consts c; return c; }

struct ActQ {                 // one activation vector, int8 per 32-group
  int8_t* aq = nullptr;       // [KB][4][32] lane order
  int32_t* ac = nullptr;      // [KB][8] 4 * lane sums
  float* asv = nullptr;       // [KB][8] group scale per lane
  int cap = 0;
  void reserve(int K) {
    if (cap >= K) return;
    free(aq); free(ac); free(asv);
    aq = (int8_t*)aligned_alloc(64, (size_t)K * 2);
    ac = (int32_t*)aligned_alloc(64, (size_t)K / 8 * 4);
    asv = (float*)aligned_alloc(64, (size_t)K / 8 * 4);
    cap = K;
  }
};

inline float hmax8(__m256 v) {
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}

// 64 values of chunk kb -> lane-order int8, lane sums, lane scales
inline void prep_kb(const float* x, ActQ& o, int kb) {
  const Consts& c = C();
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 v[8];
  for (int i = 0; i < 8; i++) v[i] = _mm256_loadu_ps(x + 8 * i);
  __m256 a0 = _mm256_and_ps(v[0], absm), a1 = _mm256_and_ps(v[4], absm);
  for (int i = 1; i < 4; i++) {
    a0 = _mm256_max_ps(a0, _mm256_and_ps(v[i], absm));
    a1 = _mm256_max_ps(a1, _mm256_and_ps(v[4 + i], absm));
  }
  float m0 = hmax8(a0), m1 = hmax8(a1);
  float s0 = m0 > 0.f ? m0 / 127.f : 1.f, s1 = m1 > 0.f ? m1 / 127.f : 1.f;
  __m256 inv[2] = {_mm256_set1_ps(1.f / s0), _mm256_set1_ps(1.f / s1)};
  alignas(32) int32_t q[64];
  for (int i = 0; i < 8; i++) {
    __m256i t = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], inv[i >> 2]));
    t = _mm256_max_epi32(_mm256_min_epi32(t, _mm256_set1_epi32(127)),
                         _mm256_set1_epi32(-127));
    _mm256_store_si256((__m256i*)(q + 8 * i), t);
  }
  int8_t* aq = o.aq + (size_t)kb * 128;
  int32_t* ac = o.ac + (size_t)kb * 8;
  for (int l = 0; l < 8; l++) ac[l] = 0;
  for (int i = 0; i < 128; i++) {
    int8_t a = (int8_t)q[c.aidx[i]];
    aq[i] = a;
    ac[(i >> 2) & 7] += 4 * a;
  }
  float* asv = o.asv + (size_t)kb * 8;
  for (int l = 0; l < 8; l++) asv[l] = (l & 1) ? s1 : s0;
}

inline __m256 exp8(__m256 x) {
  x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.f)), _mm256_set1_ps(-88.f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693145752f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(1.42860677e-06f), r);
  __m256 p = _mm256_set1_ps(1.f / 720);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f / 120));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f / 24));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f / 6));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.f));
  __m256i e = _mm256_slli_epi32(
      _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// g = silu(min(gate, lim)) * clamp(up, +-lim) for 64 values, then prep_kb
// (vq: the natural-order layout of prep_kb_vq)
inline void prep_kb_vq(const float* x, ActQ& o, int kb);
inline void act_kb(const float* gate, const float* up, float lim, ActQ& o, int kb,
                   bool vq = false) {
  alignas(32) float g[64];
  __m256 L = _mm256_set1_ps(lim), NL = _mm256_set1_ps(-lim), one = _mm256_set1_ps(1.f);
  for (int i = 0; i < 8; i++) {
    __m256 gt = _mm256_min_ps(_mm256_loadu_ps(gate + 8 * i), L);
    __m256 u = _mm256_max_ps(_mm256_min_ps(_mm256_loadu_ps(up + 8 * i), L), NL);
    __m256 e = exp8(_mm256_sub_ps(_mm256_setzero_ps(), gt));
    _mm256_store_ps(g + 8 * i, _mm256_mul_ps(_mm256_div_ps(gt, _mm256_add_ps(one, e)), u));
  }
  if (vq) prep_kb_vq(g, o, kb);
  else prep_kb(g, o, kb);
}

// the 4 expanded X of chunk (kb, g) and its scale bits (k32, tile, row g)
struct Chunk {
  __m256i X[4];
  __m256 ws;
};

inline void expand(const Consts& c, const uint8_t* src, __m256i Y, __m256i W, int g,
                   Chunk& o) {
  __m256i S = _mm256_loadu_si256((const __m256i*)src);
  __m256i L = _mm256_and_si256(S, c.nib);
  __m256i Hn = _mm256_and_si256(_mm256_srli_epi16(S, 4), c.nib);
  __m256i V[2] = {_mm256_unpacklo_epi64(L, Hn), _mm256_unpackhi_epi64(L, Hn)};
  for (int i = 0; i < 4; i++)
    o.X[i] = _mm256_shuffle_epi8(
        c.lut, _mm256_and_si256(_mm256_shuffle_epi8(V[i >> 1], c.ctl[i & 1]), c.fmask));
  o.ws = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_or_si256(_mm256_shuffle_epi8(Y, c.sca[g]), _mm256_shuffle_epi8(W, c.scb[g])),
      23));
}

inline __m256 dot(const Consts& c, const Chunk& k, const ActQ& A, int kb, __m256 acc) {
  const __m256i* q = (const __m256i*)(A.aq + (size_t)kb * 128);
  __m256i p = _mm256_maddubs_epi16(k.X[0], _mm256_load_si256(q));
  for (int i = 1; i < 4; i++)
    p = _mm256_add_epi16(p, _mm256_maddubs_epi16(k.X[i], _mm256_load_si256(q + i)));
  __m256i d = _mm256_sub_epi32(_mm256_madd_epi16(p, c.ones16),
                               _mm256_load_si256((const __m256i*)(A.ac + (size_t)kb * 8)));
  return _mm256_fmadd_ps(_mm256_cvtepi32_ps(d),
                         _mm256_mul_ps(k.ws, _mm256_load_ps(A.asv + (size_t)kb * 8)), acc);
}

inline void store_rows(__m256 acc, int g, float* y) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  alignas(16) float f[4];
  _mm_store_ps(f, s);
  y[g] = f[0] + f[1];
  y[8 + g] = f[2] + f[3];
}

// rows [nb*16, nb*16 + 16) of y = W x: one contiguous pass, all 8 row
// groups in registers
inline void gemv_nb(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                    const ActQ& a, float* y) {
  const Consts& c = C();
  const int KB = K / 64, KS = K / 32;
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  const uint8_t* sb = sc + (size_t)nb * KS * 16;
  __m256 acc[8];
  for (int g = 0; g < 8; g++) acc[g] = _mm256_setzero_ps();
  for (int kb = 0; kb < KB; kb++) {
    __m256i Y = _mm256_loadu_si256((const __m256i*)(sb + kb * 32));
    __m256i W = _mm256_permute4x64_epi64(Y, 0x4E);
    for (int g = 0; g < 8; g++) {
      Chunk k;
      expand(c, pb + kb * 256 + 32 * g, Y, W, g, k);
      acc[g] = dot(c, k, a, kb, acc[g]);
    }
  }
  for (int g = 0; g < 8; g++) store_rows(acc[g], g, y);
}

// the same for NT tokens of one expert: each chunk is expanded once and dotted
// with every token; two row groups per pass over the block
template <int NT>
inline void gemv_nb_multi(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                          const ActQ* const* a, float* const* y) {
  const Consts& c = C();
  const int KB = K / 64, KS = K / 32;
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  const uint8_t* sb = sc + (size_t)nb * KS * 16;
  for (int g0 = 0; g0 < 8; g0 += 2) {
    __m256 acc[2][NT];
    for (int h = 0; h < 2; h++)
      for (int t = 0; t < NT; t++) acc[h][t] = _mm256_setzero_ps();
    for (int kb = 0; kb < KB; kb++) {
      if (g0 == 0) prefetch_kb(pb + (size_t)(kb + PF_KB) * 256);
      __m256i Y = _mm256_loadu_si256((const __m256i*)(sb + kb * 32));
      __m256i W = _mm256_permute4x64_epi64(Y, 0x4E);
      for (int h = 0; h < 2; h++) {
        Chunk k;
        expand(c, pb + kb * 256 + 32 * (g0 + h), Y, W, g0 + h, k);
        for (int t = 0; t < NT; t++) acc[h][t] = dot(c, k, *a[t], kb, acc[h][t]);
      }
    }
    for (int h = 0; h < 2; h++)
      for (int t = 0; t < NT; t++) store_rows(acc[h][t], g0 + h, y[t]);
  }
}

// 64 bf16 -> f32
inline void bf16_64(const uint16_t* s, float* xf) {
  for (int i = 0; i < 8; i++) {
    __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(s + 8 * i)));
    _mm256_store_ps(xf + 8 * i, _mm256_castsi256_ps(_mm256_slli_epi32(v, 16)));
  }
}
// ---- vector codes, AVX2 (layout as in the AVX-512 branch). The table is
// kept as signed int8 (w * 16, the XOR undone in w2e_vq); per (row, k32)
// four 8-byte patterns load into one register, the code signs negate them,
// and maddubs(|w|, sign(a, w)) dots them with natural-order int8
// activations (|w| <= 127, so the pair sums cannot saturate). The int32
// lanes of the 8 row groups reduce with hadd to one vector of 8 rows,
// scaled once per (kb, k32).
#define W2E_VQ 1
#define W2E_VQ_SIGNED_TAB 1
constexpr int VQ_KMAX = 8192;

inline __m256i parity16(__m256i s, __m256i one16) {
  s = _mm256_xor_si256(s, _mm256_srli_epi16(s, 4));
  s = _mm256_xor_si256(s, _mm256_srli_epi16(s, 2));
  s = _mm256_xor_si256(s, _mm256_srli_epi16(s, 1));
  return _mm256_and_si256(s, one16);
}

struct VQConsts {
  __m256i sgsel[4];     // (r, h): sign byte of group t -> qword t
  __m256i bitsel;
  VQConsts() {
    alignas(32) uint8_t b[32];
    for (int r = 0; r < 2; r++)
      for (int h = 0; h < 2; h++) {
        for (int i = 0; i < 32; i++) b[i] = (uint8_t)(((i >> 3) & 1) * 8 + 2 * (2 * h + r));
        sgsel[r * 2 + h] = _mm256_load_si256((const __m256i*)b);
      }
    for (int i = 0; i < 32; i++) b[i] = (uint8_t)(1 << (i & 7));
    bitsel = _mm256_load_si256((const __m256i*)b);
  }
};
const VQConsts& VC() { static VQConsts c; return c; }

// 64 values of chunk kb -> int8 in natural order (aq + kb*64), the two k32
// scales (asv + kb*2)
inline void prep_kb_vq(const float* x, ActQ& o, int kb) {
  const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
  __m256 v[8];
  for (int i = 0; i < 8; i++) v[i] = _mm256_loadu_ps(x + 8 * i);
  __m256 a0 = _mm256_and_ps(v[0], absm), a1 = _mm256_and_ps(v[4], absm);
  for (int i = 1; i < 4; i++) {
    a0 = _mm256_max_ps(a0, _mm256_and_ps(v[i], absm));
    a1 = _mm256_max_ps(a1, _mm256_and_ps(v[4 + i], absm));
  }
  float m0 = hmax8(a0), m1 = hmax8(a1);
  float s0 = m0 > 0.f ? m0 / 127.f : 1.f, s1 = m1 > 0.f ? m1 / 127.f : 1.f;
  __m256 inv[2] = {_mm256_set1_ps(1.f / s0), _mm256_set1_ps(1.f / s1)};
  __m256i q[8];
  for (int i = 0; i < 8; i++) {
    q[i] = _mm256_cvtps_epi32(_mm256_mul_ps(v[i], inv[i >> 2]));
    q[i] = _mm256_max_epi32(_mm256_min_epi32(q[i], _mm256_set1_epi32(127)),
                            _mm256_set1_epi32(-127));
  }
  const __m256i fix = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (int i = 0; i < 2; i++) {
    __m256i w = _mm256_packs_epi16(_mm256_packs_epi32(q[4 * i], q[4 * i + 1]),
                                   _mm256_packs_epi32(q[4 * i + 2], q[4 * i + 3]));
    _mm256_store_si256((__m256i*)(o.aq + (size_t)kb * 64 + 32 * i),
                       _mm256_permutevar8x32_epi32(w, fix));
  }
  o.asv[(size_t)kb * 2] = s0;
  o.asv[(size_t)kb * 2 + 1] = s1;
}

// the nb block's row scales R[kb][h][row] (times the table scale cs)
inline void vq_scales(const uint8_t* sp, int KB, float cs, float* R) {
  const __m256 rt2 = _mm256_set1_ps(1.41421356237f), csv = _mm256_set1_ps(cs);
  const __m256i three = _mm256_set1_epi32(3), one = _mm256_set1_epi32(1);
  for (int sb = 0; sb < KB / 4; sb++, sp += 48) {
    __m256i T[2];
    for (int r = 0; r < 2; r++)
      T[r] = _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(sp + 8 * r))),
                              _mm256_set1_epi32(128));
    __m256i js = _mm256_loadu_si256((const __m256i*)(sp + 16));
    for (int c = 0; c < 8; c++)
      for (int r = 0; r < 2; r++) {
        __m256i j = _mm256_and_si256(_mm256_srli_epi32(js, 2 * (2 * c + r)), three);
        __m256i h = _mm256_sub_epi32(T[r], j);
        __m256 f = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(_mm256_srai_epi32(h, 1), _mm256_set1_epi32(127)), 23));
        __m256 odd = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(h, one), one));
        f = _mm256_blendv_ps(f, _mm256_mul_ps(f, rt2), odd);
        _mm256_store_ps(R + ((size_t)(sb * 4 + (c >> 1)) * 2 + (c & 1)) * 16 + 8 * r,
                        _mm256_mul_ps(f, csv));
      }
  }
}

// 8 int32 vectors -> one vector of their lane sums
inline __m256i hsum8(const __m256i* d) {
  __m256i y = _mm256_hadd_epi32(_mm256_hadd_epi32(d[0], d[1]), _mm256_hadd_epi32(d[2], d[3]));
  __m256i z = _mm256_hadd_epi32(_mm256_hadd_epi32(d[4], d[5]), _mm256_hadd_epi32(d[6], d[7]));
  return _mm256_add_epi32(_mm256_permute2x128_si256(y, z, 0x20),
                          _mm256_permute2x128_si256(y, z, 0x31));
}

// rows [nb*16, nb*16 + 16) of y_t = W x_t for NT tokens; one pass per
// (r, h), every group decoded once for all NT tokens
template <int NT>
inline void gemv_vq(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                    const ActQ* const* a, float* const* y, const uint8_t* tab,
                    float cs) {
  const VQConsts& v = VC();
  const int KB = K / 64;
  const uint8_t* pb = plane + (size_t)nb * KB * 256;
  const uint64_t* tb = (const uint64_t*)tab;
  alignas(32) float R[VQ_KMAX / 64 * 32];
  vq_scales(sc + (size_t)nb * (K / 256) * 48, KB, cs, R);
  const __m256i m7f = _mm256_set1_epi16(0x7f), one16 = _mm256_set1_epi16(1);
  __m256 acc[NT][4];
  for (int t = 0; t < NT; t++)
    for (int i = 0; i < 4; i++) acc[t][i] = _mm256_setzero_ps();
  for (int kb = 0; kb < KB; kb++) {
    prefetch_kb(pb + (size_t)(kb + PF_KB) * 256);
    const uint8_t* blk = pb + (size_t)kb * 256;
    __m256i Sg[8];
    alignas(32) uint16_t ix[8][16];
    for (int g = 0; g < 8; g++) {
      __m256i C = _mm256_loadu_si256((const __m256i*)(blk + 32 * g));
      __m256i s7 = _mm256_and_si256(C, m7f);
      Sg[g] = _mm256_or_si256(s7, _mm256_slli_epi16(parity16(s7, one16), 7));
      _mm256_store_si256((__m256i*)ix[g], _mm256_srli_epi16(C, 7));
    }
    for (int rh = 0; rh < 4; rh++) {
      const int r = rh >> 1, h = rh & 1;
      __m256i A[NT];
      for (int t = 0; t < NT; t++)
        A[t] = _mm256_load_si256((const __m256i*)(a[t]->aq + (size_t)kb * 64 + 32 * h));
      __m256i d[NT][8];
      for (int g = 0; g < 8; g++) {
        const int u = 2 * h + r;
        __m256i W = _mm256_setr_epi64x((long long)tb[ix[g][u]], (long long)tb[ix[g][4 + u]],
                                       (long long)tb[ix[g][8 + u]], (long long)tb[ix[g][12 + u]]);
        __m256i M = _mm256_shuffle_epi8(Sg[g], v.sgsel[rh]);
        M = _mm256_cmpeq_epi8(_mm256_and_si256(M, v.bitsel), v.bitsel);
        W = _mm256_sub_epi8(_mm256_xor_si256(W, M), M);
        __m256i aw = _mm256_abs_epi8(W);
        for (int t = 0; t < NT; t++)
          d[t][g] = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(A[t], W)),
                                      one16);
      }
      const __m256 Rv = _mm256_load_ps(R + ((size_t)kb * 2 + h) * 16 + 8 * r);
      for (int t = 0; t < NT; t++)
        acc[t][rh] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(hsum8(d[t])),
            _mm256_mul_ps(Rv, _mm256_set1_ps(a[t]->asv[(size_t)kb * 2 + h])), acc[t][rh]);
    }
  }
  for (int t = 0; t < NT; t++) {
    _mm256_storeu_ps(y[t], _mm256_add_ps(acc[t][0], acc[t][1]));
    _mm256_storeu_ps(y[t] + 8, _mm256_add_ps(acc[t][2], acc[t][3]));
  }
}

inline void gemv_tokens_vq(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                           const ActQ* const* a, float* const* y, int n,
                           const uint8_t* tab, float cs) {
  for (int i = 0; i < n; i += 4) {
    switch (n - i < 4 ? n - i : 4) {
      case 1: gemv_vq<1>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      case 2: gemv_vq<2>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      case 3: gemv_vq<3>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
      default: gemv_vq<4>(plane, sc, K, nb, a + i, y + i, tab, cs); break;
    }
  }
}
#else
#error "moe_w2_cpu needs AVX-512 VNNI + VBMI or AVX2 + FMA"
#endif

// y_t for n tokens of one expert, in chunks of up to 4
inline void gemv_tokens(const uint8_t* plane, const uint8_t* sc, int K, int nb,
                        const ActQ* const* a, float* const* y, int n) {
  for (int i = 0; i < n; i += 4) {
    switch (n - i < 4 ? n - i : 4) {
      case 1: gemv_nb(plane, sc, K, nb, *a[i], y[i]); break;
      case 2: gemv_nb_multi<2>(plane, sc, K, nb, a + i, y + i); break;
      case 3: gemv_nb_multi<3>(plane, sc, K, nb, a + i, y + i); break;
      default: gemv_nb_multi<4>(plane, sc, K, nb, a + i, y + i); break;
    }
  }
}

// ---------------------------------------------------------------- engine
// One master thread watches the doorbell the decode graph rings per MoE
// layer (pinned host memory). A request names the layer; its block holds
// the routing, the per-route CPU mask and x (bf16). The master and n-1
// spinning workers compute the masked pairs from the pinned host rows in
// four phases (activation prep, w13, act + requant, w2 with the weighted
// sum), write y (f32) and ack with the request's sequence number; the
// graph polls the ack before adding y to the layer output.

constexpr int HDR = 16;     // i32 words per layer: 0 req seq, 1 ack seq, 2 T
constexpr int TMAX = 32;   // = _TCAP in moe_w2_cpu.py

inline void cpu_relax() { _mm_pause(); }

// Idle wait for the polling threads: spin `spin` times, then 20 us naps (next layer or step comes soon), and after
// VLLM_MOE_W2_CPU_IDLE_MS (default 50) without work, 1 ms naps. Without the last stage an idle server kept every
// worker at ~11% CPU (IDLE1: 6.5 cores on a 60-thread epyc, ~1.4 on a 14600KF); its cost is up to ~1 ms on the
// first job after a pause. 0 = 20 us naps forever.
inline long idle_ms_cfg() {
  static const long v = [] { const char* e = getenv("VLLM_MOE_W2_CPU_IDLE_MS"); return e ? atol(e) : 50L; }();
  return v;
}
struct IdleWait {
  long n = 0;
  std::chrono::steady_clock::time_point t0;
  void reset() { n = 0; }
  void wait(long spin) {
    if (++n <= spin) { cpu_relax(); return; }
    auto now = std::chrono::steady_clock::now();
    if (n == spin + 1) t0 = now;
    long lim = idle_ms_cfg();
    bool deep = lim > 0 && now - t0 > std::chrono::milliseconds(lim);
    std::this_thread::sleep_for(std::chrono::microseconds(deep ? 1000 : 20));
  }
};

void pin_to(int cpu) {
  if (cpu < 0) return;
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

struct Engine {
  int n_keys, H, I, top_k, n_experts, n_thr;
  size_t slot_bytes, off_s13, off_c2, off_s2;
  float limit;
  volatile int32_t* hdr;
  volatile int64_t* glob;
  const int32_t* ids;
  const float* wts;
  const int32_t* msk;
  const uint16_t* xb;
  float* y;
  std::vector<const uint8_t*> host;
  std::vector<int> cpus;      // logical CPU per thread (master = 0), or -1

  // current job: pairs grouped by expert (ex_first[x] .. ex_first[x+1])
  std::vector<const uint8_t*> ex_row;
  std::vector<int> ex_first, tok;
  std::vector<float> pw;
  int T = 0, n_pairs = 0, n_ex = 0;
  const uint16_t* jx = nullptr;
  float* jy = nullptr;
  ActQ a1[TMAX];
  std::vector<ActQ> a2;
  float* h = nullptr;
  // vector codes (w2e_vq): u8 [P][8] pattern table, its scale
  uint8_t* vqtab = nullptr;
  float vqcs = 0.f;

  // cold rows (w2e_cold): a layer whose host rows hold only its hot experts
  // (compact, rowmap[e] = row) reads each other expert's four sections from
  // the planes files (section-major, expert e at e * section bytes) into a
  // bounce row. Dedicated I/O threads read the w13 sections first, then w2,
  // while the pool runs prep and the hot experts' w13 + act; the cold
  // experts' w13 + act wait for their w13 bytes, w2 (all experts, route
  // order, so y is bit-identical) for the rest.
  struct Chunk { int fd; off_t off; uint8_t* dst; size_t len; };
  std::vector<const int32_t*> rowmap;   // per key; null = full layout
  std::vector<int> cfd;                 // [key * 4 + section]
  uint8_t* bounce = nullptr;
  int n_bounce = 0;
  bool cold_drop = false;               // fadvise(DONTNEED) after each read
  std::vector<Chunk> chunks;            // w13 chunks [0, n13), then w2
  long n13 = 0;
  std::atomic<long> io_next{0}, io_done13{0}, io_done{0};
  std::atomic<int> io_busy{0};           // woken I/O threads not yet idle
  std::atomic<uint64_t> n_cold{0}, cold_ns{0}, cold_wait_ns{0}, io_err{0};
  std::vector<std::thread> io_th;
  std::mutex io_mu;
  std::condition_variable io_cv;
  uint64_t io_gen = 0;
  // the experts (phase 1) and pairs (phase 2) of the current pass
  std::vector<int> xs, ps;
  // read-ahead hints (w2e_hint): while layer L runs, its graph posts the
  // router-lookahead prediction of layer L+1's experts (hint block: hdr 0
  // seq, 2 count; ids); a thread queues the predicted cold experts' chunks
  // and prefetch threads pread them into the page cache (fadvise WILLNEED
  // reads only the readahead window, ~128 KiB), so the job's preads find
  // them in flight or cached. Never waited on; a wrong guess costs
  // bandwidth only; chunks queued longer than PF_TTL_NS are dropped.
  struct PChunk { int fd; off_t off; size_t len; int64_t t; };
  static constexpr int64_t PF_TTL_NS = 10000000;
  volatile int32_t* hhdr = nullptr;
  const volatile int32_t* hids = nullptr;
  int hstride = 0;
  int htop = 8;                           // hint prediction ranks < htop only
  std::thread hint_th;
  std::vector<std::thread> pf_th;
  std::mutex pf_mu;
  std::condition_variable pf_cv;
  std::vector<PChunk> pf_q;               // LIFO: the newest hint first
  std::atomic<uint64_t> n_hint{0}, n_pf_drop{0};
  // hint accounting: when (and at which prediction rank, i mod 8) each
  // (key, expert) was last hinted; a job's cold read of it within
  // HINT_HIT_NS counts as a used hint (once). pf_done = chunks actually read
  static constexpr int64_t HINT_HIT_NS = 50000000;
  std::vector<int64_t> hint_t;
  std::vector<int8_t> hint_rk;
  std::atomic<uint64_t> n_cold_hinted{0}, n_pf_done{0}, n_cold_pairs{0};
  std::atomic<uint64_t> n_hint_rk[8] = {}, n_used_rk[8] = {};

  // pool
  std::atomic<uint64_t> gen{0};
  std::atomic<int> done{0};
  std::atomic<long> next_unit{0};   // dynamic split: next unit of the running phase
  bool dyn_sched = true;
  std::atomic<bool> stop{false};
  int phase = 0;
  std::vector<std::thread> th;
  std::thread master;

  std::atomic<uint64_t> n_jobs{0}, n_pairs_tot{0}, busy_ns{0}, max_ns{0};

  // helper-GPU mode (moe_w2_helper): a doorbell launches the layer's
  // captured graph for the token bucket, then the ack as a stream write
  using launch_fn = int (*)(void*, void*);
  using write_fn = int (*)(void*, uint64_t, uint32_t, unsigned);
  using ctx_fn = int (*)(void*);
  bool gpu = false;
  void* g_ctx = nullptr;
  void* g_stream = nullptr;
  std::vector<void*> g_exec;
  std::vector<int> g_bucket;
  int g_nb = 0;
  launch_fn g_launch = nullptr;
  write_fn g_write = nullptr;
  ctx_fn g_setctx = nullptr;

  void gpu_job(int key, int seq) {
    volatile int32_t* hd = hdr + (size_t)key * HDR;
    int T = hd[2];
    T = T < 1 ? 1 : (T > TMAX ? TMAX : T);
    g_launch(g_exec[(size_t)key * g_nb + g_bucket[T]], g_stream);
    g_write(g_stream, (uint64_t)(uintptr_t)(hd + 1), (uint32_t)seq, 0);
    n_jobs++;
  }
  uint64_t phase_ns[4] = {0, 0, 0, 0};    // master's wall time per phase

  void units(int ph, int tid) {
    const int N13 = 2 * I, KB1 = H / 64, KB2 = I / 64, nb13 = N13 / 16, nb2 = H / 16;
    long n = ph == 0 ? (long)T * KB1 : ph == 1 ? (long)xs.size() * nb13
           : ph == 2 ? (long)ps.size() * KB2 : nb2;
    // dynamic split (default): threads take chunks of units from a shared counter, so fast cores (P-cores on
    // hybrid CPUs) take more and no phase waits for the slowest core. Units write disjoint outputs and each is
    // computed the same way on any thread, so results are bit-identical to the static split
    // (VLLM_MOE_W2_CPU_DYN=0: equal shares per thread, the old behaviour).
    const bool dyn = dyn_sched;
    const long ch = dyn ? std::max(1L, n / ((long)n_thr * 8)) : 0;
    long lo = dyn ? 0 : n * tid / n_thr, hi = dyn ? 0 : n * (tid + 1) / n_thr;
    for (;;) {
    if (dyn) {
      lo = next_unit.fetch_add(ch, std::memory_order_relaxed);
      if (lo >= n) break;
      hi = std::min(n, lo + ch);
    }
    for (long u = lo; u < hi; u++) {
      if (ph == 0) {
        int t = (int)(u / KB1), kb = (int)(u % KB1);
        alignas(64) float xf[64];
        const uint16_t* s = jx + (size_t)t * H + kb * 64;
        bf16_64(s, xf);
#ifdef W2E_VQ
        if (vqtab) { prep_kb_vq(xf, a1[t], kb); continue; }
#endif
        prep_kb(xf, a1[t], kb);
      } else if (ph == 1) {
        int x = xs[u / nb13], nb = (int)(u % nb13);
        int p0 = ex_first[x], n = ex_first[x + 1] - p0;
        const ActQ* a[TMAX * 8];
        float* o[TMAX * 8];
        for (int i = 0; i < n; i++) {
          a[i] = &a1[tok[p0 + i]];
          o[i] = h + (size_t)(p0 + i) * N13 + nb * 16;
        }
#ifdef W2E_VQ
        if (vqtab) {
          gemv_tokens_vq(ex_row[x], ex_row[x] + off_s13, H, nb, a, o, n, vqtab, vqcs);
          continue;
        }
#endif
        gemv_tokens(ex_row[x], ex_row[x] + off_s13, H, nb, a, o, n);
      } else if (ph == 2) {
        int p = ps[u / KB2], kb = (int)(u % KB2);
        const float* hp = h + (size_t)p * N13;
#ifdef W2E_VQ
        if (vqtab) { act_kb(hp + kb * 64, hp + I + kb * 64, limit, a2[p], kb, true); continue; }
#endif
        act_kb(hp + kb * 64, hp + I + kb * 64, limit, a2[p], kb);
      } else {
        int nb = (int)u;
        float yy[TMAX][16];
        for (int x = 0; x < n_ex; x++) {
          int p0 = ex_first[x], n = ex_first[x + 1] - p0;
          const ActQ* a[TMAX];
          float* o[TMAX];
          for (int i = 0; i < n; i++) { a[i] = &a2[p0 + i]; o[i] = yy[i]; }
#ifdef W2E_VQ
          if (vqtab)
            gemv_tokens_vq(ex_row[x] + off_c2, ex_row[x] + off_s2, I, nb, a, o, n, vqtab,
                           vqcs);
          else
#endif
          gemv_tokens(ex_row[x] + off_c2, ex_row[x] + off_s2, I, nb, a, o, n);
          for (int i = 0; i < n; i++) {
            float* d = jy + (size_t)tok[p0 + i] * H + nb * 16;
            for (int r = 0; r < 16; r++) d[r] += pw[p0 + i] * yy[i][r];
          }
        }
      }
    }
    if (!dyn) break;
    }
  }

  void worker(int tid) {
    pin_to(cpus[tid]);
    uint64_t seen = 0;
    IdleWait idle;
    while (!stop.load(std::memory_order_relaxed)) {
      uint64_t g = gen.load(std::memory_order_acquire);
      if (g == seen) {
        idle.wait(200000);
        continue;
      }
      idle.reset();
      seen = g;
      units(phase, tid);
      done.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void read_chunk(const Chunk& c) {
    size_t got = 0;
    while (got < c.len) {
      ssize_t r = pread(c.fd, c.dst + got, c.len - got, c.off + (off_t)got);
      if (r <= 0) { io_err++; memset(c.dst + got, 0, c.len - got); break; }
      got += (size_t)r;
    }
    if (cold_drop) posix_fadvise(c.fd, c.off, (off_t)c.len, POSIX_FADV_DONTNEED);
  }

  // I/O threads sleep between jobs (they must not steal the pool's cores)
  void io_worker() {
    uint64_t seen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lk(io_mu);
        io_cv.wait(lk, [&] { return stop.load() || io_gen != seen; });
        if (stop.load()) return;
        seen = io_gen;
      }
      long n = (long)chunks.size();
      for (long u; (u = io_next.fetch_add(1)) < n;) {
        read_chunk(chunks[u]);
        if (u < n13) io_done13.fetch_add(1, std::memory_order_release);
        io_done.fetch_add(1, std::memory_order_release);
      }
      io_busy.fetch_sub(1, std::memory_order_release);
    }
  }

  void hint_loop() {
    std::vector<int32_t> served(n_keys);
    for (int k = 0; k < n_keys; k++) served[k] = hhdr[(size_t)k * HDR];
    const size_t soff[5] = {0, off_s13, off_c2, off_s2, slot_bytes};
    IdleWait idle;
    while (!stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (int k = 0; k < n_keys; k++) {
        int32_t s = hhdr[(size_t)k * HDR];
        if (s == served[k]) continue;
        served[k] = s;
        any = true;
        std::atomic_thread_fence(std::memory_order_acquire);
        const int32_t* rm = rowmap[k];
        int n = std::min((int)hhdr[(size_t)k * HDR + 2], hstride);
        if (!rm) continue;
        int got[128], ng = 0;
        int64_t now = now_ns();
        for (int i = 0; i < n; i++) {
          int e = hids[(size_t)k * hstride + i];
          if ((i & 7) >= htop || e < 0 || e >= n_experts || rm[e] >= 0) continue;
          bool dup = false;
          for (int j = 0; j < ng && !dup; j++) dup = got[j] == e;
          if (dup || ng == 128) continue;
          got[ng++] = e;
          n_hint++;
          size_t hi = (size_t)k * n_experts + e;
          hint_rk[hi] = (int8_t)(i & 7);
          __atomic_store_n(&hint_t[hi], now, __ATOMIC_RELEASE);
          n_hint_rk[i & 7]++;
        }
        if (ng) {
          std::lock_guard<std::mutex> lk(pf_mu);
          // w2 sections pushed first so the w13 ones (needed first) pop first
          for (int q = 3; q >= 0; q--) {
            size_t len = soff[q + 1] - soff[q];
            for (int j = ng - 1; j >= 0; j--) {
              off_t base = (off_t)got[j] * (off_t)len;
              for (size_t o = 0; o < len; o += (1u << 20))
                pf_q.push_back({cfd[(size_t)k * 4 + q], base + (off_t)o,
                                std::min(len - o, (size_t)1 << 20), now});
            }
          }
        }
        if (ng) pf_cv.notify_all();
      }
      if (any) { idle.reset(); continue; }
      idle.wait(2000);
    }
  }

  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  void pf_worker() {
    uint8_t* buf = (uint8_t*)aligned_alloc(4096, 1u << 20);
    while (true) {
      PChunk c;
      {
        std::unique_lock<std::mutex> lk(pf_mu);
        pf_cv.wait(lk, [&] { return stop.load() || !pf_q.empty(); });
        if (stop.load()) break;
        c = pf_q.back();
        pf_q.pop_back();
      }
      if (now_ns() - c.t > PF_TTL_NS) { n_pf_drop++; continue; }
      size_t got = 0;
      while (got < c.len) {
        ssize_t r = pread(c.fd, buf, c.len - got, c.off + (off_t)got);
        if (r <= 0) break;
        got += (size_t)r;
      }
      n_pf_done++;
    }
    free(buf);
  }

  void wait_io(const std::atomic<long>& cnt, long n) {
    if (cnt.load(std::memory_order_acquire) >= n) return;
    auto w0 = std::chrono::steady_clock::now();
    while (cnt.load(std::memory_order_acquire) < n) cpu_relax();
    cold_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - w0).count();
  }

  void run_phase(int ph) {
    phase = ph;
    done.store(0, std::memory_order_relaxed);
    next_unit.store(0, std::memory_order_relaxed);
    gen.fetch_add(1, std::memory_order_acq_rel);
    units(ph, 0);
    while (done.load(std::memory_order_acquire) < n_thr - 1) cpu_relax();
  }

  void job(int key, int seq) {
    auto t0 = std::chrono::steady_clock::now();
    volatile int32_t* hd = hdr + (size_t)key * HDR;
    T = hd[2];
    if (T > TMAX) T = TMAX;
    // group the masked routes by expert (a token routes an expert once)
    const int R = TMAX * top_k;
    int ex_id[TMAX * 8], cnt[TMAX * 8], rt[TMAX * 8];
    n_ex = 0;
    int nr = 0;
    for (int t = 0; t < T; t++)
      for (int k = 0; k < top_k; k++) {
        size_t i = (size_t)key * R + t * top_k + k;
        if (!msk[i] || host[key] == nullptr || ids[i] < 0 || ids[i] >= n_experts)
          continue;
        rt[nr++] = (int)i;
        int e = ids[i], x = 0;
        while (x < n_ex && ex_id[x] != e) x++;
        if (x == n_ex) { ex_id[n_ex] = e; cnt[n_ex++] = 0; }
        cnt[x]++;
      }
    ex_row.resize(n_ex);
    ex_first.assign(n_ex + 1, 0);
    const int32_t* rm = rowmap.empty() ? nullptr : rowmap[key];
    int nc = 0;
    if (rm) {
      for (int x = 0; x < n_ex; x++) nc += rm[ex_id[x]] < 0;
      if (nc > n_bounce) {
        free(bounce);
        n_bounce = nc + 8;
        bounce = (uint8_t*)aligned_alloc(4096, (size_t)n_bounce * slot_bytes);
      }
    }
    while (io_busy.load(std::memory_order_acquire) > 0) cpu_relax();   // chunks is ours
    chunks.clear();
    const size_t soff[5] = {0, off_s13, off_c2, off_s2, slot_bytes};
    std::vector<int> hot, cold;
    for (int x = 0, c = 0; x < n_ex; x++) {
      int r = rm ? rm[ex_id[x]] : ex_id[x];
      if (r >= 0) {
        ex_row[x] = host[key] + (size_t)r * slot_bytes;
        hot.push_back(x);
      } else {
        ex_row[x] = bounce + (size_t)(c++) * slot_bytes;
        cold.push_back(x);
      }
      ex_first[x + 1] = ex_first[x] + cnt[x];
    }
    int ncp = 0;                                  // pairs on cold rows
    for (int x : cold) ncp += cnt[x];
    if (!hint_t.empty() && !cold.empty()) {
      int64_t now = now_ns();
      for (int x : cold) {
        size_t hi = (size_t)key * n_experts + ex_id[x];
        int64_t t = __atomic_load_n(&hint_t[hi], __ATOMIC_ACQUIRE);
        if (t && now - t < HINT_HIT_NS) {
          n_cold_hinted++;
          n_used_rk[hint_rk[hi] & 7]++;
          __atomic_store_n(&hint_t[hi], (int64_t)0, __ATOMIC_RELAXED);
        }
      }
    }
    for (int pass = 0; pass < 2; pass++) {       // w13 sections, then w2
      for (int x : cold)
        for (int s = 2 * pass; s < 2 * pass + 2; s++) {
          size_t len = soff[s + 1] - soff[s];
          off_t base = (off_t)ex_id[x] * (off_t)len;
          for (size_t o = 0; o < len; o += (1u << 20))
            chunks.push_back({cfd[(size_t)key * 4 + s], base + (off_t)o,
                              const_cast<uint8_t*>(ex_row[x]) + soff[s] + o,
                              std::min(len - o, (size_t)1 << 20)});
        }
      if (pass == 0) n13 = (long)chunks.size();
    }
    tok.resize(nr);
    pw.resize(nr);
    std::vector<int> fill(ex_first.begin(), ex_first.end() - 1);
    for (int r = 0; r < nr; r++) {
      int i = rt[r], e = ids[i], x = 0;
      while (ex_id[x] != e) x++;
      int p = fill[x]++;
      tok[p] = (i - key * R) / top_k;
      pw[p] = wts[i];
    }
    n_pairs = nr;
    jx = xb + (size_t)key * TMAX * H;
    jy = y + (size_t)key * TMAX * H;
    memset(jy, 0, sizeof(float) * (size_t)T * H);
    if (n_pairs) {
      auto c0 = std::chrono::steady_clock::now();
      long nch = (long)chunks.size();
      if (nch) {
        io_next.store(0);
        io_done13.store(0);
        io_done.store(0);
        if (io_th.empty()) {      // no I/O threads: read on the master
          for (auto& ch : chunks) read_chunk(ch);
          io_done13.store(n13);
          io_done.store(nch);
        } else {
          {
            std::lock_guard<std::mutex> lk(io_mu);
            io_busy.store((int)io_th.size());
            io_gen++;
          }
          io_cv.notify_all();
        }
      }
      auto timed = [&](int ph) {
        auto p0 = std::chrono::steady_clock::now();
        run_phase(ph);
        phase_ns[ph] += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - p0).count();
      };
      auto pairs_of = [&](const std::vector<int>& xv) {
        ps.clear();
        for (int x : xv)
          for (int p = ex_first[x]; p < ex_first[x + 1]; p++) ps.push_back(p);
      };
      timed(0);
      for (int pass = 0; pass < 2; pass++) {    // hot experts, then cold
        xs = pass ? cold : hot;
        if (xs.empty()) continue;
        if (pass) wait_io(io_done13, n13);
        pairs_of(xs);
        timed(1);
        timed(2);
      }
      if (nch) {
        wait_io(io_done, nch);
        n_cold += nc;
        n_cold_pairs += ncp;
        cold_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - c0).count();
      }
      timed(3);
      n_jobs++;
      n_pairs_tot += n_pairs;
      uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - t0).count();
      busy_ns += ns;
      if (ns > max_ns) max_ns = ns;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    hd[1] = seq;
  }

  void master_loop() {
    pin_to(cpus[0]);
    if (gpu) g_setctx(g_ctx);
    int64_t last = glob[0];
    IdleWait idle;
    while (!stop.load(std::memory_order_relaxed)) {
      int64_t v = glob[0];
      if (v == last) {
        idle.wait(2000000);
        continue;
      }
      idle.reset();
      last = v;
      std::atomic_thread_fence(std::memory_order_acquire);
      if (gpu) gpu_job((int)(v & 127), (int)(v >> 7));
      else job((int)(v & 127), (int)(v >> 7));
    }
  }
};

}  // namespace

extern "C" {

void* w2e_create(int n_keys, const uint64_t* host, uint64_t slot_bytes,
                 uint64_t off_s13, uint64_t off_c2, uint64_t off_s2, int H,
                 int I, int top_k, int n_experts, float limit, int n_thr, void* hdr,
                 void* glob, void* ids, void* wts, void* msk, void* x,
                 void* y, const int* cpus) {
  Engine* e = new Engine();
  e->n_keys = n_keys; e->H = H; e->I = I; e->top_k = top_k;
  e->n_experts = n_experts;
  e->n_thr = n_thr < 1 ? 1 : n_thr;
  e->slot_bytes = slot_bytes; e->off_s13 = off_s13; e->off_c2 = off_c2;
  e->off_s2 = off_s2; e->limit = limit;
  e->hdr = (volatile int32_t*)hdr; e->glob = (volatile int64_t*)glob;
  e->ids = (const int32_t*)ids; e->wts = (const float*)wts;
  e->msk = (const int32_t*)msk; e->xb = (const uint16_t*)x; e->y = (float*)y;
  e->host.assign(n_keys, nullptr);
  e->cpus.assign(e->n_thr, -1);
  if (cpus)
    for (int i = 0; i < e->n_thr; i++) e->cpus[i] = cpus[i];
  for (int i = 0; i < n_keys; i++) e->host[i] = (const uint8_t*)host[i];
  for (int t = 0; t < TMAX; t++) e->a1[t].reserve(H);
  const int R = TMAX * top_k;
  e->a2.resize(R);
  for (auto& a : e->a2) a.reserve(I);
  e->h = (float*)aligned_alloc(64, sizeof(float) * (size_t)R * 2 * I);
  for (int t = 1; t < e->n_thr; t++) e->th.emplace_back(&Engine::worker, e, t);
  e->master = std::thread(&Engine::master_loop, e);
  return e;
}

void* w2e_create_gpu(int n_keys, void* hdr, void* glob, void* ctx,
                     void* stream, const uint64_t* execs, int nb,
                     const int* bucket) {
  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) return nullptr;
  Engine* e = new Engine();
  e->g_launch = (Engine::launch_fn)dlsym(lib, "cuGraphLaunch");
  e->g_write = (Engine::write_fn)dlsym(lib, "cuStreamWriteValue32_v2");
  if (!e->g_write) e->g_write = (Engine::write_fn)dlsym(lib, "cuStreamWriteValue32");
  e->g_setctx = (Engine::ctx_fn)dlsym(lib, "cuCtxSetCurrent");
  if (!e->g_launch || !e->g_write || !e->g_setctx) { delete e; return nullptr; }
  e->gpu = true;
  e->n_keys = n_keys; e->n_thr = 1;
  e->hdr = (volatile int32_t*)hdr; e->glob = (volatile int64_t*)glob;
  e->g_ctx = ctx; e->g_stream = stream; e->g_nb = nb;
  for (size_t i = 0; i < (size_t)n_keys * nb; i++)
    e->g_exec.push_back((void*)(uintptr_t)execs[i]);
  e->g_bucket.assign(bucket, bucket + TMAX + 1);
  e->cpus.assign(1, -1);
  e->master = std::thread(&Engine::master_loop, e);
  return e;
}

void w2e_stats(void* p, uint64_t* out) {
  Engine* e = (Engine*)p;
  out[0] = e->n_jobs; out[1] = e->n_pairs_tot; out[2] = e->busy_ns; out[3] = e->max_ns;
  for (int i = 0; i < 4; i++) out[4 + i] = e->phase_ns[i];
  out[8] = e->n_cold; out[9] = e->cold_ns; out[10] = e->io_err;
  out[11] = e->cold_wait_ns; out[12] = e->n_hint; out[13] = e->n_pf_drop;
  out[14] = e->n_cold_hinted; out[15] = e->n_pf_done;
  for (int i = 0; i < 8; i++) { out[16 + i] = e->n_hint_rk[i]; out[24 + i] = e->n_used_rk[i]; }
  out[32] = e->n_cold_pairs;
}

// read-ahead hint blocks (after w2e_cold): hdr [n_keys][16] i32, ids
// [n_keys][stride] i32, pinned; starts the hint thread and n_pf prefetch
// threads
int w2e_hint(void* p, void* hdr, void* ids, int stride, int n_pf) {
  Engine* e = (Engine*)p;
  if (e->rowmap.empty() || e->hint_th.joinable() || stride > 128) return -1;
  e->hhdr = (volatile int32_t*)hdr;
  e->hids = (const volatile int32_t*)ids;
  e->hstride = stride;
  if (const char* v = getenv("VLLM_MOE_W2_COLD_HINT_TOP")) e->htop = std::max(1, atoi(v));
  if (const char* v = getenv("VLLM_MOE_W2_CPU_DYN")) e->dyn_sched = atoi(v) != 0;
  e->hint_t.assign((size_t)e->n_keys * e->n_experts, 0);
  e->hint_rk.assign((size_t)e->n_keys * e->n_experts, 0);
  e->hint_th = std::thread(&Engine::hint_loop, e);
  for (int i = 0; i < std::max(n_pf, 1); i++) e->pf_th.emplace_back(&Engine::pf_worker, e);
  return 0;
}

// cold rows for layer `key`: rowmap [n_experts] (compact host row, or -1 =
// read from the files; kept alive by the caller), fds[4] of its section
// files (codes13, sc13, codes2, sc2); n_io I/O threads (started once; 0 =
// the master reads, no overlap). Before the first job.
int w2e_cold(void* p, int key, const int32_t* rowmap, const int* fds, int drop,
             int n_io) {
  Engine* e = (Engine*)p;
  if (key < 0 || key >= e->n_keys) return -1;
  if (e->rowmap.empty()) {
    e->rowmap.assign(e->n_keys, nullptr);
    e->cfd.assign((size_t)e->n_keys * 4, -1);
  }
  e->rowmap[key] = rowmap;
  for (int s = 0; s < 4; s++) e->cfd[(size_t)key * 4 + s] = fds[s];
  e->cold_drop = drop != 0;
  while ((int)e->io_th.size() < n_io) e->io_th.emplace_back(&Engine::io_worker, e);
  return 0;
}

const char* w2e_isa() { return W2E_ISA; }

// vector codes: tab = u8 [P][8], int8 (w / cs) ^ 0x80. Before the first job.
// 0 = ok, -1 = this build has no VQ kernels, -2 = shape unsupported
int w2e_vq(void* p, const uint8_t* tab, int P, float cs) {
#ifdef W2E_VQ
  Engine* e = (Engine*)p;
  if (e->H > VQ_KMAX || e->I > VQ_KMAX || e->H % 256 || e->I % 256) return -2;
  e->vqtab = (uint8_t*)aligned_alloc(64, (size_t)P * 8);
  memcpy(e->vqtab, tab, (size_t)P * 8);
#ifdef W2E_VQ_SIGNED_TAB
  for (size_t i = 0; i < (size_t)P * 8; i++) e->vqtab[i] ^= 0x80;
#endif
  e->vqcs = cs;
  return 0;
#else
  (void)p; (void)tab; (void)P; (void)cs;
  return -1;
#endif
}

// test hook: y[t][N] = W x_t for nt tokens (f32 x [nt][K]) on one thread,
// the engine's activation format and kernels (vq: tab as for w2e_vq)
int w2e_gemv(const uint8_t* plane, const uint8_t* sc, int N, int K, const uint8_t* tab,
             float cs, const float* x, int nt, float* y) {
  if (nt < 1 || nt > TMAX || N % 16 || K % 64) return -2;
#ifndef W2E_VQ
  if (tab) return -1;
#endif
#ifdef W2E_VQ_SIGNED_TAB
  std::vector<uint8_t> stab;
  if (tab) {
    stab.assign(tab, tab + 512 * 8);   // a code holds a 9-bit pattern index
    for (auto& b : stab) b ^= 0x80;
    tab = stab.data();
  }
#endif
  std::vector<ActQ> a(nt);
  std::vector<std::vector<float>> o(nt, std::vector<float>(16));
  const ActQ* ap[TMAX];
  float* op[TMAX];
  for (int t = 0; t < nt; t++) {
    a[t].reserve(K);
    for (int kb = 0; kb < K / 64; kb++) {
#ifdef W2E_VQ
      if (tab) { prep_kb_vq(x + (size_t)t * K + kb * 64, a[t], kb); continue; }
#endif
      prep_kb(x + (size_t)t * K + kb * 64, a[t], kb);
    }
    ap[t] = &a[t];
    op[t] = o[t].data();
  }
  for (int nb = 0; nb < N / 16; nb++) {
#ifdef W2E_VQ
    if (tab) gemv_tokens_vq(plane, sc, K, nb, ap, op, nt, tab, cs);
    else
#endif
    gemv_tokens(plane, sc, K, nb, ap, op, nt);
    for (int t = 0; t < nt; t++)
      for (int r = 0; r < 16; r++) y[(size_t)t * N + nb * 16 + r] = o[t][r];
  }
  for (auto& q : a) { free(q.aq); free(q.ac); free(q.asv); }
  return 0;
}

void w2e_destroy(void* p) {
  Engine* e = (Engine*)p;
  e->stop = true;
  e->master.join();
  for (auto& t : e->th) t.join();
  { std::lock_guard<std::mutex> lk(e->io_mu); }
  e->io_cv.notify_all();
  for (auto& t : e->io_th) t.join();
  if (e->hint_th.joinable()) e->hint_th.join();
  { std::lock_guard<std::mutex> lk(e->pf_mu); }
  e->pf_cv.notify_all();
  for (auto& t : e->pf_th) t.join();
  free(e->bounce);
  delete e;
}

}  // extern "C"
