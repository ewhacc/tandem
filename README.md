# Tandem-vLLM 0.2.1

Run DeepSeek-V4.1-Flash on one or two workstation GPUs. The routed experts live in host RAM as
2-bit VQ codes, a GPU cache holds the most-used ones, and missed experts are computed by the CPU in parallel
with the GPU ("Tandem"). Output is exact for the package's weights: no expert is skipped or substituted.
Everything else is stock vLLM 0.31.0: the OpenAI-compatible server, continuous batching, prefix caching, and
DSpark speculative decoding.

Tandem is built on [vLLM](https://github.com/vllm-project/vllm) and on
[vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet), whose expert-offload engine it ports to vLLM 0.31 (see
[Built on](#built-on)).

Two package variants are served from the same download:

| variant | how | quality vs the original FP4 model |
|---|---|---|
| VQ only (default) | `FP4_TOP=0` | same-token KL 0.151, top-1 agreement 0.877 |
| VQ + FP4 hot set | `FP4_TOP=52`: the 52 most-routed experts per layer stay in original FP4 | KL 0.110, top-1 0.903; GPQA +3.4 points pooled (p = 0.006) |

## Hardware

Tandem is a serving engine (continuous batching), and the GPU expert cache is what sets its speed. It is
built for workstation and server-class machines:

| | recommended | practical minimum (estimate) | runs, slowly |
|---|---|---|---|
| GPU | 96 GB SM120 (RTX PRO 6000 Blackwell) | 48 GB, 2× 24 GB (main + helper GPU), or 2× 16 GB (tensor parallel) | 32 GB (RTX 5090: `BASE_GB=10`, no drafter); one 24 GB card (`BASE_GB=7`) |
| RAM | 256 GB+ (all experts and Engram pinned; engine RSS 233 GiB) | 128 GB + RAID0 NVMe | 128 GB (`RAM_GB=90 ENGRAM=files`) |
| CPU | AVX-512, 16+ cores, DDR5 (the CPU experts are memory-bandwidth-bound) | AVX2, 8+ cores | |
| speed, 1 / 4 / 32 streams | 65 / 182 / 414 tok/s (124 at 1 stream with DSpark) | | 16 / 20 / – tok/s |

Also needed: 300 GB of disk for the package (NVMe), Linux x86-64, an NVIDIA driver with the CUDA 13.x toolkit
(nvcc), Python 3.12 and g++.

Supported GPUs:

| architecture | cards | expert kernels | checked on |
|---|---|---|---|
| SM120 (Blackwell) | RTX 50xx, RTX PRO 6000 Blackwell | SM120 SASS (fastest) | RTX PRO 6000, RTX 5090 |
| SM90 (Hopper) | H100, H200 | portable (Triton / mma.sync) | H200 |
| SM89 (Ada) | RTX 40xx, RTX 6000 Ada, L40 | portable, e4m3 | RTX 6000 Ada |
| SM86 (Ampere) | RTX 30xx, RTX A6000 | portable, f16 | RTX A6000, also limited to 24 GB (3090/4090 memory) |

Every portable kernel is checked against a reference at boot. The KV cache is fp8 on every GPU. One 16 GB card does not fit: the non-expert weights
plus one double-buffered prefill layer need about 19–20 GiB. Two 16 GB cards fit in tensor parallel, where each GPU holds
half of both (`GPU=0,1 BASE_GB=3.5 KV_GB=1 CHUNK=1024`). A 24 GB card fits with `BASE_GB=7` (about 450 MiB to
spare) and runs best with a second card as a helper GPU (see [Two GPUs](#two-gpus)).
On a desktop CPU (two memory channels, fewer cores) the CPU's work on the missed experts sets the decode speed.

For one user on a 16–32 GB card with 64–128 GB of RAM, use the sister project
[tandem.cpp](https://github.com/ewhacc/tandem.cpp) (a llama.cpp patch set; DeepSeek-V4.1-Flash IQ2_XS at 33 tok/s on an
RTX 5090 + 128 GB, 14 tok/s on a 16 GB card + 64 GB). Tandem-vLLM is for many streams on a big card.

## Two GPUs

Two ways to use a second card:

| mode | how | best for |
|---|---|---|
| tensor parallel | `GPU=0,1` | a strong CPU (16+ cores), or two 16 GB cards. `BASE_GB` and `KV_GB` are per GPU. |
| helper GPU | `GPU=0 HELPER_GPU=1 HELPER_GB=20` | an 8-core CPU, or a 24 GB main card. The helper caches the hot experts the main GPU does not hold and computes them. No P2P or NVLink needed. |

`HELPER_GB` is the helper's expert cache: about 20 on a 24 GB card, 40 on a 48 GB card.

GSM8K-50, one stream (decode tok/s):

| GPUs | CPU | one GPU | tensor parallel | helper GPU |
|---|---|---|---|---|
| 2× RTX PRO 6000 96 GB (`BASE_GB=65.5`) | EPYC 9534, 64 cores | 75.5 | **93.1** | |
| 2× RTX A6000 48 GB (`BASE_GB=24`) | EPYC 7F32, 8 cores | 22.1 | 11.8 | **26.7** |
| 2× 24 GB (A6000 limited to 24 GB, `BASE_GB=7`) | EPYC 7F32, 8 cores | 16.8 | 12.5 | **21.7** |
| 2× 16 GB (H200 limited to 16 GB, `BASE_GB=3.5`) | Threadripper PRO 7965WX, 24 cores | – | **36.6** | |

With 8 cores, tensor parallel is slower than one GPU: each GPU gets half of the CPU. With a strong CPU the helper does
not pay: a third 16 GB card as helper for tensor parallel (`GPU=0,1 HELPER_GPU=2`) ran 32.8 against 36.6 without it.

## Install

```bash
git clone https://github.com/ewhacc/tandem && cd tandem
tools/install_tandem.sh --check          # prerequisites only
tools/install_tandem.sh                  # venv at ~/.venvs/tandem (stock vllm==0.31.0 wheel + Tandem files)
hf download ewhacc/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 --local-dir ~/models/v41-tandem
```

The installer writes the launcher `~/.venvs/tandem/bin/tandem-serve` and links it as `~/.local/bin/tandem-serve`,
so `tandem-serve` works from any folder once `~/.local/bin` is on your PATH (Ubuntu adds it at login when the folder
exists; otherwise `export PATH="$HOME/.local/bin:$PATH"`, or call the full path). It takes `KEY=value` settings
followed by any `vllm serve` options. A fresh install reproduces the reference same-token KL bit for bit.

## Quick start

```bash
# personal use, 96 GB card: one or two streams, DSpark k=3
tandem-serve MODEL=~/models/v41-tandem BASE_GB=65.5 SPEC_N=3

# serving, 96 GB card: up to 32 streams, no drafter
tandem-serve MODEL=~/models/v41-tandem BASE_GB=65.5

# 32 GB card + 128 GB RAM box (RTX 5090; up to 8 streams: STEP_T=8 is picked on cards under 48 GB;
# the drafter does not fit on 32 GB)
tandem-serve MODEL=~/models/v41-tandem BASE_GB=10 RAM_GB=90 ENGRAM=files

# same box, one user: a 1 GiB KV cache (enough for one 32k conversation) leaves room for a larger expert cache
tandem-serve MODEL=~/models/v41-tandem BASE_GB=12 KV_GB=1 STEP_T=2 RAM_GB=90 ENGRAM=files

# same, with image input (the vision encoder takes 1 GiB of the expert cache)
tandem-serve MODEL=~/models/v41-tandem BASE_GB=11 KV_GB=1 STEP_T=2 RAM_GB=90 ENGRAM=files IMAGES=1

# two 24 GB cards (RTX 3090 / 4090) + 128 GB RAM: main GPU 0, helper GPU 1
tandem-serve MODEL=~/models/v41-tandem BASE_GB=7 KV_GB=2 STEP_T=8 CHUNK=2048 RAM_GB=90 ENGRAM=files HELPER_GPU=1 HELPER_GB=20

# two 16 GB cards (RTX 5080 / 5070 Ti) + 256 GB RAM: tensor parallel
tandem-serve MODEL=~/models/v41-tandem GPU=0,1 BASE_GB=3.5 KV_GB=1 STEP_T=8 CHUNK=1024

# 48 GB card (RTX A6000 / 6000 Ada)
tandem-serve MODEL=~/models/v41-tandem BASE_GB=24 KV_GB=2 STEP_T=8
```

The server speaks the OpenAI API on port 8000 (served model name `deepseek-v4-flash`). Tool calling is on, and the
thinking text comes back in its own `reasoning` field, so chat front ends such as Open WebUI show it separately.
Image input is off by default; `IMAGES=1` turns it on (one image per prompt, OpenAI `image_url` content). The vision
encoder adds about 1 GiB of VRAM at boot and 1.5 GiB at peak with a large image, so give `BASE_GB` 1–2 GiB less.

The first boot builds FlashInfer and CPU kernels (several minutes). Later boots take about 2–4 minutes; most of
that is pinning the experts in RAM.

## Settings

| setting | meaning |
|---|---|
| `MODEL=` | package directory (required) |
| `BASE_GB=` | GPU cache for the VQ experts, in GiB. Give it what is left of the card after KV and the drafter. Larger is faster (BASE 65.5 vs 60: +6–9%). |
| `FP4_TOP=K` | keep the K most-routed experts per layer in FP4 (0–64). Better quality; about 10–16% slower at 1–4 streams, about half the per-stream speed at 16 streams (smaller VQ cache). Use it for quality-first, few-stream use. |
| `SPEC_N=k` | DSpark speculative decoding with the package's drafter, k draft tokens. Pays off at 1–2 streams. |
| `STEP_T=` | largest decode step in tokens (streams × (1+k)). Default 32 without a drafter, 8 with one. |
| `NUM_SEQS=` | concurrent streams; default `STEP_T/(1+k)` |
| `CHUNK=` | prefill chunk in tokens (default 4096). 8192 prefills a 94k prompt in 46 s instead of 70 s but needs `BASE_GB` ≤ 60 on a 96 GB card. A too-large CHUNK + BASE pair boots, then runs out of memory on the first long prompt: lower `BASE_GB` by 5. |
| `KV_GB=`, `MAX_LEN=` | KV cache in GiB (default 4, about 140k tokens shared by all streams) and the per-request context length (default 32768; tested to 131072 with KV_GB=8). A longer `MAX_LEN` costs no VRAM. |
| `RAM_GB=N` | pin only the N GiB of most-routed experts; read the rest from the package files |
| `ENGRAM=files` | read Engram rows from the files instead of pinning 97 GiB of tables |
| `THINKING=0/1`, `REASONING_EFFORT=low/high/max` | chat template defaults |
| `TOOLS=1/0` | tool calling and the separate `reasoning` field (DeepSeek-V4.1 parsers; default on) |
| `IMAGES=N` | image inputs per prompt (default 0: the vision encoder is not loaded). Costs 1–1.5 GiB of VRAM; lower `BASE_GB` to match (5090 one-user: 11 instead of 12). |
| `GPU=`, `PORT=`, `HOST=`, `ALLOW=cidr,...`, `PANEL=1` | device(s) (`GPU=0,1`: tensor parallel), network, client allowlist, status page at `/tandem/` |
| `HELPER_GPU=`, `HELPER_GB=` | helper GPU and its expert cache in GiB (default 20); see [Two GPUs](#two-gpus) |

## Measured (RTX PRO 6000 96 GB + EPYC 9534, DDR5-4800)

System decode tok/s, all streams active (512-token GSM8K answers, thinking high):

| shape | drafter | 1 stream | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 96 GB card (BASE 65.5) | off | 65 | 115 | 182 | 284 | 359 | 414 |
| 96 GB card | k=3 | 124 | 194 | | | | |

- At 8+ streams the drafter does not help. Serve without it. With the drafter and `STEP_T=32` the card fits only
  BASE 60.
- RTX 5090 32 GB + i5-14600KF, 128 GB DDR4-3200 (`BASE_GB=10 RAM_GB=90 ENGRAM=files`): 16 / 18 / 20 tok/s at
  1 / 2 / 4 streams; one user with `BASE_GB=12 KV_GB=1 STEP_T=2`: 17 tok/s; a 32k-token prompt waits about 2 minutes. There the CPU's work on the missed
  experts sets the speed. With `IMAGES=1` and `BASE_GB=11`: a 2400×2400 image is read in 3–7 s to the first token; peak
  VRAM 30.8 of 32 GB.

Other GPUs, GSM8K-50, one stream, thinking low (decode tok/s; the CPU matters as much as the card):

| GPU + CPU | settings | correct | tok/s |
|---|---|---|---|
| H200 141 GB + Threadripper PRO 7965WX (24 cores) | `BASE_GB=100` | 49/50 | 60.7 |
| H200 + Threadripper PRO 7965WX | `BASE_GB=24 KV_GB=2 STEP_T=8` | 50/50 | 55.3 |
| 2× RTX PRO 6000 96 GB + EPYC 9534 | `GPU=0,1 BASE_GB=65.5` | 50/50 | 93.1 |
| RTX PRO 6000 96 GB + EPYC 9534 | `BASE_GB=65.5` | 49/50 | 75.5 |
| RTX 6000 Ada 48 GB + EPYC 7F32 8 cores | `BASE_GB=24 KV_GB=2 STEP_T=8` | 50/50 | 25.5 |
| RTX A6000 48 GB + EPYC 7F32 8 cores | `BASE_GB=24 KV_GB=2 STEP_T=8` | 49/50 | 22.1 |
| RTX A6000 + A6000 helper + EPYC 7F32 | same + `HELPER_GPU=1 HELPER_GB=40` | 50/50 | 26.7 |

Quality (thinking low, greedy). Reference: the original FP4 model on stock vLLM 0.32 (4× H200).

| test | original FP4 (reference) | VQ + FP4 top 52 | VQ only |
|---|---|---|---|
| GSM8K-200 | 192 | 194 | 192 |
| GPQA-diamond 198 | 161 (81.3%) | 152 (76.8%) | 145 (73.2%) |
| passkey retrieval, 8k / 32k / 94k tokens × 3 depths | 9/9 | 9/9 | 9/9 |

The 2-bit experts cost GPQA accuracy (−8 points VQ only, −4.5 with the FP4 hot set; both significant against the
reference); easier tasks and long-context retrieval do not show it. Pick `FP4_TOP=52` when quality matters more than
many-stream speed.

Long prompts prefill at about 1350 tok/s (CHUNK 4096): a 94k-token prompt waits about 70 s for its first token
(46 s with CHUNK=8192 BASE_GB=60).

## Package format

The package's `config.json` carries a `"tandem"` block with `"format": 1`. An engine that does not know the
format refuses to boot with a message to upgrade. Engine and package versions are independent.

## Models

| model | status in 0.2.1 |
|---|---|
| DeepSeek-V4.1-Flash-1017, Tandem VQ2 package | supported (this release) |
| DeepSeek-V4-Flash | runs on the engine with plain 2-bit experts; not packaged or validated |
| Qwen3.8-Flash-Next and other MoE models | planned; each needs its own expert package and quality check |

The original FP4 checkpoint is not served directly: the CPU experts compute VQ / 2-bit codes, not FP4. For original
precision with a lot of host RAM (about 512 GB), vLLM-Moet's MoET hybrid runs the original checkpoint with FP4 CPU
experts (on our box: native quality, about 60 tok/s single stream, 455 GiB resident). A later Tandem release adds a
quality setting from VQ + FP4 top 52 up to all-FP4.

## Built on

- **[vLLM](https://github.com/vllm-project/vllm)** 0.31.0 (Apache-2.0): the serving engine. Tandem installs the
  stock wheel and replaces 65 of its files.
- **[vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)** (Apache-2.0): the expert-offload engine Tandem is a fork of: the
  2-bit expert planes and SM120 kernels, the GPU expert cache with exact miss replay, the FP4 tier and its
  confidence gate (available here, off by default for V4.1).
- **Tandem adds:** VQ expert codes, CPU expert execution in parallel with the GPU, the CPU/PCIe split of missed
  experts, DSpark drafter support with offloaded experts, the lite package format and its loader, huge-page pinned
  stores, the helper GPU, portable expert kernels for SM90/SM89/SM86, and the port to vLLM 0.31 / DeepSeek-V4.1
  (SM120 attention fixes; split-KV sparse-MLA below SM90).

## Changes in 0.2.1

- Two 16 GB cards in tensor parallel (the pool floor counts each GPU's own experts).
- `HELPER_GPU` also works with `GPU=0,1`: each GPU uses the helper for its own experts.
- Fixed: `HELPER_GPU` without `RAM_GB` (all experts pinned) turned the helper and the CPU experts off; a helper that
  fails to start now leaves the CPU experts on.

## Changes in 0.2.0

- GPUs: Hopper (SM90), Ada (SM89) and Ampere (SM86) on portable expert kernels with a boot parity check; V4.1
  sparse-MLA attention below SM90 through a Triton port with split-KV decode (+17% on an A6000).
- Two GPUs: tensor parallel (`GPU=0,1`) and the helper GPU (`HELPER_GPU=`).
- Installer: checks the GPU architecture; uses `uv` when Python 3.12 or its venv module is missing.

## Changes in 0.1.1

- Image input: `IMAGES=1` (vision encoder of the package; checked on reading text and tables from images).
- Tool calling and the separate `reasoning` field on by default (`TOOLS=0` turns them off).
- `MAX_LEN` default 32768 (was 8192); `STEP_T` 8 by default on cards under 48 GB.
- CPU expert engine: no longer keeps cores busy while the server is idle; the work split adapts to mixed P/E-core
  CPUs (+6% on an i5-14600KF), with unchanged outputs.
- The installer links `tandem-serve` into `~/.local/bin`.
- One-user lines for 32 GB cards in the quick start.

## Not in 0.2.1
- AMD GPUs, V100 (SM70), and more than two GPUs as a tested setup.
- Models other than DeepSeek-V4.1-Flash (see [Models](#models)).
- Original-precision experts on the CPU (see [Models](#models)).
