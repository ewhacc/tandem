"""Build the Tandem lite package of DeepSeek-V4.1-Flash from the official checkpoint.

  fp4hot  ranked FP4 hot set: per layer the top-N experts by calibration routing count, as the
          engine's [fp4_13|sc13|fp4_2|sc2] slot rows (VLLM_MOE_W2_FP4_HOT): layer<L>.idx (int32
          expert ids, rank order), layer<L>.fp4.bin, ranking.tsv, top<K>.tsv static lists.
  engram  Engram tables as packed e2m1 nibbles + UE8M0 per 32 (config "engram_embed_format":
          "e2m1"): engram-e2m1-<L>.weight.safetensors / .scale.safetensors per layer.
  hotlist  your own FP4 hot list: rank the experts by routing counts from your prompts (engine
          profile: serve with VLLM_MOE_W2_NEED_PROF=<prefix> VLLM_MOE_W2_DELTA_TRACE=1, run your
          workload, stop; it writes <prefix>.base.rank0of1.pt), optionally blended with the shipped
          calibration ranking (--mix), restricted to the experts in the hot-set store; write a
          static list for VLLM_MOE_W2_DELTA_STATIC. Experts outside the store need the official
          checkpoint: rebuild the store with `fp4hot --rank <profile .pt>` instead.
  base    every tensor except the routed experts of layers 0-39 (and, with --engram e2m1, the
          FP8 Engram tables) in <= --shard-gb safetensors shards, plus the index, config,
          tokenizer and the small files. The VQ planes and table go in vq/ (link them with
          `cp -al`: they are already in the engine's planes-cache layout).

  python tools/e2e/tandem_package.py fp4hot --model /models/DeepSeek-V4.1-Flash-1017 \\
      --rank ~/moe-cache/static/prune-rank-512sb.pt --top 64 --lists 16,27,52,64 --out PKG/fp4hot
  python tools/e2e/tandem_package.py engram --model ... --out PKG
  python tools/e2e/tandem_package.py base --model ... --engram e2m1 --out PKG
"""
import argparse
import json
import os
import re
import shutil
import struct
import time

import torch
from safetensors import safe_open

EXPERT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.")
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.embed\.(weight|scale)$")
ST_SIZE = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I16": 2, "U16": 2,
           "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
CHUNK = 1 << 30


def engine_module(name):
    """From the installed engine (the moet31 venv links its moe_w2 / engram files to the worktree)."""
    import importlib
    return importlib.import_module(name)


class Index:
    def __init__(self, model):
        self.model = model
        self.map = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
        self._open = {}

    def f(self, name):
        fn = self.map[name]
        if fn not in self._open:
            self._open[fn] = safe_open(os.path.join(self.model, fn), "pt")
        return self._open[fn]

    def slice(self, name):
        return self.f(name).get_slice(name)

    def tensor(self, name):
        return self.f(name).get_tensor(name)


class Writer:
    """Streaming safetensors writer: declare all tensors, then write their bytes in order."""

    def __init__(self, path, specs, metadata=None):
        self.path, self.specs = path, specs           # [(name, dtype str, shape)]
        hdr, off = {}, 0
        for name, dt, shape in specs:
            n = ST_SIZE[dt]
            for d in shape:
                n *= d
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
            off += n
        if metadata:
            hdr["__metadata__"] = metadata
        h = json.dumps(hdr, separators=(",", ":")).encode()
        h += b" " * (-len(h) % 8)
        self.f = open(path + ".tmp", "wb")
        self.f.write(struct.pack("<Q", len(h)) + h)
        self.total = off
        self.written = 0

    def write(self, t):
        b = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes() if isinstance(t, torch.Tensor) else t
        self.f.write(b)
        self.written += len(b)

    def close(self):
        assert self.written == self.total, (self.path, self.written, self.total)
        self.f.close()
        os.replace(self.path + ".tmp", self.path)


def copy_tensor(ix, name, w):
    """Stream one checkpoint tensor's raw bytes (row chunks for the big ones)."""
    sl = ix.slice(name)
    shape = sl.get_shape()
    row = ST_SIZE[sl.get_dtype()]
    for d in shape[1:]:
        row *= d
    step = max(1, CHUNK // max(row, 1))
    if not shape or shape[0] * row <= CHUNK:
        w.write(ix.tensor(name))
        return
    for r0 in range(0, shape[0], step):
        w.write(sl[r0:min(r0 + step, shape[0])])


# ---------------------------------------------------------------- fp4hot
def cmd_fp4hot(a):
    P = engine_module("vllm.model_executor.layers.quantization.utils.moe_w2_planes")
    ix = Index(a.model)
    r = torch.load(os.path.expanduser(a.rank), map_location="cpu", weights_only=False)
    cnt, wsum = r["cnt"], r["wsum"]
    L, E = cnt.shape
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    order = []
    for li in range(L):
        # routing count, then routing weight as the tie-break
        key = cnt[li].double() * 1e6 + wsum[li].double() / max(float(wsum[li].max()), 1e-30)
        order.append(torch.argsort(key, descending=True).tolist())
    with open(os.path.join(a.out, "ranking.tsv"), "w") as f:
        f.write(f"# layer\trank\texpert\tcount (calibration {a.rank}, {r.get('ntok', '?')} tokens)\n")
        for li in range(L):
            for k, e in enumerate(order[li]):
                f.write(f"{li}\t{k}\t{e}\t{int(cnt[li, e])}\n")
    for K in [int(x) for x in a.lists.split(",") if x]:
        assert K <= a.top, f"list top{K} > stored top {a.top}"
        with open(os.path.join(a.out, f"top{K}.tsv"), "w") as f:
            f.write(f"# top {K} experts per layer by calibration routing count (ranking.tsv)\n")
            for li in range(L):
                for e in sorted(order[li][:K]):
                    f.write(f"{li}\t{e}\n")
    for li in ([int(x) for x in a.layers.split(",")] if a.layers else range(L)):
        t0 = time.time()
        ids = order[li][:a.top]
        path = os.path.join(a.out, f"layer{li}.fp4.bin")
        with open(path + ".tmp", "wb") as fo:
            for e in ids:
                p = f"layers.{li}.ffn.experts.{e}."
                w13 = torch.cat([ix.tensor(p + "w1.weight"), ix.tensor(p + "w3.weight")]).view(torch.uint8).to(dev)
                s13 = torch.cat([ix.tensor(p + "w1.scale"), ix.tensor(p + "w3.scale")]).view(torch.uint8).to(dev)
                w2 = ix.tensor(p + "w2.weight").view(torch.uint8).to(dev)
                s2 = ix.tensor(p + "w2.scale").view(torch.uint8).to(dev)
                for part in (P.pack_fp4_fragment_major(P.mxfp4_to_nibbles(w13)), P.pack_scales(s13),
                             P.pack_fp4_fragment_major(P.mxfp4_to_nibbles(w2)), P.pack_scales(s2)):
                    fo.write(part.contiguous().view(-1).cpu().numpy().tobytes())
        os.replace(path + ".tmp", path)
        import numpy as np
        np.asarray(ids, dtype=np.int32).tofile(os.path.join(a.out, f"layer{li}.idx"))
        print(f"layer {li}: top {len(ids)} -> {os.path.getsize(path) / 2**30:.2f} GiB, {time.time() - t0:.0f}s",
              flush=True)
    print("FP4HOT-DONE", flush=True)


# ---------------------------------------------------------------- hotlist
def load_counts(path):
    r = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
    c = r["cnt"] if "cnt" in r else r["freq"]          # calibration .pt | engine NEED_PROF .pt
    return c.double()


def cmd_hotlist(a):
    import numpy as np
    user = sum(load_counts(p) for p in a.prof.split(","))
    # the store's layers (an engine profile can carry extra rows, e.g. layer 40)
    L = len([f for f in os.listdir(a.store) if re.fullmatch(r"layer\d+\.idx", f)])
    user, E = user[:L], user.shape[1]
    calib = torch.zeros(L, E, dtype=torch.float64)
    with open(os.path.join(a.store, "ranking.tsv")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            li, _, e, c = (int(x) for x in line.split("\t"))
            calib[li, e] = c
    norm = lambda t: t / t.sum(1, keepdim=True).clamp_min(1e-30)
    score = (1 - a.mix) * norm(user) + a.mix * norm(calib)
    n_out = 0
    with open(a.out, "w") as f:
        f.write(f"# top {a.top} experts per layer: {a.prof} (mix {a.mix} with {a.store}/ranking.tsv), "
                f"from the hot-set store\n")
        for li in range(L):
            stored = np.fromfile(os.path.join(a.store, f"layer{li}.idx"), dtype=np.int32).tolist()
            assert a.top <= len(stored), f"--top {a.top} > {len(stored)} experts in the store"
            pick = sorted(stored, key=lambda e: -float(score[li, e]))[:a.top]
            for e in sorted(pick):
                f.write(f"{li}\t{e}\n")
            # how much of this layer's routing the store could not offer
            best = torch.topk(score[li], a.top).indices.tolist()
            n_out += len(set(best) - set(stored))
            cov = float(norm(user)[li, pick].sum())
            if a.verbose:
                print(f"layer {li}: user routing share in the list {cov:.3f}", flush=True)
    print(f"wrote {a.out}: {L} layers x {a.top}; {n_out} of the {L * a.top} best experts are outside "
          f"the store (use fp4hot --rank with the official checkpoint to include them)", flush=True)


# ---------------------------------------------------------------- engram
def cmd_engram(a):
    Q = engine_module("vllm.models.deepseek_v41.common.engram_e2m1")
    ix = Index(a.model)
    dev = torch.device("cuda")
    layers = sorted({int(m.group(1)) for n in ix.map if (m := ENGRAM_RE.match(n))})
    os.makedirs(a.out, exist_ok=True)
    for li in layers:
        t0 = time.time()
        wn, sn = f"layers.{li}.engram.embed.weight", f"layers.{li}.engram.embed.scale"
        R, D = ix.slice(wn).get_shape()
        pw = os.path.join(a.out, f"engram-e2m1-{li}.weight.safetensors")
        ps = os.path.join(a.out, f"engram-e2m1-{li}.scale.safetensors")
        ww = Writer(pw, [(wn, "U8", (R, D // 2))], {"format": "pt", "engram_embed_format": "e2m1"})
        ws = Writer(ps, [(sn, "U8", (R, D // 32))], {"format": "pt", "engram_embed_format": "e2m1"})
        err = ref = 0.0
        step = 1 << 21
        swl, ssl = ix.slice(wn), ix.slice(sn)
        for r0 in range(0, R, step):
            r1 = min(r0 + step, R)
            w = swl[r0:r1].to(dev)
            s = ssl[r0:r1].view(torch.uint8).to(dev)
            pk, s2, e, rf = Q.requant_packed(w, s)
            ww.write(pk.cpu())
            ws.write(s2.cpu())
            err += e
            ref += rf
        ww.close()
        ws.close()
        print(f"engram layer {li}: {R} x {D} -> {(os.path.getsize(pw) + os.path.getsize(ps)) / 2**30:.2f} GiB, "
              f"rel RMS {(err / ref) ** 0.5:.4f}, {time.time() - t0:.0f}s", flush=True)
    print("ENGRAM-DONE", flush=True)


# ---------------------------------------------------------------- base
def cmd_base(a):
    ix = Index(a.model)
    os.makedirs(a.out, exist_ok=True)
    keep = []
    for n in ix.map:
        if EXPERT_RE.match(n) and int(n.split(".")[1]) < a.vq_layers:
            continue
        if a.engram == "e2m1" and ENGRAM_RE.match(n):
            continue
        keep.append(n)
    keep.sort(key=lambda n: (ix.map[n], n))          # read each source shard once, in order
    shards, cur, size, lim = [], [], 0, int(a.shard_gb * 2**30)
    for n in keep:
        sl = ix.slice(n)
        nb = ST_SIZE[sl.get_dtype()]
        for d in sl.get_shape():
            nb *= d
        if cur and size + nb > lim:
            shards.append(cur)
            cur, size = [], 0
        cur.append((n, sl.get_dtype(), tuple(sl.get_shape())))
        size += nb
    if cur:
        shards.append(cur)
    wmap = {}
    for i, specs in enumerate(shards):
        fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        t0 = time.time()
        w = Writer(os.path.join(a.out, fn), specs, {"format": "pt"})
        for n, _, _ in specs:
            copy_tensor(ix, n, w)
            wmap[n] = fn
        w.close()
        print(f"{fn}: {len(specs)} tensors, {w.total / 2**30:.2f} GiB, {time.time() - t0:.0f}s", flush=True)
    if a.engram == "e2m1":
        for n in ix.map:
            m = ENGRAM_RE.match(n)
            if m:
                wmap[n] = f"engram-e2m1-{m.group(1)}.{m.group(2)}.safetensors"
    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in set(wmap.values())
                if os.path.exists(os.path.join(a.out, f)))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(wmap.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    if a.engram == "e2m1":
        cfg.get("text_config", cfg)["engram_embed_format"] = "e2m1"   # next to the engram_* fields
    cfg["tandem"] = {"format": 1, "experts": "vq", "vq_layers": a.vq_layers, "planes": "vq", "fp4_hot": "fp4hot",
                     "ckpt_id": a.ckpt_id, "engram_embed_format": a.engram}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.model):
        src = os.path.join(a.model, f)
        if f.endswith(".safetensors") or f in ("model.safetensors.index.json", "config.json"):
            continue
        dst = os.path.join(a.out, f)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print("BASE-DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fp4hot")
    p.add_argument("--model", required=True)
    p.add_argument("--rank", required=True, help="calibration routing counts (prune-rank .pt: cnt, wsum)")
    p.add_argument("--top", type=int, default=64)
    p.add_argument("--lists", default="16,27,52,64")
    p.add_argument("--layers", default="", help="subset, e.g. 0,1 (default: all)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("hotlist")
    p.add_argument("--prof", required=True, help="engine profile(s) .pt (freq) or calibration .pt (cnt), comma list")
    p.add_argument("--store", required=True, help="the package's fp4hot dir")
    p.add_argument("--top", type=int, required=True)
    p.add_argument("--mix", type=float, default=0.25, help="weight of the shipped calibration ranking")
    p.add_argument("--out", required=True)
    p.add_argument("--verbose", action="store_true")
    p = sub.add_parser("engram")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("base")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--engram", choices=("fp8", "e2m1"), default="e2m1")
    p.add_argument("--vq-layers", type=int, default=40, help="layers whose routed experts are VQ")
    p.add_argument("--ckpt-id", default="v41-1017-g51")
    p.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()
    {"fp4hot": cmd_fp4hot, "hotlist": cmd_hotlist, "engram": cmd_engram, "base": cmd_base}[a.cmd](a)


if __name__ == "__main__":
    main()
