// Engram rows from the package files (engram_files.py): a host fetcher that
// serves the GPU's per-step row requests from mmap'd tables, so the tables
// live in the page cache / on NVMe instead of pinned RAM.
//
// Per layer a pinned block: hdr (i32: 0 request seq, 1 ack seq, 2 rows),
// req (i32 table rows), landing values (vb bytes/row) and scales (sb).
// The master thread polls every layer's request seq; a new one is split
// over the pool (rows in parallel, so cold pages fault concurrently), then
// acked. Out-of-range rows (padding, dead ids) land as zeros.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

struct Layer {
  const uint8_t* w = nullptr;   // values table (file mapping + data offset)
  const uint8_t* s = nullptr;   // scales table
  int64_t n_rows = 0;
  volatile int32_t* hdr = nullptr;
  const int32_t* req = nullptr;   // the hash ids are i32
  uint8_t* lv = nullptr;
  uint8_t* ls = nullptr;
  int32_t served = 0;
  void* maps[2] = {nullptr, nullptr};
  size_t map_len[2] = {0, 0};
};

struct Fetch {
  std::vector<Layer> L;
  size_t vb = 0, sb = 0;
  int n_thr = 1;
  // current job
  const Layer* cur = nullptr;
  int rows = 0;
  std::atomic<uint64_t> gen{0};
  std::atomic<int> done{0};
  std::atomic<bool> stop{false};
  std::vector<std::thread> th;
  std::thread master;
  std::atomic<uint64_t> n_req{0}, n_rows{0}, busy_ns{0}, max_ns{0};

  void part(int tid) {
    const Layer& l = *cur;
    long lo = (long)rows * tid / n_thr, hi = (long)rows * (tid + 1) / n_thr;
    for (long i = lo; i < hi; i++) {
      int64_t r = l.req[i];
      if (r < 0 || r >= l.n_rows) {
        memset(l.lv + i * vb, 0, vb);
        memset(l.ls + i * sb, 0, sb);
        continue;
      }
      memcpy(l.lv + i * vb, l.w + (size_t)r * vb, vb);
      memcpy(l.ls + i * sb, l.s + (size_t)r * sb, sb);
    }
  }

  void worker(int tid) {
    uint64_t seen = 0;
    long idle = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      uint64_t g = gen.load(std::memory_order_acquire);
      if (g == seen) {
        if (++idle > 20000) std::this_thread::sleep_for(std::chrono::microseconds(20));
        else _mm_pause();
        continue;
      }
      idle = 0;
      seen = g;
      part(tid);
      done.fetch_add(1, std::memory_order_acq_rel);
    }
  }

  void job(Layer& l, int32_t seq) {
    auto t0 = std::chrono::steady_clock::now();
    std::atomic_thread_fence(std::memory_order_acquire);
    cur = &l;
    rows = l.hdr[2];
    done.store(0, std::memory_order_relaxed);
    gen.fetch_add(1, std::memory_order_acq_rel);
    part(0);
    while (done.load(std::memory_order_acquire) < n_thr - 1) _mm_pause();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    l.hdr[1] = seq;
    l.served = seq;
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    n_req++;
    n_rows += rows;
    busy_ns += ns;
    if (ns > max_ns) max_ns = ns;
  }

  void master_loop() {
    long idle = 0;
    while (!stop.load(std::memory_order_relaxed)) {
      bool any = false;
      for (auto& l : L) {
        int32_t s = l.hdr[0];
        if (s != l.served) { job(l, s); any = true; }
      }
      if (any) { idle = 0; continue; }
      if (++idle > 2000000) std::this_thread::sleep_for(std::chrono::microseconds(20));
      else _mm_pause();
    }
  }
};

const uint8_t* map_file(const char* path, int64_t off, Layer& l, int k) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return nullptr;
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); return nullptr; }
  void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (p == MAP_FAILED) return nullptr;
  madvise(p, (size_t)st.st_size, MADV_RANDOM);   // rows are scattered: no readahead
  l.maps[k] = p;
  l.map_len[k] = (size_t)st.st_size;
  return (const uint8_t*)p + off;
}

}  // namespace

extern "C" {

// layer i: values file + data offset, scales file + data offset, rows; its
// pinned hdr / req / landing blocks. Returns null on a mapping failure.
void* ef_create(int n_layers, const char** wpath, const int64_t* woff,
                const char** spath, const int64_t* soff, const int64_t* n_rows,
                void** hdr, void** req, void** lv, void** ls, uint64_t vb,
                uint64_t sb, int n_thr) {
  Fetch* f = new Fetch();
  f->vb = vb; f->sb = sb;
  f->n_thr = n_thr < 1 ? 1 : n_thr;
  f->L.resize(n_layers);
  for (int i = 0; i < n_layers; i++) {
    Layer& l = f->L[i];
    l.w = map_file(wpath[i], woff[i], l, 0);
    l.s = map_file(spath[i], soff[i], l, 1);
    if (!l.w || !l.s) {
      for (auto& m : f->L)
        for (int k = 0; k < 2; k++)
          if (m.maps[k]) munmap(m.maps[k], m.map_len[k]);
      delete f;
      return nullptr;
    }
    l.n_rows = n_rows[i];
    l.hdr = (volatile int32_t*)hdr[i];
    l.req = (const int32_t*)req[i];
    l.lv = (uint8_t*)lv[i];
    l.ls = (uint8_t*)ls[i];
    l.served = l.hdr[0];
  }
  for (int t = 1; t < f->n_thr; t++) f->th.emplace_back(&Fetch::worker, f, t);
  f->master = std::thread(&Fetch::master_loop, f);
  return f;
}

void ef_stats(void* p, uint64_t* out) {
  Fetch* f = (Fetch*)p;
  out[0] = f->n_req; out[1] = f->n_rows; out[2] = f->busy_ns; out[3] = f->max_ns;
}

void ef_destroy(void* p) {
  Fetch* f = (Fetch*)p;
  f->stop = true;
  f->master.join();
  for (auto& t : f->th) t.join();
  for (auto& l : f->L)
    for (int k = 0; k < 2; k++)
      if (l.maps[k]) munmap(l.maps[k], l.map_len[k]);
  delete f;
}

}  // extern "C"
