// Latency comparison across the three backends.
//
// Batch timing, because macOS's clock ticks at ~42 ns and a single
// forward() is faster than that. Each sample is the mean over BATCH
// records, so the reported percentiles are percentiles OF BATCH MEANS --
// individual tails are hidden. Real per-event tails need a Linux host
// where CLOCK_MONOTONIC_RAW has ~1 ns resolution.

#include "features.hpp"
#include "model.hpp"
#include "model_int8.hpp"
#include "model_neon.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <time.h>
static inline uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
#else
#include <time.h>
static inline uint64_t now_ns() {
  timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
#endif

namespace {

constexpr int BATCH = 1000;

struct Stats {
  std::vector<uint32_t> samples;
  void add(uint32_t ns) { samples.push_back(ns); }

  void report(const char* name, double baseline_p50 = 0.0) {
    if (samples.empty()) { std::printf("%s: no samples\n", name); return; }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) { return samples[size_t(p * (samples.size() - 1))]; };
    double p50 = pct(0.50);
    std::printf("%-16s p50=%6.1f  p99=%6.1f  min=%6u  (n=%zu)",
                name, p50, double(pct(0.99)), samples.front(), samples.size());
    if (baseline_p50 > 0.0) std::printf("   %.2fx", baseline_p50 / p50);
    std::printf("\n");
  }
};

// One timing loop, shared by all three backends.
//
// Template rather than a virtual base class: a virtual call would add an
// indirect branch per record that the real serving path would not have,
// and at a few hundred nanoseconds per forward() that is not negligible.
// The template also lets the compiler inline forward() into this loop,
// which is what a real caller would get.
//
// `checksum` is an out-parameter, not a discarded local. If nothing
// observes the logits, the optimizer is entitled to delete the entire
// forward() call and you would benchmark an empty loop. Printing the
// checksum makes the work observable.
template <typename B>
Stats run(const B& backend, const std::vector<feat::Record>& recs,
          int reps, double* checksum) {
  Stats st;
  float logits[model::N_OUT];
  double sum = 0.0;

  for (int r = 0; r < reps; ++r) {
    size_t i = 0;
    while (i + BATCH <= recs.size()) {
      const uint64_t t0 = now_ns();
      for (int k = 0; k < BATCH; ++k) {
        backend.forward(recs[i + k].f, logits);
        sum += logits[0];          // observable, so the call cannot vanish
      }
      const uint64_t elapsed = now_ns() - t0;
      st.add(uint32_t(elapsed / BATCH));
      i += BATCH;
    }
  }

  *checksum = sum;
  return st;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <weights.bin> <weights_int8.bin> <features.bin> [n] [reps]\n",
                 argv[0]);
    return 1;
  }
  const int n_want = (argc > 4) ? std::atoi(argv[4]) : 100000;
  const int reps   = (argc > 5) ? std::atoi(argv[5]) : 20;

  model::Weights wf;
  if (!wf.load(argv[1])) return 1;
  model::WeightsInt8 wq;
  if (!wq.load(argv[2])) return 1;

  // Load records once, up front -- file I/O must not be inside the timed
  // region.
  std::vector<feat::Record> recs;
  {
    FILE* fin = std::fopen(argv[3], "rb");
    if (!fin) { std::perror("features"); return 1; }
    uint32_t hdr[2];
    std::fread(hdr, sizeof(uint32_t), 2, fin);
    if (hdr[0] != feat::DUMP_MAGIC || hdr[1] != feat::DUMP_VERSION) {
      std::fprintf(stderr, "features: bad magic/version\n");
      return 1;
    }
    feat::Record r;
    while ((int)recs.size() < n_want &&
           std::fread(&r, sizeof(feat::Record), 1, fin) == 1) {
      if (r.ts >= feat::MARKET_OPEN && r.ts < feat::MARKET_CLOSE) {
        recs.push_back(r);
      }
    }
    std::fclose(fin);
  }
  std::printf("loaded %zu market-hours records\n\n", recs.size());

  model::ScalarBackend scalar(wf);
  model::Int8Backend   int8b(wq);
  model::NeonBackend   neon(wq);

  // Warm up untimed: the first pass faults in pages and pulls weights into
  // cache, and timing that would measure memory rather than the kernel.
  double warm = 0.0;
  run(scalar, recs, 1, &warm);
  run(int8b,  recs, 1, &warm);
  run(neon,   recs, 1, &warm);

  double c1 = 0, c2 = 0, c3 = 0;
  Stats s_scalar = run(scalar, recs, reps, &c1);
  Stats s_int8   = run(int8b,  recs, reps, &c2);
  Stats s_neon   = run(neon,   recs, reps, &c3);

  std::sort(s_scalar.samples.begin(), s_scalar.samples.end());
  const double base = s_scalar.samples[s_scalar.samples.size() / 2];

  std::printf("ns per forward(), mean over %d-record batches:\n\n", BATCH);
  s_scalar.report("float32 scalar");
  s_int8.report("int8 scalar", base);
  s_neon.report("int8 NEON", base);

  std::printf("\nchecksums: %.3f %.3f %.3f\n", c1, c2, c3);
  std::printf("(float32 and int8 differ -- quantization. int8 and NEON must match.)\n");
  return 0;
}