// Run the first N records of a feature dump through the scalar backend and
// write the raw logits, so the Python side can diff them against PyTorch.
//
// This exists once, to prove ScalarBackend is correct. After that the
// scalar backend is the oracle and later backends are diffed against it.

#include "features.hpp"
#include "model.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <weights.bin> <features.bin> <logits.bin> [n]\n",
                 argv[0]);
    return 1;
  }
  const int n_want = (argc > 4) ? std::atoi(argv[4]) : 1000;

  model::Weights w;
  if (!w.load(argv[1])) return 1;
  model::ScalarBackend backend(w);

  FILE* fin = std::fopen(argv[2], "rb");
  if (!fin) { std::perror("features"); return 1; }

  uint32_t hdr[2];
  if (std::fread(hdr, sizeof(uint32_t), 2, fin) != 2) {
    std::fprintf(stderr, "features: truncated header\n");
    std::fclose(fin);
    return 1;
  }
  if (hdr[0] != feat::DUMP_MAGIC || hdr[1] != feat::DUMP_VERSION) {
    std::fprintf(stderr,
                 "features: magic 0x%08x version %u (want 0x%08x / %u)\n",
                 hdr[0], hdr[1], feat::DUMP_MAGIC, feat::DUMP_VERSION);
    std::fclose(fin);
    return 1;
  }

  FILE* fout = std::fopen(argv[3], "wb");
  if (!fout) { std::perror("logits"); std::fclose(fin); return 1; }

  // Read records in blocks rather than one at a time -- the diff is not a
  // latency measurement, but a per-record fread would dominate the runtime
  // for no reason.
  constexpr int BLOCK = 4096;
  std::vector<feat::Record> recs(BLOCK);
  std::vector<float> logits(BLOCK * model::N_OUT);

  int done = 0;
  float first[model::N_OUT] = {};
  bool have_first = false;

  while (done < n_want) {
    const int want = (n_want - done < BLOCK) ? (n_want - done) : BLOCK;
    const size_t got = std::fread(recs.data(), sizeof(feat::Record), want, fin);
    if (got == 0) break;

    for (size_t i = 0; i < got; ++i) {
      backend.forward(recs[i].f, &logits[i * model::N_OUT]);
    }

    if (!have_first && got > 0) {
      for (int k = 0; k < model::N_OUT; ++k) first[k] = logits[k];
      have_first = true;
    }

    // Only the logits go to the file: the Python side reads it as a flat
    // (n, 3) float32 array with no header of its own.
    std::fwrite(logits.data(), sizeof(float), got * model::N_OUT, fout);
    done += static_cast<int>(got);

    if (got < static_cast<size_t>(want)) break;   // end of the dump
  }

  std::fclose(fin);
  std::fclose(fout);

  std::printf("scored %d records -> %s\n", done, argv[3]);
  if (have_first) {
    std::printf("first logits: %.6f %.6f %.6f\n", first[0], first[1], first[2]);
  }
  return 0;
}