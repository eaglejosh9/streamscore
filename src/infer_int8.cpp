// Run the first N records of a feature dump through the int8 backend and
// write the raw logits, so they can be diffed against the float32 logits
// from infer_dump.
//
// The comparison is int8-vs-float32, two genuinely different numeric
// schemes -- so the expected agreement is much looser than the
// scalar-vs-PyTorch diff, which compared two implementations of one scheme.

#include "features.hpp"
#include "model.hpp"
#include "model_int8.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s <weights_int8.bin> <features.bin> <logits.bin> [n]\n",
                 argv[0]);
    return 1;
  }
  const int n_want = (argc > 4) ? std::atoi(argv[4]) : 1000;

  model::WeightsInt8 w;
  if (!w.load(argv[1])) return 1;
  model::Int8Backend backend(w);

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

  constexpr int BLOCK = 4096;
  std::vector<feat::Record> recs(BLOCK);
  std::vector<float> logits(BLOCK * model::N_OUT);

  int done = 0;
  float first[model::N_OUT] = {};
  bool have_first = false;
  int skipped = 0;

  while (done < n_want) {
    const size_t got = std::fread(recs.data(), sizeof(feat::Record), BLOCK, fin);
    if (got == 0) break;

    int kept = 0;
    for (size_t i = 0; i < got; ++i) {
        const uint64_t ts = recs[i].ts;
        if (ts < feat::MARKET_OPEN || ts >= feat::MARKET_CLOSE) {
        ++skipped;
        continue;
        }
        backend.forward(recs[i].f, &logits[kept * model::N_OUT]);
        if (!have_first) {
        for (int k = 0; k < model::N_OUT; ++k) first[k] = logits[kept * model::N_OUT + k];
        have_first = true;
        }
        ++kept;
        if (done + kept >= n_want) break;
    }

    std::fwrite(logits.data(), sizeof(float), kept * model::N_OUT, fout);
    done += kept;

    if (got < BLOCK) break;
}

  std::fclose(fin);
  std::fclose(fout);

  std::printf("scored %d records (int8) -> %s\n", done, argv[3]);
  std::printf("scored %d records, skipped %d outside market hours -> %s\n",
            done, skipped, argv[3]);
  if (have_first) {
    std::printf("first logits: %.6f %.6f %.6f\n", first[0], first[1], first[2]);
  }
  return 0;
}