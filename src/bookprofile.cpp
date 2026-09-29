// Profile the order book's access pattern before replacing it.
//
// Phase 7 showed 782,659 price levels held in scattered red-black-tree
// nodes, and phase 5 attributed ~858 ns/event to walking them. A flat
// sorted array would make the top five levels contiguous -- one cache
// line instead of ten pointer chases -- but a flat array pays O(n) on
// insert where a tree pays O(log n).
//
// Whether that trade is worth taking depends on three things this program
// measures and nothing so far has:
//
//   1. How deep are books, really? A flat array is obviously right at 10
//      levels and obviously wrong at 10,000. The answer is a distribution,
//      not an average -- AAPL and a micro-cap behave nothing alike.
//
//   2. WHERE do inserts land? An insert at the far end of a sorted array
//      shifts nothing. An insert at the front shifts everything. If most
//      inserts are near the touch, the shift cost is the whole story; if
//      they are deep in the book, it is close to free.
//
//   3. How often is a level touched vs created? An update to an existing
//      level is O(1) in both designs -- only insert and erase differ.
//
// Nothing here changes the book. It observes the existing std::map and
// reports what a flat array WOULD have cost.

#include "itch.hpp"
#include "feed.hpp"
#include "universe.hpp"
#include "features.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t CHUNK = 1 << 20;
constexpr size_t MAX_LOCATE = 16384;

std::string sym_to_string(const uint8_t* p) {
  std::string s(reinterpret_cast<const char*>(p), 8);
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

// A histogram over power-of-two buckets. Depth and shift distance both
// span several orders of magnitude, so linear buckets would be useless --
// almost everything would land in the first one.
struct LogHist {
  uint64_t bucket[40] = {};
  uint64_t count = 0;
  uint64_t sum = 0;
  uint64_t max = 0;

  void add(uint64_t v) {
    ++count;
    sum += v;
    if (v > max) max = v;
    int b = 0;
    while ((1ull << b) <= v && b < 39) ++b;
    bucket[b]++;      // bucket b holds values in [2^(b-1), 2^b)
  }

  void report(const char* name) const {
    if (!count) { std::printf("  %s: no samples\n", name); return; }
    std::printf("\n  %s  (n=%llu, mean=%.1f, max=%llu)\n",
                name, (unsigned long long)count,
                double(sum) / double(count), (unsigned long long)max);
    uint64_t running = 0;
    for (int b = 0; b < 40; ++b) {
      if (!bucket[b]) continue;
      running += bucket[b];
      const uint64_t lo = b ? (1ull << (b - 1)) : 0;
      const uint64_t hi = (1ull << b) - 1;
      std::printf("    %6llu-%-6llu  %10llu  %5.1f%%   (cum %5.1f%%)\n",
                  (unsigned long long)lo, (unsigned long long)hi,
                  (unsigned long long)bucket[b],
                  100.0 * double(bucket[b]) / double(count),
                  100.0 * double(running) / double(count));
    }
  }

  // Value at the given percentile, read off the cumulative histogram.
  // Bucket-resolution only -- good enough to decide a design.
  uint64_t pct(double p) const {
    if (!count) return 0;
    const uint64_t target = uint64_t(p * double(count));
    uint64_t running = 0;
    for (int b = 0; b < 40; ++b) {
      running += bucket[b];
      if (running >= target) return b ? (1ull << b) - 1 : 0;
    }
    return max;
  }
};

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <itch.bin>\n", argv[0]);
    return 1;
  }

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("itch"); return 1; }

  std::vector<uint8_t> buf(CHUNK);
  size_t valid = 0, pos = 0;
  auto refill = [&]() -> size_t {
    size_t leftover = valid - pos;
    std::memmove(buf.data(), buf.data() + pos, leftover);
    pos = 0;
    size_t got = std::fread(buf.data() + leftover, 1, CHUNK - leftover, f);
    valid = leftover + got;
    return got;
  };

  feed::Universe uni(MAX_LOCATE);
  feed::ItchAdapter adapter(uni);

  LogHist depth;         // levels on the touched side, sampled per update
  LogHist shift;         // elements a flat array would have to move
  LogHist rank_hist;     // how far from the touch the change landed

  uint64_t total = 0, updates = 0;
  uint64_t op_insert = 0, op_erase = 0, op_modify = 0;
  uint64_t insert_at_touch = 0, insert_top5 = 0;

  // Per-symbol peak depth, to find whether deep books are one outlier or
  // a whole class of symbol.
  std::vector<uint32_t> peak_depth(MAX_LOCATE, 0);

  for (;;) {
    if ((valid - pos) < 2) { if (refill() == 0) break; continue; }
    uint16_t len = itch::read_be16(&buf[pos]);
    if ((valid - pos) < size_t(2) + len) { if (refill() == 0) break; continue; }

    const uint8_t* m = &buf[pos + 2];
    const uint8_t  type = m[0];
    const uint16_t locate = itch::read_be16(m + 1);
    const uint64_t ts = itch::read_be48(m + 5);
    ++total;

    if (type == 'R') {
      if (uni.valid(locate)) uni.name(locate, sym_to_string(m + 11));
      pos += 2 + len; continue;
    }
    if (type == 'H') {
      if (uni.valid(locate)) uni[locate].state = char(m[19]);
      pos += 2 + len; continue;
    }
    if (!uni.valid(locate)) { pos += 2 + len; continue; }

    feed::LevelUpdate u1, u2;
    bool have1 = false, have2 = false;

    if (type == 'A' || type == 'F') {
      u1 = adapter.on_add(ts, locate, itch::read_be64(m + 11), char(m[19]),
                          itch::read_be32(m + 32), itch::read_be32(m + 20));
      have1 = true;
    } else if (type == 'E' || type == 'C' || type == 'X') {
      have1 = adapter.on_reduce(ts, itch::read_be64(m + 11),
                                itch::read_be32(m + 19), &u1);
    } else if (type == 'D') {
      have1 = adapter.on_delete(ts, itch::read_be64(m + 11), &u1);
    } else if (type == 'U') {
      uint16_t rl; char rs;
      if (adapter.replace_begin(ts, itch::read_be64(m + 11), &u1, &rl, &rs)) {
        uni[u1.locate].book.apply(u1);
        have1 = false;
        u2 = adapter.replace_finish(ts, rl, itch::read_be64(m + 19), rs,
                                    itch::read_be32(m + 31),
                                    itch::read_be32(m + 27));
        have2 = true;
      }
    } else {
      pos += 2 + len; continue;
    }

    // Measure each update BEFORE applying it: the question is what a flat
    // array would have had to do, which depends on the state it finds.
    auto measure = [&](const feed::LevelUpdate& u) {
      const feed::Book& bk = uni[u.locate].book;

      // Rank = how many levels sit between the touch and this price.
      // In a sorted array that is the index; everything at or after it
      // shifts on an insert or erase.
      size_t n = 0, rank = 0;
      bool exists = false;
      if (u.side == 'B') {
        n = bk.bids.size();
        for (auto it = bk.bids.begin(); it != bk.bids.end(); ++it, ++rank) {
          if (it->first == u.price) { exists = true; break; }
          if (it->first < u.price) break;   // sorted descending: past it
        }
      } else {
        n = bk.asks.size();
        for (auto it = bk.asks.begin(); it != bk.asks.end(); ++it, ++rank) {
          if (it->first == u.price) { exists = true; break; }
          if (it->first > u.price) break;   // sorted ascending: past it
        }
      }

      depth.add(n);
      rank_hist.add(rank);
      if (n > peak_depth[u.locate]) peak_depth[u.locate] = uint32_t(n);

      if (exists && u.qty != 0) {
        // Level already there, quantity changes. Same cost either way --
        // one store. This is the case a flat array does NOT pay for.
        ++op_modify;
      } else if (exists && u.qty == 0) {
        ++op_erase;
        shift.add(n > rank ? n - rank - 1 : 0);   // elements after it move up
      } else if (u.qty != 0) {
        ++op_insert;
        shift.add(n > rank ? n - rank : 0);       // elements at/after move down
        if (rank == 0) ++insert_at_touch;
        if (rank < 5)  ++insert_top5;
      }
      ++updates;
    };

    if (have1) { measure(u1); uni[u1.locate].book.apply(u1); }
    if (have2) { measure(u2); uni[u2.locate].book.apply(u2); }

    pos += 2 + len;
  }
  std::fclose(f);

  std::printf("messages:      %12llu\n", (unsigned long long)total);
  std::printf("level updates: %12llu\n", (unsigned long long)updates);
  std::printf("\noperation mix:\n");
  std::printf("  modify existing level  %12llu  %5.1f%%   (free in both designs)\n",
              (unsigned long long)op_modify, 100.0 * double(op_modify) / double(updates));
  std::printf("  insert new level       %12llu  %5.1f%%\n",
              (unsigned long long)op_insert, 100.0 * double(op_insert) / double(updates));
  std::printf("  erase level            %12llu  %5.1f%%\n",
              (unsigned long long)op_erase, 100.0 * double(op_erase) / double(updates));

  std::printf("\n  inserts at the touch (rank 0): %llu  (%.1f%% of inserts)\n",
              (unsigned long long)insert_at_touch,
              100.0 * double(insert_at_touch) / double(op_insert));
  std::printf("  inserts in the top 5 levels:   %llu  (%.1f%% of inserts)\n",
              (unsigned long long)insert_top5,
              100.0 * double(insert_top5) / double(op_insert));

  depth.report("book depth at update time (levels on the touched side)");
  rank_hist.report("rank of the changed level (0 = the touch)");
  shift.report("elements a flat array would memmove");

  std::printf("\n  depth p50=%llu p90=%llu p99=%llu p99.9=%llu max=%llu\n",
              (unsigned long long)depth.pct(0.50), (unsigned long long)depth.pct(0.90),
              (unsigned long long)depth.pct(0.99), (unsigned long long)depth.pct(0.999),
              (unsigned long long)depth.max);
  std::printf("  shift p50=%llu p90=%llu p99=%llu p99.9=%llu max=%llu\n",
              (unsigned long long)shift.pct(0.50), (unsigned long long)shift.pct(0.90),
              (unsigned long long)shift.pct(0.99), (unsigned long long)shift.pct(0.999),
              (unsigned long long)shift.max);

  // Deepest books by symbol: is depth one outlier or a whole class?
  std::vector<uint16_t> deep;
  for (size_t i = 0; i < MAX_LOCATE; ++i) if (peak_depth[i]) deep.push_back(uint16_t(i));
  std::sort(deep.begin(), deep.end(), [&](uint16_t a, uint16_t b) {
    return peak_depth[a] > peak_depth[b];
  });
  std::printf("\n  deepest books by peak one-side depth:\n");
  for (size_t i = 0; i < deep.size() && i < 12; ++i) {
    std::printf("    %-8s %u\n", uni.names[deep[i]].c_str(), peak_depth[deep[i]]);
  }
  size_t over64 = 0, over256 = 0;
  for (uint16_t l : deep) {
    if (peak_depth[l] > 64)  ++over64;
    if (peak_depth[l] > 256) ++over256;
  }
  std::printf("  %zu of %zu symbols ever exceeded 64 levels, %zu exceeded 256\n",
              over64, deep.size(), over256);
  return 0;
}