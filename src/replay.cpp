#include "itch.hpp"
#include "features.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__APPLE__)
#include <time.h>
static inline uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
#else
#include <time.h>
static inline uint64_t now_ns() {
  timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return uint64_t(t.tv_sec) * 1000000000ull + t.tv_nsec;
}
#endif

namespace {
constexpr size_t CHUNK = 1 << 20;
static_assert(CHUNK > 64 * 1024, "chunk must exceed max message size");
constexpr uint16_t NO_LOCATE = 0xFFFF;

// A resting order. We need this because Delete/Cancel/Execute reference
// an order only by its reference number.
struct Order {
  uint32_t price;
  uint32_t shares;
  char     side;    // 'B' or 'S'
};

struct Book {
  // Price -> total resting shares at that price.
  // std::map is ordered, so begin() gives the extreme price on each side.
  // bids uses std::greater so begin() is the HIGHEST bid;
  // asks uses the default so begin() is the LOWEST ask.
  std::map<uint32_t, uint64_t, std::greater<uint32_t>> bids;
  std::map<uint32_t, uint64_t> asks;

  std::unordered_map<uint64_t, Order> orders;

  uint32_t unknown_ref = 0;
  uint32_t bad_reduce = 0;

  // TODO: add_order(ref, side, price, shares)
  void add_order(uint64_t ref, char side, uint32_t price, uint32_t shares) {
    orders[ref] = Order{price, shares, side};
    if (side != 'B' && side != 'S') {
        std::fprintf(stderr, "fatal: bad side byte 0x%02x for ref %llu\n",
                    (unsigned)side, (unsigned long long)ref);
        std::exit(1);
    }
    if (side == 'B') bids[price] += shares;
    else             asks[price] += shares; //add guard?
  }
  // TODO: reduce(ref, shares)   — shared by Cancel and Execute
  void reduce(uint64_t ref, uint32_t shares) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return; }
    Order& o = it->second;

    char     side  = o.side;      // copy before any erase
    uint32_t price = o.price;

    if (o.shares < shares) { bad_reduce++; return; }
    o.shares -= shares;
    if (o.shares == 0) orders.erase(it);

    if (side == 'B') {
        bids[price] -= shares;
        if (bids[price] == 0)
            bids.erase(price);
    }
    else {
        asks[price] -= shares;
        if (asks[price] == 0)
            asks.erase(price);
    }
  }
  // TODO: remove(ref)           — Delete
  void remove(uint64_t ref) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return; }
    uint32_t all = it->second.shares;
    reduce(ref, all);
  }
  // TODO: replace(old_ref, new_ref, price, shares)
  void replace (uint64_t old_ref, uint64_t new_ref, uint32_t price, uint32_t shares) {
    auto it = orders.find(old_ref);
    if (it == orders.end()) { unknown_ref++; return; }
    char side = it->second.side;
    remove(old_ref);
    add_order(new_ref, side, price, shares);
  }
  // TODO: best_bid() / best_ask() — return 0 if that side is empty
  uint32_t best_bid() const {
    if (bids.empty())
        return 0;
    return bids.begin()->first;
  }

  uint32_t best_ask() const {
    if (asks.empty())
        return 0;
    return asks.begin()->first;
  }

  // Returns false if the book is crossed (best bid >= best ask).
  bool consistent() const {
    if (bids.empty() || asks.empty()) return true;   // nothing to cross
    return best_bid() < best_ask();
  }
};

// Latency samples, pre-allocated. Recording into a growing vector would
// allocate in the hot path and pollute the measurement.
struct LatencyLog {
  std::vector<uint32_t> samples;
  explicit LatencyLog(size_t cap) { samples.reserve(cap); }
  inline void add(uint32_t ns) { if (samples.size() < samples.capacity()) samples.push_back(ns); }

  void report(const char* name) {
    if (samples.empty()) { std::printf("no latency samples\n"); return; }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) {
      size_t i = size_t(p * (samples.size() - 1));
      return samples[i];
    };
    std::printf("%s ns: p50=%u p99=%u p99.9=%u max=%u  (n=%zu)\n",
                name, pct(0.50), pct(0.99), pct(0.999), samples.back(), samples.size());
  }
};

std::string sym_to_string(const uint8_t* p) {
  std::string s(reinterpret_cast<const char*>(p), 8);
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

template <typename MapT>
void top_levels(const MapT& side, feat::Levels& out) {
  out.n = 0;
  for (auto it = side.begin();
       it != side.end() && out.n < feat::N_LEVELS; ++it, ++out.n) {
    out.px[out.n]  = it->first;
    out.qty[out.n] = it->second;
  }
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <itch-file> <symbol> <out.bin>\n", argv[0]);
    return 1;
  }
  const std::string want = argv[2];

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("fopen input"); return 1; }
  FILE* out = std::fopen(argv[3], "wb");
  if (!out) { std::perror("fopen output"); return 1; }

  // Dump header so a reader can verify the format.
  std::fwrite(&feat::DUMP_MAGIC, sizeof(uint32_t), 1, out);
  std::fwrite(&feat::DUMP_VERSION, sizeof(uint32_t), 1, out);

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

  Book book;
  feat::Engine engine;
  LatencyLog combined_lat(4u << 20);

  // Records are batched into this vector and flushed in blocks — one
  // fwrite per event would make the syscall dominate the measurement.
  std::vector<feat::Record> pending;
  pending.reserve(8192);
  auto flush = [&]() {
    if (pending.empty()) return;
    std::fwrite(pending.data(), sizeof(feat::Record), pending.size(), out);
    pending.clear();
  };

  uint16_t target = NO_LOCATE;
  uint64_t applied = 0, emitted = 0, skipped = 0;
  // Near the other counters:
  uint64_t batch_start = 0, batch_count = 0;

  for (;;) {
    if ((valid - pos) < 2) { if (refill() == 0) break; continue; }
    uint16_t len = itch::read_be16(&buf[pos]);
    if ((valid - pos) < size_t(2) + len) { if (refill() == 0) break; continue; }

    const uint8_t* m = &buf[pos + 2];
    const uint8_t  type = m[0];
    const uint16_t locate = itch::read_be16(m + 1);
    const uint64_t ts = itch::read_be48(m + 5);

    // 1. If type is 'R' and target is still NO_LOCATE, compare the symbol
    //    at m + 11 against `want`. On a match, record `locate` as target.
    //    (R messages all appear near the start of the file.)
    if (type == 'R' && target == NO_LOCATE) {
        if (sym_to_string(m + 11) == want) {
            target = locate;
            std::printf("locate for %s: %u\n", want.c_str(), locate);
        }
    }

    // 2. If target is NO_LOCATE, or locate != target, skip this message.
    if (target == NO_LOCATE || locate != target) {
        pos += 2 + len;
        continue;
    }

    // 3. Dispatch on type. Extract the fields per the table and call the
    //    matching Book method. Remember: A and F have identical layouts
    //    for the fields we care about; E, C, and X all reduce an order
    //    by a share count; D removes it entirely.
    if (type == 'A' || type == 'F') {
        uint64_t ref    = itch::read_be64(m + 11);
        char     side   = static_cast<char>(m[19]);
        uint32_t shares = itch::read_be32(m + 20);
        uint32_t price  = itch::read_be32(m + 32);
        book.add_order(ref, side, price, shares);
        applied++;
    }
    else if (type == 'E' || type == 'C' || type == 'X') {
        // ref at 11, shares at 19 for all three → book.reduce
        uint64_t ref    = itch::read_be64(m + 11);
        uint32_t shares = itch::read_be32(m + 19);
        if (type == 'E' || type == 'C') {
            auto it = book.orders.find(ref);
            if (it != book.orders.end()) {
                engine.on_trade(shares, it->second.side);
            }
        }
        book.reduce(ref, shares);
        applied++;
    }
    else if (type == 'D') {
        // ref at 11, nothing else → book.remove
        uint64_t ref    = itch::read_be64(m + 11);
        book.remove(ref);
        applied++;
    }
    else if (type == 'U') {
        // old ref at 11, new ref at 19, shares at 27, price at 31 → book.replace
        uint64_t old_ref    = itch::read_be64(m + 11);
        uint64_t new_ref    = itch::read_be64(m + 19);
        uint32_t shares    = itch::read_be32(m + 27);
        uint32_t price    = itch::read_be32(m + 31);
        book.replace(old_ref, new_ref, price, shares);
        applied++;
    }
    else {
        pos += 2 + len;
        continue;
    }

    // Both sides must have depth: mid is meaningless with one side missing,
    // and imbalance divides by (bid_qty + ask_qty).
    if (book.bids.empty() || book.asks.empty()) {
        skipped++;
        pos += 2 + len;
        continue;
    }

    feat::Levels bid;
    feat::Levels ask;
    if (batch_count == 0) batch_start = now_ns();
    top_levels(book.bids, bid);
    top_levels(book.asks, ask);
    feat::Record r = engine.compute(ts, bid, ask);
    if (++batch_count == 1000) {
        combined_lat.add(uint32_t((now_ns() - batch_start) / 1000));
        batch_count = 0;
    }

    pending.push_back(r);
    if (pending.size() == pending.capacity()) flush();
    emitted++;

    pos += 2 + len;
  }

  flush();
  std::fclose(out);
  std::fclose(f);

  std::printf("applied: %llu  emitted: %llu  skipped: %llu\n",
            (unsigned long long)applied, (unsigned long long)emitted,
            (unsigned long long)skipped);
  combined_lat.report("combined lat");
  return 0;
}