// Multi-symbol replay: score the entire universe, not one ticker.
//
// The single-symbol version processed 1.5M events -- so little work that a
// single core handled it 650x over, which made the inference cost
// irrelevant and batching pointless. Across all ~8,900 symbols the same
// day is 268M events, the inference runs ~180x more often, and many
// symbols have pending work at any instant, which is what makes batching
// natural rather than contrived.
//
// This phase answers: how fast can one core score the whole universe,
// where does the time go, and what does holding 8,900 books cost.
//
// It deliberately does NOT persist feature records. At 268M events that
// dump would be ~21 GB and the run would become I/O-bound, as phase 1
// showed. replay.cpp still exists for producing training data.

#include "itch.hpp"
#include "feed.hpp"
#include "universe.hpp"
#include "features.hpp"
#include "model.hpp"
#include "model_int8.hpp"
#include "backend_select.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

constexpr size_t CHUNK = 1 << 20;
static_assert(CHUNK > 64 * 1024, "chunk must exceed max message size");

// ITCH locates are assigned from 1 and NASDAQ lists under 10k symbols, so
// this bound is generous. Checked at runtime rather than trusted.
constexpr size_t MAX_LOCATE = 16384;

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

struct Stats {
  std::vector<uint32_t> samples;
  explicit Stats(size_t cap) { samples.reserve(cap); }
  void add(uint32_t v) {
    if (samples.size() < samples.capacity()) samples.push_back(v);
  }
  void report(const char* name, const char* unit) {
    if (samples.empty()) { std::printf("  %-18s (no samples)\n", name); return; }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) { return samples[size_t(p * (samples.size() - 1))]; };
    std::printf("  %-18s p50=%-7u p99=%-7u p99.9=%-7u max=%-8u %s\n",
                name, pct(0.50), pct(0.99), pct(0.999), samples.back(), unit);
  }
};

void print_hhmmss(uint64_t ns) {
  const uint64_t s = ns / 1000000000ull;
  std::printf("%02llu:%02llu:%02llu",
              (unsigned long long)(s / 3600),
              (unsigned long long)((s / 60) % 60),
              (unsigned long long)(s % 60));
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <itch.bin> <weights_int8.bin>\n", argv[0]);
    return 1;
  }

  model::WeightsInt8 wq;
  if (!wq.load(argv[2])) return 1;
  model::FastBackend backend(wq);
  std::fprintf(stderr, "backend: %s\n", model::fast_backend_name());

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

  uint64_t total = 0, applied = 0, scored = 0, skipped = 0, crossings = 0;

  // Peak state, sampled periodically. Counting at the END of the session
  // always yields zero -- NASDAQ cancels every resting order after the
  // close -- so the number that matters is the maximum held during the
  // day. Measuring at the wrong moment made this look like a leak.
  size_t peak_orders = 0, peak_levels = 0;
  uint64_t peak_ts = 0;

  // Crossings attributed to the message type that produced them. A count
  // concentrated in one type points at a bug in that path; spread across
  // types and clustered before 09:30 points at the opening auction, where
  // books legitimately cross until the opening trade resolves them.
  uint64_t cross_by_type[256] = {};
  uint64_t first_cross_ts = 0, last_cross_ts = 0;
  uint64_t cross_in_hours = 0;

  // Batch timing, as in phase 5: a single event is below this platform's
  // clock resolution, so 1000 are measured together and the mean reported.
  Stats per_event(8u << 20);
  constexpr int BATCH = 1000;
  uint64_t batch_start = 0, batch_count = 0;

  double checksum = 0.0;
  const uint64_t wall_start = now_ns();

  for (;;) {
    if ((valid - pos) < 2) { if (refill() == 0) break; continue; }
    uint16_t len = itch::read_be16(&buf[pos]);
    if ((valid - pos) < size_t(2) + len) { if (refill() == 0) break; continue; }

    const uint8_t* m = &buf[pos + 2];
    const uint8_t  type = m[0];
    const uint16_t locate = itch::read_be16(m + 1);
    const uint64_t ts = itch::read_be48(m + 5);
    ++total;

    // Stock directory: build the locate -> ticker table. No symbol filter
    // any more -- the whole universe is in scope.
    if (type == 'R') {
      if (uni.valid(locate)) uni.name(locate, sym_to_string(m + 11));
      pos += 2 + len;
      continue;
    }
    if (type == 'H') {
        if (uni.valid(locate)) {
            uni[locate].state = char(m[19]);
            uni[locate].last_state_change = ts;
        }
        pos += 2 + len;
        continue;
    }

    if (!uni.valid(locate)) { adapter.bad_locate++; pos += 2 + len; continue; }

    if (batch_count == 0) batch_start = now_ns();

    feed::LevelUpdate u1, u2;
    bool have1 = false, have2 = false;

    if (type == 'A' || type == 'F') {
      uint64_t ref    = itch::read_be64(m + 11);
      char     side   = char(m[19]);
      uint32_t shares = itch::read_be32(m + 20);
      uint32_t price  = itch::read_be32(m + 32);
      u1 = adapter.on_add(ts, locate, ref, side, price, shares);
      have1 = true;
    } else if (type == 'E' || type == 'C' || type == 'X') {
      uint64_t ref    = itch::read_be64(m + 11);
      uint32_t shares = itch::read_be32(m + 19);
      if (type == 'E' || type == 'C') {
        auto it = adapter.orders.find(ref);
        if (it != adapter.orders.end()) {
          uni[it->second.locate].engine.on_trade(shares, it->second.side);
        }
      }
      have1 = adapter.on_reduce(ts, ref, shares, &u1);
    } else if (type == 'D') {
      have1 = adapter.on_delete(ts, itch::read_be64(m + 11), &u1);
    } else if (type == 'U') {
      uint64_t old_ref = itch::read_be64(m + 11);
      uint64_t new_ref = itch::read_be64(m + 19);
      uint32_t shares  = itch::read_be32(m + 27);
      uint32_t price   = itch::read_be32(m + 31);
      uint16_t rl; char rs;
      if (adapter.replace_begin(ts, old_ref, &u1, &rl, &rs)) {
        // Apply the removal BEFORE computing the add, so a same-price
        // replace reads a level that has already lost the old order.
        uni[u1.locate].book.apply(u1.side, u1.price, u1.qty);
        have1 = false;                    // already applied
        u2 = adapter.replace_finish(ts, rl, new_ref, rs, price, shares);
        have2 = true;
      }
    } else {
      pos += 2 + len;
      continue;
    }

    if (have1) uni[u1.locate].book.apply(u1.side, u1.price, u1.qty);
    if (have2) uni[u2.locate].book.apply(u2.side, u2.price, u2.qty);
    applied++;
    uni[locate].events++;

    if (!uni[locate].book.consistent()) {
      crossings++;
      cross_by_type[type]++;
      if (!first_cross_ts) first_cross_ts = ts;
      last_cross_ts = ts;
      if (ts >= feat::MARKET_OPEN && ts < feat::MARKET_CLOSE) cross_in_hours++;

      feed::SymbolState& cs = uni[locate];
      cs.crossings++;
      if (cs.state != 'T') {
        cs.cross_while_halted++;
      }
      else {
         if (ts - cs.last_state_change < 1000000) cs.cross_near_resume++;
      }
      const int64_t depth = int64_t(cs.book.best_bid()) - int64_t(cs.book.best_ask());
      if (depth > cs.worst_cross) cs.worst_cross = depth;
    }

    // Peak sampling, every ~1M messages. The level sweep touches 16k
    // books so it is not free; sampling keeps it well under 1% of runtime
    // while still catching the peak, which moves slowly over a session.
    if ((total & 0xFFFFF) == 0) {
      if (adapter.orders.size() > peak_orders) peak_orders = adapter.orders.size();
      size_t lv = 0;
      for (const auto& s : uni.states) lv += s.book.levels();
      if (lv > peak_levels) { peak_levels = lv; peak_ts = ts; }
    }

    feed::SymbolState& st = uni[locate];
    if (!st.book.two_sided() || st.state != 'T' ||
        ts < feat::MARKET_OPEN || ts >= feat::MARKET_CLOSE) {
      skipped++;
      // Reset the batch so a long run of skipped events does not leave a
      // stale start timestamp and inflate one sample.
      batch_count = 0;
      pos += 2 + len;
      continue;
    }

    feat::Levels bid, ask;
    bid.n = st.book.bids.top(feat::N_LEVELS, bid.px, bid.qty);
    ask.n = st.book.asks.top(feat::N_LEVELS, ask.px, ask.qty);

    feat::Record r = st.engine.compute(ts, bid, ask);
    float logits[model::N_OUT];
    backend.forward(r.f, logits);
    checksum += logits[0];

    scored++;
    st.emitted++;

    if (++batch_count == BATCH) {
      per_event.add(uint32_t((now_ns() - batch_start) / BATCH));
      batch_count = 0;
    }

    pos += 2 + len;
  }

  const uint64_t wall = now_ns() - wall_start;
  std::fclose(f);

  std::printf("\nmessages read:    %12llu\n", (unsigned long long)total);
  std::printf("book updates:     %12llu\n", (unsigned long long)applied);
  std::printf("scored:           %12llu\n", (unsigned long long)scored);
  std::printf("skipped:          %12llu\n", (unsigned long long)skipped);
  std::printf("symbols active:   %12zu\n", uni.active());
  std::printf("unknown ref:      %12llu\n", (unsigned long long)adapter.unknown_ref);
  std::printf("bad reduce:       %12llu\n", (unsigned long long)adapter.bad_reduce);
  std::printf("bad locate:       %12llu\n", (unsigned long long)adapter.bad_locate);

  const double secs = double(wall) / 1e9;
  std::printf("\nwall: %.2f s   throughput: %.2f M messages/sec, "
              "%.2f M scores/sec\n",
              secs, double(total) / secs / 1e6, double(scored) / secs / 1e6);
  per_event.report("per event", "ns (mean over 1000-event batches)");

  std::printf("\npeak orders resting: %12zu\n", peak_orders);
  std::printf("peak price levels:   %12zu  at ", peak_levels);
  print_hhmmss(peak_ts);
  std::printf("\n");
  // A std::map node is roughly 64 bytes: key, value, three pointers and a
  // colour bit, each separately allocated. This is the number the flat
  // book layout in phase 8 is trying to reduce.
  std::printf("  ~%.1f MB of book state at peak, in scattered "
              "red-black-tree nodes\n", double(peak_levels) * 64.0 / 1e6);

  std::printf("\ncrossings:        %12llu  (%llu inside market hours)\n",
              (unsigned long long)crossings, (unsigned long long)cross_in_hours);
  if (crossings) {
    std::printf("  by message type:");
    for (int t = 0; t < 256; ++t) {
      if (cross_by_type[t]) {
        std::printf("  '%c'=%llu", t, (unsigned long long)cross_by_type[t]);
      }
    }
    std::printf("\n  first at ");
    print_hhmmss(first_cross_ts);
    std::printf(", last at ");
    print_hhmmss(last_cross_ts);
    std::printf("\n");
  }

  if (crossings) {
    // Which symbols, and how deep. A handful of illiquid names with
    // one-tick inversions is market microstructure -- NASDAQ is one venue
    // among many and its book can legitimately lock or cross briefly.
    // Many symbols, or deep inversions, would mean a real bug.
    std::vector<uint16_t> order;
    for (size_t i = 0; i < uni.states.size(); ++i) {
      if (uni.states[i].crossings) order.push_back(uint16_t(i));
    }
    std::sort(order.begin(), order.end(), [&](uint16_t a, uint16_t b) {
      return uni.states[a].crossings > uni.states[b].crossings;
    });

    std::printf("\n  %zu symbols crossed at least once (of %zu active)\n",
                order.size(), uni.active());
    std::printf("  worst offenders:\n");
    for (size_t i = 0; i < order.size() && i < 10; ++i) {
      const auto& s = uni.states[order[i]];
      std::printf("    %-8s %6u crossings (%u while halted), worst %.4f, "
                "%llu events\n",
                uni.names[order[i]].c_str(), s.crossings, s.cross_while_halted,
                double(s.worst_cross) / 10000.0,
                (unsigned long long)s.events);
    }

    // after the loop
    uint64_t by_state[256] = {};
    for (const auto& s : uni.states) if (s.events) by_state[(unsigned char)s.state]++;
    std::printf("\nfinal trading state by symbol:");
    for (int c = 0; c < 256; ++c)
    if (by_state[c]) std::printf("  '%c'=%llu", c, (unsigned long long)by_state[c]);
    std::printf("\n");

    uint64_t halted_total = 0;
    for (uint16_t l : order) halted_total += uni.states[l].cross_while_halted;
    std::printf("  %llu of %llu crossings occurred while the symbol was "
                "halted or paused\n",
                (unsigned long long)halted_total,
                (unsigned long long)crossings);
    
    uint64_t resume_total = 0;
    for (uint16_t l : order) resume_total += uni.states[l].cross_near_resume;
    const uint64_t unexplained = crossings - halted_total;
    std::printf("  of the %llu that crossed while state=='T', %llu were "
                "within 1ms of a state change\n",
                (unsigned long long)unexplained,
                (unsigned long long)resume_total);

    size_t one_tick = 0;
    for (uint16_t l : order) if (uni.states[l].worst_cross <= 100) one_tick++;
    std::printf("  %zu of %zu crossing symbols never exceeded one cent\n",
                one_tick, order.size());
  }

  std::printf("\nchecksum: %.3f\n", checksum);
  return 0;
}