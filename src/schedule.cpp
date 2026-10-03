// Scheduler simulation: bounded queue, deadline-aware admission control,
// deterministic drop accounting.
//
// The pipeline splits into mandatory and droppable work:
//
//   parse + book update   mandatory. The book is a running reconstruction
//                         from deltas -- skip one message and it is wrong
//                         for the rest of the day. Cannot be shed.
//
//   features + inference  droppable. Per-event and stateless, so shedding
//                         one costs a prediction and nothing more.
//
// Only optional work can be shed. Same reason a video decoder can drop a
// P-frame but not a keyframe.
//
// Single-threaded with a virtual clock. The inference genuinely runs on
// every admitted event -- cache behaviour and predictions are real -- but
// the clock advances by a benchmarked constant rather than measured
// elapsed time, because a single forward pass is below this platform's
// clock resolution (~42 ns tick on Apple Silicon) and per-event timing
// noise would accumulate across a million events.
//
// A threaded SPSC version is phase 6b, with this as its oracle.

#include "itch.hpp"
#include "features.hpp"
#include "model.hpp"
#include "model_int8.hpp"
#include "backend_select.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr size_t CHUNK = 1 << 20;
static_assert(CHUNK > 64 * 1024, "chunk must exceed max message size");
constexpr uint16_t NO_LOCATE = 0xFFFF;

// Per-event cost of the droppable work, from the phase 3 and 5 benchmarks:
//   book walk   858 ns   (two std::map walks, docs/03-features.md)
//   features    123 ns   (docs/03-features.md)
//   inference    49 ns   (int8 NEON, docs/05-backends.md)
// Overridable from the command line so the sweep can explore sensitivity.
constexpr uint64_t DEFAULT_WORK_NS = 858 + 123 + 49;

// ---------------------------------------------------------------------------
// Order book (unchanged from replay.cpp)
// ---------------------------------------------------------------------------

struct Order {
  uint32_t price;
  uint32_t shares;
  char     side;
};

struct Book {
  std::map<uint32_t, uint64_t, std::greater<uint32_t>> bids;
  std::map<uint32_t, uint64_t> asks;
  std::unordered_map<uint64_t, Order> orders;

  uint32_t unknown_ref = 0;
  uint32_t bad_reduce = 0;

  void add_order(uint64_t ref, char side, uint32_t price, uint32_t shares) {
    if (side != 'B' && side != 'S') {
      std::fprintf(stderr, "fatal: bad side byte 0x%02x for ref %llu\n",
                   (unsigned)side, (unsigned long long)ref);
      std::exit(1);
    }
    orders[ref] = Order{price, shares, side};
    if (side == 'B') bids[price] += shares;
    else             asks[price] += shares;
  }

  void reduce(uint64_t ref, uint32_t shares) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return; }
    Order& o = it->second;
    if (o.shares < shares) { bad_reduce++; return; }

    char     side  = o.side;
    uint32_t price = o.price;

    o.shares -= shares;
    if (o.shares == 0) orders.erase(it);

    if (side == 'B') {
      bids[price] -= shares;
      if (bids[price] == 0) bids.erase(price);
    } else {
      asks[price] -= shares;
      if (asks[price] == 0) asks.erase(price);
    }
  }

  void remove(uint64_t ref) {
    auto it = orders.find(ref);
    if (it == orders.end()) { unknown_ref++; return; }
    uint32_t all = it->second.shares;
    reduce(ref, all);
  }

  void replace(uint64_t old_ref, uint64_t new_ref, uint32_t price,
               uint32_t shares) {
    auto it = orders.find(old_ref);
    if (it == orders.end()) { unknown_ref++; return; }
    char side = it->second.side;
    remove(old_ref);
    add_order(new_ref, side, price, shares);
  }
};

template <typename MapT>
void top_levels(const MapT& side, feat::Levels& out) {
  out.n = 0;
  for (auto it = side.begin();
       it != side.end() && out.n < feat::N_LEVELS; ++it, ++out.n) {
    out.px[out.n]  = it->first;
    out.qty[out.n] = it->second;
  }
}

std::string sym_to_string(const uint8_t* p) {
  std::string s(reinterpret_cast<const char*>(p), 8);
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

// One queued scoring request.
//
// Carries a COPY of the book levels it needs, not a pointer into the book.
// The producer keeps applying messages while this waits in the queue, so a
// pointer would see a book from a later instant than the request describes
// -- the queued work would silently score the wrong state.
struct Request {
  uint64_t seq;        // monotonic, for ordering and debugging
  uint64_t arrive_ns;  // virtual arrival time
  uint64_t ts;         // ITCH timestamp, carried into the feature record
  feat::Levels bid, ask;
};

// Fixed-capacity ring buffer. Capacity is a power of two so the wrap is a
// mask rather than a modulo.
//
// head and tail are free-running counters, never wrapped themselves --
// size() is just their difference, with no ambiguity between full and
// empty that a wrapped-pointer scheme would have.
struct RingQueue {
  std::vector<Request> buf;
  size_t mask;
  uint64_t head = 0;   // next slot to read
  uint64_t tail = 0;   // next slot to write

  explicit RingQueue(size_t capacity_pow2)
      : buf(capacity_pow2), mask(capacity_pow2 - 1) {}

  size_t size()  const { return size_t(tail - head); }
  bool   empty() const { return head == tail; }
  bool   full()  const { return size() == buf.size(); }

  bool push(const Request& r) {
    if (full()) return false;
    buf[tail & mask] = r;
    ++tail;
    return true;
  }

  const Request& pop() {
    const Request& r = buf[head & mask];
    ++head;
    return r;
  }
};

struct Stats {
  std::vector<uint32_t> samples;
  explicit Stats(size_t cap) { samples.reserve(cap); }
  void add(uint32_t v) {
    if (samples.size() < samples.capacity()) samples.push_back(v);
  }
  void report(const char* name, const char* unit) {
    if (samples.empty()) { std::printf("  %-14s (no samples)\n", name); return; }
    std::sort(samples.begin(), samples.end());
    auto pct = [&](double p) { return samples[size_t(p * (samples.size() - 1))]; };
    std::printf("  %-14s p50=%-8u p99=%-8u p99.9=%-8u max=%-8u %s\n",
                name, pct(0.50), pct(0.99), pct(0.999), samples.back(), unit);
  }
};

struct Args {
  const char* itch = nullptr;
  const char* weights = nullptr;
  std::string symbol = "AAPL";
  uint64_t rate = 500000;        // offered events/sec
  uint64_t deadline_ns = 5000;   // per-event budget
  size_t queue_cap = 4096;       // must be a power of two
  uint64_t work_ns = DEFAULT_WORK_NS;
};

} // namespace

int main(int argc, char** argv) {
  Args a;
  if (argc < 3) {
    std::fprintf(stderr,
        "usage: %s <itch.bin> <weights_int8.bin> [--symbol S] [--rate N]\n"
        "          [--deadline NS] [--queue N] [--work NS]\n", argv[0]);
    return 1;
  }
  a.itch = argv[1];
  a.weights = argv[2];
  for (int i = 3; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if      (k == "--symbol")   a.symbol = argv[i + 1];
    else if (k == "--rate")     a.rate = std::strtoull(argv[i + 1], nullptr, 10);
    else if (k == "--deadline") a.deadline_ns = std::strtoull(argv[i + 1], nullptr, 10);
    else if (k == "--queue")    a.queue_cap = std::strtoull(argv[i + 1], nullptr, 10);
    else if (k == "--work")     a.work_ns = std::strtoull(argv[i + 1], nullptr, 10);
    else { std::fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
  }
  if (a.queue_cap == 0 || (a.queue_cap & (a.queue_cap - 1)) != 0) {
    std::fprintf(stderr, "--queue must be a power of two\n");
    return 1;
  }

  model::WeightsInt8 wq;
  if (!wq.load(a.weights)) return 1;
  model::FastBackend backend(wq);
  std::fprintf(stderr, "backend: %s\n", model::fast_backend_name());
  feat::Engine engine;

  FILE* f = std::fopen(a.itch, "rb");
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

  Book book;
  RingQueue q(a.queue_cap);
  Stats queue_wait(8u << 20);   // arrival -> scoring start
  Stats depth(8u << 20);

  uint16_t target = NO_LOCATE;
  uint64_t applied = 0, offered = 0, processed = 0;
  uint64_t drop_full = 0, drop_deadline = 0;

  // Virtual clock. Arrivals are synthesised at the target rate rather than
  // taken from the ITCH timestamps: the file's own timing is whatever the
  // market did that day, and the point here is to control offered load.
  const uint64_t interval_ns = 1'000'000'000ull / a.rate;
  uint64_t now = 0;
  uint64_t next_arrival = 0;
  uint64_t seq = 0;

  double checksum = 0.0;   // keeps the compiler from eliding the inference

  // Drain the queue up to `until` on the virtual clock. Called before each
  // arrival, so the consumer gets exactly the time between arrivals.
  auto drain = [&](uint64_t until) {
    while (!q.empty() && now + a.work_ns <= until) {
      const Request r = q.pop();     // copy: pop() returns a ref into buf

      // TODO A. Admission control.
      //   age = now - r.arrive_ns
      //   If age > a.deadline_ns, this request has already missed its
      //   budget. Increment drop_deadline, `continue`, and do NOT advance
      //   `now` -- see the design notes on why the check precedes the work.
      uint64_t age = now - r.arrive_ns;
      if (age > a.deadline_ns) {
        drop_deadline++;
        continue;
      }

      
      // TODO B. Otherwise record the wait and do the work.
      //   queue_wait.add(uint32_t(age));
      //   then the real inference, then now += a.work_ns; processed++;
      //
      //   feat::Record rec = engine.compute(r.ts, r.bid, r.ask);
      //   float logits[model::N_OUT];
      //   backend.forward(rec.f, logits);
      //   checksum += logits[0];

      queue_wait.add(uint32_t(age));

      feat::Record rec = engine.compute(r.ts, r.bid, r.ask);
      float logits[model::N_OUT];
      backend.forward(rec.f, logits);
      checksum += logits[0];

      now += a.work_ns;
      processed++;
    }
  };

  for (;;) {
    if ((valid - pos) < 2) { if (refill() == 0) break; continue; }
    uint16_t len = itch::read_be16(&buf[pos]);
    if ((valid - pos) < size_t(2) + len) { if (refill() == 0) break; continue; }

    const uint8_t* m = &buf[pos + 2];
    const uint8_t  type = m[0];
    const uint16_t locate = itch::read_be16(m + 1);
    const uint64_t ts = itch::read_be48(m + 5);

    if (type == 'R' && target == NO_LOCATE) {
      if (sym_to_string(m + 11) == a.symbol) {
        target = locate;
        std::printf("locate for %s: %u\n", a.symbol.c_str(), locate);
      }
    }
    if (target == NO_LOCATE || locate != target) { pos += 2 + len; continue; }

    // Mandatory work: the book must see every message.
    if (type == 'A' || type == 'F') {
      uint64_t ref    = itch::read_be64(m + 11);
      char     side   = char(m[19]);
      uint32_t shares = itch::read_be32(m + 20);
      uint32_t price  = itch::read_be32(m + 32);
      book.add_order(ref, side, price, shares);
      applied++;
    } else if (type == 'E' || type == 'C' || type == 'X') {
      uint64_t ref    = itch::read_be64(m + 11);
      uint32_t shares = itch::read_be32(m + 19);
      if (type == 'E' || type == 'C') {
        auto it = book.orders.find(ref);
        if (it != book.orders.end()) engine.on_trade(shares, it->second.side);
      }
      book.reduce(ref, shares);
      applied++;
    } else if (type == 'D') {
      book.remove(itch::read_be64(m + 11));
      applied++;
    } else if (type == 'U') {
      uint64_t old_ref = itch::read_be64(m + 11);
      uint64_t new_ref = itch::read_be64(m + 19);
      uint32_t shares  = itch::read_be32(m + 27);
      uint32_t price   = itch::read_be32(m + 31);
      book.replace(old_ref, new_ref, price, shares);
      applied++;
    } else {
      pos += 2 + len;
      continue;
    }

    // Droppable work is only meaningful with a two-sided book inside
    // market hours -- same scope rule the model was calibrated for.
    if (book.bids.empty() || book.asks.empty() ||
        ts < feat::MARKET_OPEN || ts >= feat::MARKET_CLOSE) {
      pos += 2 + len;
      continue;
    }

    // Let the consumer run up to this event's arrival, then admit it.
    drain(next_arrival);
    if (q.empty()) now = std::max(now, next_arrival);

    Request r;
    r.seq = seq++;
    r.arrive_ns = next_arrival;
    r.ts = ts;
    top_levels(book.bids, r.bid);
    top_levels(book.asks, r.ask);
    offered++;

    
    if (q.full()) {
        q.pop();          // evict the stalest pending request
        drop_full++;
    }
    q.push(r);          // the newest event always gets in

    depth.add(uint32_t(q.size()));
    next_arrival += interval_ns;

    pos += 2 + len;
  }

  // Let whatever is still queued finish, so the tail is not counted as a
  // drop that the policy caused.
  drain(UINT64_MAX);
  std::fclose(f);

  const double pct_proc = offered ? 100.0 * double(processed) / double(offered) : 0.0;
  const double pct_full = offered ? 100.0 * double(drop_full) / double(offered) : 0.0;
  const double pct_dl   = offered ? 100.0 * double(drop_deadline) / double(offered) : 0.0;

  std::printf("\noffered rate:   %llu events/sec  (interval %llu ns)\n",
              (unsigned long long)a.rate, (unsigned long long)interval_ns);
  std::printf("deadline:       %llu ns\n", (unsigned long long)a.deadline_ns);
  std::printf("queue capacity: %zu\n", a.queue_cap);
  std::printf("work per event: %llu ns\n\n", (unsigned long long)a.work_ns);

  std::printf("book messages applied: %llu\n", (unsigned long long)applied);
  std::printf("offered:   %10llu\n", (unsigned long long)offered);
  std::printf("processed: %10llu  (%.1f%%)\n",
              (unsigned long long)processed, pct_proc);
  std::printf("dropped:   %10llu  (%.1f%%)\n",
              (unsigned long long)(drop_full + drop_deadline), pct_full + pct_dl);
  std::printf("  queue full:  %10llu  (%.1f%%)\n",
              (unsigned long long)drop_full, pct_full);
  std::printf("  past deadline: %8llu  (%.1f%%)\n\n",
              (unsigned long long)drop_deadline, pct_dl);

  queue_wait.report("queue wait", "ns");
  depth.report("queue depth", "events");

  std::printf("\nchecksum: %.3f\n", checksum);
  return 0;
}