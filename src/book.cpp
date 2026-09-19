#include "itch.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

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

// Strip trailing spaces from an 8-byte ITCH symbol field.
std::string sym_to_string(const uint8_t* p) {
  std::string s(reinterpret_cast<const char*>(p), 8);
  while (!s.empty() && s.back() == ' ') s.pop_back();
  return s;
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <file> <symbol>\n", argv[0]);
    return 1;
  }
  const std::string want = argv[2];

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("fopen"); return 1; }

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
  uint16_t target = NO_LOCATE;
  uint64_t applied = 0, crossings = 0;

  for (;;) {
    if ((valid - pos) < 2) { if (refill() == 0) break; continue; }
    uint16_t len = itch::read_be16(&buf[pos]);
    if ((valid - pos) < size_t(2) + len) { if (refill() == 0) break; continue; }

    const uint8_t* m = &buf[pos + 2];   // start of message body
    const uint8_t  type = m[0];
    const uint16_t locate = itch::read_be16(m + 1);

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

    // 4. After each applied message, check book.consistent(). Count
    //    crossings — don't abort on the first one until you've seen
    //    how often it happens.
    if (!book.consistent()) crossings++;

    pos += 2 + len;
  }

  std::fclose(f);
  std::printf("applied: %llu  crossings: %llu\n",
              (unsigned long long)applied, (unsigned long long)crossings);
  std::printf("orders resting: %zu  unknown_ref: %u  bad_reduce: %u\n",
            book.orders.size(), book.unknown_ref, book.bad_reduce);
  // TODO: print final best bid / best ask and a few levels of depth.
  return 0;
}