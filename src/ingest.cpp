#include "itch.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
constexpr size_t CHUNK = 1 << 20;  // 1 MiB

// CHUNK is far larger than the largest ITCH message (~50 bytes), so a
// successful refill always satisfies any single-message request. The loop's
// second guard relies on this.
static_assert(CHUNK > 64 * 1024, "chunk must exceed max message size");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file>\n", argv[0]);
    return 1;
  }

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::perror("fopen"); return 1; }

  std::vector<uint8_t> buf(CHUNK);
  uint64_t counts[256] = {};
  uint64_t total = 0;

  size_t valid = 0;   // bytes of real data currently in buf
  size_t pos   = 0;   // read cursor within buf

  auto refill = [&]() -> size_t {
    size_t leftover = valid - pos;
    std::memmove(buf.data(), buf.data() + pos, leftover);
    size_t got = fread(buf.data() + leftover, 1, CHUNK - leftover, f);
    valid = leftover + got;
    pos = 0;
    return got;
  };

  for (;;) {
    // 1. If fewer than 2 bytes remain, we can't read a length prefix.
    //    Move the leftover to the front of buf and refill from f.
    //    If the refill returns 0 bytes, we're done.
    if ((valid - pos) < 2) {
        size_t filled = refill();
        if (filled == 0)
            break;
        continue;
    }

    // 2. Read the 2-byte big-endian length prefix at buf[pos].
    uint16_t length = itch::read_be16(&buf[pos])

    // 3. If fewer than 2 + len bytes remain in buf, the message straddles
    //    the chunk boundary. Compact and refill, then re-read the prefix.
    if ((valid - pos) < 2 + length) {
        size_t filled = refill();
        if (filled == 0)
            break;
        continue;
     }

    // 4. The type byte is at buf[pos + 2]. Count it, advance pos by
    //    2 + len, increment total.
    uint8_t type = buf[pos + 2];
    counts[type]++;
    pos += static_cast<size_t>(2) + length;
    total++;

  }

  std::fclose(f);

  std::printf("total messages: %llu\n", (unsigned long long)total);
  for (int i = 0; i < 256; ++i) {
    if (counts[i])
      std::printf("  %c  %llu\n", i, (unsigned long long)counts[i]);
  }
  // TODO: assert we consumed to exactly EOF with no leftover bytes.
  return 0;
}