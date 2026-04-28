# binpack — header-only reader/writer for the chess `.binpack` format

A standalone, header-only C++17 library for reading and writing the
`.binpack` training-data format used by Stockfish / nnue-pytorch. Depends
only on the C++ standard library — drop the `include/` directory into your
project and `#include <binpack/binpack.hpp>`.

This library was extracted from
`src/extra/nnue_data_binpack_format.h` in this repository. The codec
internals are unchanged; the public API is new.

## Quick start

```cpp
#include <binpack/binpack.hpp>
#include <iostream>

int main() {
    binpack::Reader reader("training.binpack");
    binpack::Writer writer("first_ten.binpack");

    std::size_t i = 0;
    for (const auto& e : reader) {
        std::cout << e.fen << "  " << e.move_uci
                  << "  score=" << e.score << "\n";
        writer.write(e);
        if (++i == 10) break;
    }
}
```

## Public API

```cpp
namespace binpack {

struct Entry {
    std::string   fen;       // FEN of the position
    std::string   move_uci;  // best move in UCI notation, e.g. "e2e4", "a7a8q"
    std::int16_t  score;     // raw score field (perspective inherited; see Caveats)
    std::uint16_t ply;       // half-move ply within the source game
    std::int16_t  result;    // -1 / 0 / +1 (perspective inherited; see Caveats)
};

class format_error : public std::runtime_error { /* ... */ };

class Reader {
public:
    explicit Reader(const std::filesystem::path&);
    bool                 has_next() const noexcept;
    Entry                next();         // throws format_error if exhausted
    std::optional<Entry> try_next();
    // range-based-for support: for (const auto& e : reader) { ... }
    iterator begin();
    iterator end();
};

class Writer {
public:
    enum class OpenMode { Truncate, Append };
    explicit Writer(const std::filesystem::path&, OpenMode = OpenMode::Truncate);
    void write(const Entry&);            // throws format_error on bad FEN/UCI
    // destructor flushes the pending in-memory chunk
};

}  // namespace binpack
```

`std::system_error` (errno-based) is thrown when a file cannot be opened.
`format_error` is thrown on malformed file payloads (bad chunk magic /
oversized chunk) and on FEN / UCI strings that the codec cannot parse.

`Reader` is single-pass and prefetches one entry, so `has_next()` is
`const noexcept`. The iterator models the older *InputIterator* named
requirement (works with C++11 range-for); it does not claim to satisfy
the C++20 `std::input_iterator` concept.

## Building the example

```sh
cd binpacks/example
make           # -> ./print_first_ten
make run       # reads ../../small.binpack, writes ./first_ten.binpack
```

The example reads up to 10 entries, prints each as a labeled record, and
writes them back to a fresh `.binpack`. Re-running it on its own output
is the round-trip test.

## File layout

```
binpacks/
├── README.md
├── include/
│   └── binpack/
│       ├── binpack.hpp     # public API: Entry, Reader, Writer, format_error
│       ├── chess.hpp       # internal chess primitives (binpack::detail::chess)
│       └── detail.hpp      # internal codec  (binpack::detail)
└── example/
    ├── print_first_ten.cpp
    └── Makefile
```

Anything under `binpack::detail` is implementation detail — no stability
guarantees. Public users should only touch `binpack::Entry`,
`binpack::Reader`, `binpack::Writer`, and `binpack::format_error`.

## On-disk format (informative)

A `.binpack` file is a sequence of chunks. Each chunk:

```
+---------+---------+---------------------+
| 4 bytes | 4 bytes | <size> bytes        |
| "BINP"  | size LE | packed entries      |
+---------+---------+---------------------+
```

The chunk-size header is **little-endian uint32**. Inside a chunk, an
entry is 32 bytes (huffman-packed position + compressed move + score +
ply/result + rule50), optionally followed by a 2-byte big-endian
`numPlies` and a bit-packed sequence of continuation moves and
score-deltas (variable-length encoding, 4-bit blocks). The endianness
asymmetry — chunk size LE, numPlies BE — is part of the format; do
not "fix" it.

## Caveats

- **`score` and `result` perspective.** These fields are stored
  *verbatim* from the on-disk record. Producers (e.g. Stockfish's
  data-generation tools) tend to write side-to-move-relative values,
  but the convention is not enforced by the format. If you are
  consuming binpacks from an unfamiliar source, sanity-check a few
  entries against known evaluations before training on them.

- **`OpenMode::Truncate` is the default.** This is a deliberate
  break from the underlying codec's `std::ios::app` default, which
  silently grew an existing shard on every open. To append to an
  existing file, pass `OpenMode::Append` explicitly.

- **Missing files throw `std::system_error`, not `format_error`.**
  This lets callers distinguish "no such file" from "corrupt file".

- **Single-pass, no random access.** A `Reader` reads forward only.
  To restart, construct a new `Reader`.

- **Header-only with ~7,500 lines.** Compile-time cost is real if the
  header is included in many translation units. Consider isolating
  it behind a single `.cpp` in your build.
