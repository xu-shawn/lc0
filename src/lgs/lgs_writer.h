/*
  LGS (Leela game store) v1 writer. Spec: docs/FORMAT.md in the T91 packing
  repo. Little-endian host assumed.
*/
#pragma once

#include <zlib.h>
#include <zstd.h>

#include <atomic>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace lgs {

static_assert(std::endian::native == std::endian::little,
              "LGS writer assumes a little-endian host");

enum Kind : uint8_t { kPerGame = 0, kPerPosition = 1, kStream = 2 };
enum DType : uint8_t {
  kU8 = 0,
  kI8 = 1,
  kU16 = 2,
  kI16 = 3,
  kU32 = 4,
  kI32 = 5,
  kU64 = 6,
  kF16 = 7,
  kF32 = 8,
  kBytes = 9
};

inline size_t DTypeSize(uint8_t dtype) {
  switch (dtype) {
    case kU8:
    case kI8:
    case kBytes:
      return 1;
    case kU16:
    case kI16:
    case kF16:
      return 2;
    case kU32:
    case kI32:
    case kF32:
      return 4;
    case kU64:
      return 8;
  }
  throw std::runtime_error("bad dtype");
}

struct Column {
  std::string name;
  uint8_t kind = kPerPosition;
  uint8_t dtype = kU8;
  uint8_t filter = 0;  // 1 = byte shuffle, 2 = xor with column `base` + shuffle
  uint8_t base = 0;    // directory index of the xor base column (filter 2)
  std::vector<uint8_t> raw;

  template <typename T>
  void Append(const T* p, size_t n) {
    const auto* b = reinterpret_cast<const uint8_t*>(p);
    raw.insert(raw.end(), b, b + n * sizeof(T));
  }
  template <typename T>
  void Append(const std::vector<T>& v) {
    Append(v.data(), v.size());
  }
  template <typename T>
  void Push(T v) {
    Append(&v, 1);
  }
};

inline void AppendVarint(std::vector<uint8_t>* out, uint64_t v) {
  while (v >= 0x80) {
    out->push_back(static_cast<uint8_t>(v) | 0x80);
    v >>= 7;
  }
  out->push_back(static_cast<uint8_t>(v));
}

inline std::vector<uint8_t> Shuffle(const std::vector<uint8_t>& raw,
                                    size_t width) {
  if (width <= 1) return raw;
  const size_t n = raw.size() / width;
  std::vector<uint8_t> out(raw.size());
  for (size_t i = 0; i < n; i++) {
    for (size_t k = 0; k < width; k++) out[k * n + i] = raw[i * width + k];
  }
  return out;
}

struct EncodedColumn {
  std::string name;
  uint8_t kind, dtype, filter, base;
  uint64_t raw_len;
  uint32_t crc;
  std::vector<uint8_t> comp;
};

// `base_raw` is the raw bytes of the xor base column (filter 2 only).
inline EncodedColumn Encode(const Column& c, int level,
                            const std::vector<uint8_t>* base_raw = nullptr) {
  EncodedColumn e;
  e.name = c.name;
  e.kind = c.kind;
  e.dtype = c.dtype;
  e.filter = c.filter;
  e.base = c.filter == 2 ? c.base : 0;
  e.raw_len = c.raw.size();
  e.crc = static_cast<uint32_t>(
      crc32(0L, c.raw.data(), static_cast<uInt>(c.raw.size())));
  // zlib crc32 takes uInt lengths; columns are far below 4 GiB.
  if (c.raw.size() > 0xFFFFFFFFull) throw std::runtime_error("column too big");
  std::vector<uint8_t> src;
  if (c.filter == 0) {
    src = c.raw;
  } else if (c.filter == 1) {
    src = Shuffle(c.raw, DTypeSize(c.dtype));
  } else if (c.filter == 2) {
    if (!base_raw || base_raw->size() != c.raw.size())
      throw std::runtime_error("xor base mismatch for " + c.name);
    std::vector<uint8_t> x(c.raw.size());
    for (size_t i = 0; i < x.size(); i++) x[i] = c.raw[i] ^ (*base_raw)[i];
    src = Shuffle(x, DTypeSize(c.dtype));
  } else {
    throw std::runtime_error("unknown filter");
  }
  e.comp.resize(ZSTD_compressBound(src.size()));
  const size_t n =
      ZSTD_compress(e.comp.data(), e.comp.size(), src.data(), src.size(), level);
  if (ZSTD_isError(n)) throw std::runtime_error(ZSTD_getErrorName(n));
  e.comp.resize(n);
  e.comp.shrink_to_fit();
  return e;
}

struct Block {
  uint32_t n_games = 0;
  uint32_t n_pos = 0;
  std::vector<Column> columns;
  std::vector<EncodedColumn> encoded;
};

// Compresses all columns of all blocks with `threads` workers.
inline void EncodeBlocks(std::vector<Block>* blocks, int level, int threads) {
  std::vector<std::pair<size_t, size_t>> tasks;
  for (size_t b = 0; b < blocks->size(); b++) {
    (*blocks)[b].encoded.resize((*blocks)[b].columns.size());
    for (size_t c = 0; c < (*blocks)[b].columns.size(); c++)
      tasks.emplace_back(b, c);
  }
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string error;
  auto worker = [&]() {
    try {
      for (size_t t; (t = next++) < tasks.size();) {
        auto [b, c] = tasks[t];
        auto& blk = (*blocks)[b];
        const auto& col = blk.columns[c];
        const std::vector<uint8_t>* base =
            col.filter == 2 ? &blk.columns.at(col.base).raw : nullptr;
        if (base && blk.columns.at(col.base).filter == 2)
          throw std::runtime_error("xor base must not itself be xor-filtered");
        blk.encoded[c] = Encode(col, level, base);
      }
    } catch (const std::exception& ex) {
      if (!failed.exchange(true)) error = ex.what();
    }
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < std::max(1, threads); i++) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (failed) throw std::runtime_error("column encode failed: " + error);
  for (auto& blk : *blocks)
    for (auto& col : blk.columns) std::vector<uint8_t>().swap(col.raw);
}

class FileWriter {
 public:
  explicit FileWriter(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot open " + path);
  }
  ~FileWriter() {
    if (f_) std::fclose(f_);
  }

  void WriteHeader(const std::string& json) {
    Put("LGS1", 4);
    PutU32(static_cast<uint32_t>(json.size()));
    Put(json.data(), json.size());
  }

  void WriteBlock(const Block& b) {
    index_.push_back({offset_, b.n_games, b.n_pos});
    Put("LGSB", 4);
    PutU32(b.n_games);
    PutU32(b.n_pos);
    PutU32(static_cast<uint32_t>(b.encoded.size()));
    for (const auto& e : b.encoded) {
      char name[16] = {0};
      if (e.name.size() > 16) throw std::runtime_error("column name too long");
      std::memcpy(name, e.name.data(), e.name.size());
      Put(name, 16);
      const uint8_t meta[4] = {e.kind, e.dtype, e.filter, e.base};
      Put(meta, 4);
      PutU64(e.raw_len);
      PutU64(e.comp.size());
      PutU32(e.crc);
    }
    for (const auto& e : b.encoded) Put(e.comp.data(), e.comp.size());
  }

  void Finish() {
    const uint64_t index_offset = offset_;
    Put("LGSI", 4);
    PutU32(static_cast<uint32_t>(index_.size()));
    for (const auto& ie : index_) {
      PutU64(ie.offset);
      PutU32(ie.n_games);
      PutU32(ie.n_pos);
    }
    PutU64(index_offset);
    Put("LGSE", 4);
    if (std::fflush(f_) != 0 || std::ferror(f_))
      throw std::runtime_error("write failed: " + path_);
    if (std::fclose(f_) != 0) {
      f_ = nullptr;
      throw std::runtime_error("close failed: " + path_);
    }
    f_ = nullptr;
  }

 private:
  struct IndexEntry {
    uint64_t offset;
    uint32_t n_games, n_pos;
  };
  void Put(const void* p, size_t n) {
    if (n && std::fwrite(p, 1, n, f_) != n)
      throw std::runtime_error("write failed: " + path_);
    offset_ += n;
  }
  void PutU32(uint32_t v) { Put(&v, 4); }
  void PutU64(uint64_t v) { Put(&v, 8); }

  std::string path_;
  FILE* f_ = nullptr;
  uint64_t offset_ = 0;
  std::vector<IndexEntry> index_;
};

// Minimal JSON string escaping for the header.
inline std::string JsonEscape(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        o += "\\\"";
        break;
      case '\\':
        o += "\\\\";
        break;
      case '\n':
        o += "\\n";
        break;
      case '\t':
        o += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += static_cast<char>(c);
        }
    }
  }
  return o;
}

}  // namespace lgs
