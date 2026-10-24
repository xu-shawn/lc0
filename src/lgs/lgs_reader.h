/*
  LGS (Leela game store) v1 reader. Spec: docs/FORMAT.md. Header-only;
  needs zstd and zlib. Little-endian host assumed.

    lgs::Reader r("shard.lgs");
    for (size_t b = 0; b < r.num_blocks(); b++) {
      lgs::BlockData blk = r.ReadBlock(b);          // or ReadBlock(b, {"best_q"})
      const float* q = blk.Get("best_q").As<float>();
      ...
    }
*/
#pragma once

#include <zlib.h>
#include <zstd.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <sys/types.h>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lgs {

static_assert(std::endian::native == std::endian::little,
              "LGS reader assumes a little-endian host");

enum ReaderDType : uint8_t {
  kRU8 = 0, kRI8 = 1, kRU16 = 2, kRI16 = 3, kRU32 = 4,
  kRI32 = 5, kRU64 = 6, kRF16 = 7, kRF32 = 8, kRBytes = 9
};

inline size_t ReaderDTypeSize(uint8_t dtype) {
  static const size_t kSizes[] = {1, 1, 2, 2, 4, 4, 8, 2, 4, 1};
  if (dtype > 9) throw std::runtime_error("lgs: bad dtype");
  return kSizes[dtype];
}

struct ColumnData {
  std::string name;
  uint8_t kind = 0, dtype = 0, filter = 0, base = 0;
  std::vector<uint8_t> raw;

  template <typename T>
  const T* As() const {
    if (sizeof(T) != ReaderDTypeSize(dtype))
      throw std::runtime_error("lgs: element size mismatch for " + name);
    return reinterpret_cast<const T*>(raw.data());
  }
  size_t Count() const { return raw.size() / ReaderDTypeSize(dtype); }
};

struct BlockData {
  uint32_t n_games = 0;
  uint32_t n_pos = 0;
  std::vector<ColumnData> columns;

  const ColumnData* Find(const std::string& name) const {
    for (const auto& c : columns)
      if (c.name == name) return &c;
    return nullptr;
  }
  const ColumnData& Get(const std::string& name) const {
    const ColumnData* c = Find(name);
    if (!c) throw std::runtime_error("lgs: missing column " + name);
    return *c;
  }
};

class Reader {
 public:
  explicit Reader(const std::string& path) : path_(path) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) throw std::runtime_error("lgs: cannot open " + path);
    char magic[4];
    ReadAt(0, magic, 4);
    if (std::memcmp(magic, "LGS1", 4) != 0)
      throw std::runtime_error("lgs: bad magic in " + path);
    uint32_t json_len;
    ReadAt(4, &json_len, 4);
    header_.resize(json_len);
    ReadAt(8, header_.data(), json_len);

    if (fseeko(f_, 0, SEEK_END) != 0) throw std::runtime_error("lgs: seek");
    const int64_t size = ftello(f_);
    if (size < 12) throw std::runtime_error("lgs: truncated " + path);
    uint64_t index_offset;
    ReadAt(size - 12, &index_offset, 8);
    ReadAt(size - 4, magic, 4);
    if (std::memcmp(magic, "LGSE", 4) != 0)
      throw std::runtime_error("lgs: missing trailer (truncated?) " + path);
    ReadAt(index_offset, magic, 4);
    if (std::memcmp(magic, "LGSI", 4) != 0)
      throw std::runtime_error("lgs: bad index " + path);
    uint32_t n_blocks;
    ReadAt(index_offset + 4, &n_blocks, 4);
    blocks_.resize(n_blocks);
    for (uint32_t i = 0; i < n_blocks; i++) {
      const uint64_t at = index_offset + 8 + uint64_t{16} * i;
      ReadAt(at, &blocks_[i].offset, 8);
      ReadAt(at + 8, &blocks_[i].n_games, 4);
      ReadAt(at + 12, &blocks_[i].n_pos, 4);
    }
  }
  ~Reader() {
    if (f_) std::fclose(f_);
  }
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  const std::string& HeaderJson() const { return header_; }
  size_t NumBlocks() const { return blocks_.size(); }
  uint32_t BlockGames(size_t b) const { return blocks_.at(b).n_games; }
  uint32_t BlockPositions(size_t b) const { return blocks_.at(b).n_pos; }

  // Reads block b. With `only` non-empty, decodes just those columns (plus
  // any xor bases they need).
  BlockData ReadBlock(size_t b, std::initializer_list<std::string> only = {}) {
    return ReadBlock(b, std::vector<std::string>(only));
  }

  BlockData ReadBlock(size_t b, const std::vector<std::string>& only) {
    const auto& ie = blocks_.at(b);
    char magic[4];
    ReadAt(ie.offset, magic, 4);
    if (std::memcmp(magic, "LGSB", 4) != 0)
      throw std::runtime_error("lgs: bad block magic");
    uint32_t hdr[3];
    ReadAt(ie.offset + 4, hdr, 12);
    BlockData blk;
    blk.n_games = hdr[0];
    blk.n_pos = hdr[1];
    const uint32_t n_cols = hdr[2];

    struct Entry {
      ColumnData meta;
      uint64_t raw_len, comp_len, offset;
      uint32_t crc;
      bool want;
    };
    std::vector<Entry> ents(n_cols);
    std::vector<uint8_t> dir(size_t{40} * n_cols);
    ReadAt(ie.offset + 16, dir.data(), dir.size());
    uint64_t payload = ie.offset + 16 + dir.size();
    for (uint32_t i = 0; i < n_cols; i++) {
      const uint8_t* d = dir.data() + size_t{40} * i;
      Entry& e = ents[i];
      e.meta.name.assign(reinterpret_cast<const char*>(d),
                         strnlen(reinterpret_cast<const char*>(d), 16));
      e.meta.kind = d[16];
      e.meta.dtype = d[17];
      e.meta.filter = d[18];
      e.meta.base = d[19];
      std::memcpy(&e.raw_len, d + 20, 8);
      std::memcpy(&e.comp_len, d + 28, 8);
      std::memcpy(&e.crc, d + 36, 4);
      e.offset = payload;
      payload += e.comp_len;
      e.want = only.empty();
      for (const auto& n : only)
        if (n == e.meta.name) e.want = true;
    }
    for (auto& e : ents)
      if (e.want && e.meta.filter == 2) ents.at(e.meta.base).want = true;

    // Decode in directory order; xor bases are decoded before use because
    // they are decoded (pass 1) before any xor is applied (pass 2).
    std::vector<uint8_t> comp;
    for (auto& e : ents) {
      if (!e.want) continue;
      comp.resize(e.comp_len);
      ReadAt(e.offset, comp.data(), comp.size());
      std::vector<uint8_t> buf(e.raw_len);
      const size_t n = ZSTD_decompress(buf.data(), buf.size(), comp.data(),
                                       comp.size());
      if (ZSTD_isError(n) || n != e.raw_len)
        throw std::runtime_error("lgs: zstd error in column " + e.meta.name);
      if (e.meta.filter == 1 || e.meta.filter == 2) {
        const size_t w = ReaderDTypeSize(e.meta.dtype);
        if (w > 1) {
          const size_t cnt = buf.size() / w;
          std::vector<uint8_t> un(buf.size());
          for (size_t k = 0; k < w; k++)
            for (size_t i = 0; i < cnt; i++) un[i * w + k] = buf[k * cnt + i];
          buf.swap(un);
        }
      } else if (e.meta.filter != 0) {
        throw std::runtime_error("lgs: unknown filter in " + e.meta.name);
      }
      e.meta.raw = std::move(buf);
    }
    for (auto& e : ents) {
      if (!e.want || e.meta.filter != 2) continue;
      const auto& base = ents.at(e.meta.base).meta;
      if (base.filter == 2 || base.raw.size() != e.meta.raw.size())
        throw std::runtime_error("lgs: bad xor base for " + e.meta.name);
      for (size_t i = 0; i < e.meta.raw.size(); i++)
        e.meta.raw[i] ^= base.raw[i];
    }
    for (auto& e : ents) {
      if (!e.want) continue;
      const uint32_t crc = static_cast<uint32_t>(
          crc32(0L, e.meta.raw.data(), static_cast<uInt>(e.meta.raw.size())));
      if (crc != e.crc)
        throw std::runtime_error("lgs: CRC mismatch in column " + e.meta.name);
    }
    for (auto& e : ents) {
      if (!e.want) continue;
      bool requested = only.empty();
      for (const auto& n : only)
        if (n == e.meta.name) requested = true;
      if (requested) blk.columns.push_back(std::move(e.meta));
    }
    return blk;
  }

 private:
  struct IndexEntry {
    uint64_t offset = 0;
    uint32_t n_games = 0, n_pos = 0;
  };
  void ReadAt(uint64_t off, void* dst, size_t n) {
    if (fseeko(f_, static_cast<off_t>(off), SEEK_SET) != 0 ||
        std::fread(dst, 1, n, f_) != n)
      throw std::runtime_error("lgs: read failed in " + path_);
  }

  std::string path_;
  FILE* f_ = nullptr;
  std::string header_;
  std::vector<IndexEntry> blocks_;
};

// Reads one unsigned LEB128 varint; advances *p.
inline uint64_t ReadVarint(const uint8_t** p, const uint8_t* end) {
  uint64_t v = 0;
  int shift = 0;
  while (*p < end) {
    const uint8_t b = *(*p)++;
    v |= uint64_t{b & 0x7Fu} << shift;
    if (!(b & 0x80)) return v;
    shift += 7;
    if (shift > 63) break;
  }
  throw std::runtime_error("lgs: bad varint");
}

}  // namespace lgs
