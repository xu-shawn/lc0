/*
  Minimal SHA-256 (FIPS 180-4) for LGS side-file headers.
    std::string hex = lgs::Sha256File("shard.lgs");
*/
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lgs {

class Sha256 {
 public:
  Sha256() { Reset(); }
  void Reset() {
    h_ = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    len_ = 0;
    fill_ = 0;
  }
  void Update(const uint8_t* p, size_t n) {
    len_ += n;
    while (n > 0) {
      const size_t take = std::min(n, size_t{64} - fill_);
      for (size_t i = 0; i < take; i++) buf_[fill_ + i] = p[i];
      fill_ += take;
      p += take;
      n -= take;
      if (fill_ == 64) {
        Block(buf_.data());
        fill_ = 0;
      }
    }
  }
  std::string HexDigest() {
    const uint64_t bits = len_ * 8;
    uint8_t pad = 0x80;
    Update(&pad, 1);
    const uint8_t zero = 0;
    while (fill_ != 56) Update(&zero, 1);
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) lenb[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    Update(lenb, 8);
    static const char* kHex = "0123456789abcdef";
    std::string out;
    for (uint32_t w : h_)
      for (int s = 28; s >= 0; s -= 4) out += kHex[(w >> s) & 0xF];
    return out;
  }

 private:
  static uint32_t Rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void Block(const uint8_t* p) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
      w[i] = (uint32_t{p[4 * i]} << 24) | (uint32_t{p[4 * i + 1]} << 16) |
             (uint32_t{p[4 * i + 2]} << 8) | uint32_t{p[4 * i + 3]};
    for (int i = 16; i < 64; i++) {
      const uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5],
             g = h_[6], h = h_[7];
    for (int i = 0; i < 64; i++) {
      const uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }

  std::array<uint32_t, 8> h_;
  std::array<uint8_t, 64> buf_;
  uint64_t len_ = 0;
  size_t fill_ = 0;
};

inline std::string Sha256File(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("sha256: cannot open " + path);
  Sha256 s;
  std::vector<uint8_t> buf(1 << 20);
  size_t n;
  while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) s.Update(buf.data(), n);
  const bool err = std::ferror(f);
  std::fclose(f);
  if (err) throw std::runtime_error("sha256: read error " + path);
  return s.HexDigest();
}

}  // namespace lgs
