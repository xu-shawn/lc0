// binpack/binpack.hpp — public API of the standalone binpack library.
//
// A header-only reader/writer for the chess-training "binpack" format
// (Stockfish/nnue-pytorch). Drop in <binpack/binpack.hpp>; that pulls in the
// internal codec automatically. The library depends only on the C++17
// standard library.
//
// Public surface:
//   binpack::Entry          POD record (FEN + UCI move + score/ply/result)
//   binpack::Reader         RAII sequential reader, with range-based-for
//   binpack::Writer         RAII sequential writer
//   binpack::format_error   thrown on malformed file payloads
//
// On-disk file errors throw `format_error`; OS-level open failures throw
// `std::system_error` with the captured errno.

#pragma once

#include "detail.hpp"

#include <cstdint>
#include <cerrno>
#include <filesystem>
#include <iterator>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace binpack {

    // -------------------------------------------------------------------------
    // Entry — one training-data record.
    //
    // `score` and `result` are stored verbatim from the on-disk record; their
    // perspective (side-to-move vs. white) follows whatever the writing tool
    // used. See README.md "Caveats" before interpreting their sign.
    // -------------------------------------------------------------------------
    struct Entry {
        std::string   fen;
        std::string   move_uci;
        std::int16_t  score  = 0;
        std::uint16_t ply    = 0;
        std::int16_t  result = 0;
    };

    // -------------------------------------------------------------------------
    // Conversions between the public stringified Entry and the internal codec
    // record. Defined inline below so users can reach for them if they need
    // to interleave the two layers.
    // -------------------------------------------------------------------------
    [[nodiscard]] inline Entry to_entry(const detail::TrainingDataEntry& raw) {
        Entry e;
        e.fen      = raw.pos.fen();
        e.move_uci = detail::chess::uci::moveToUci(raw.pos, raw.move);
        e.score    = raw.score;
        e.ply      = raw.ply;
        e.result   = raw.result;
        return e;
    }

    [[nodiscard]] inline detail::TrainingDataEntry to_raw(const Entry& e) {
        detail::TrainingDataEntry raw;
        try {
            raw.pos = detail::chess::Position::fromFen(e.fen.c_str());
        } catch (const std::exception& ex) {
            throw format_error(std::string("binpack: invalid FEN: ") + ex.what());
        } catch (...) {
            throw format_error("binpack: invalid FEN");
        }
        try {
            raw.move = detail::chess::uci::uciToMove(raw.pos, e.move_uci);
        } catch (const std::exception& ex) {
            throw format_error(std::string("binpack: invalid UCI move: ") + ex.what());
        } catch (...) {
            throw format_error("binpack: invalid UCI move");
        }
        raw.score  = e.score;
        raw.ply    = e.ply;
        raw.result = e.result;
        return raw;
    }

    // -------------------------------------------------------------------------
    // Reader — sequential, single-pass read of a .binpack file.
    //
    // Construction throws std::system_error (errno-based) if the file cannot
    // be opened, and format_error if the first chunk header is malformed.
    // `has_next()` is O(1) and noexcept thanks to one-entry prefetch.
    // `next()` throws format_error if the stream is exhausted.
    // Range-based for is supported via single-pass input iterator.
    // -------------------------------------------------------------------------
    class Reader {
    public:
        explicit Reader(const std::filesystem::path& path)
            : inner_(open_or_throw(path))
        {
            prefetch();
        }

        Reader(const Reader&)            = delete;
        Reader& operator=(const Reader&) = delete;
        Reader(Reader&&)                 = delete;
        Reader& operator=(Reader&&)      = delete;

        [[nodiscard]] bool has_next() const noexcept { return cached_.has_value(); }

        Entry next() {
            if (!cached_) {
                throw format_error("binpack::Reader::next: stream is exhausted");
            }
            Entry out = std::move(*cached_);
            prefetch();
            return out;
        }

        [[nodiscard]] std::optional<Entry> try_next() {
            if (!cached_) return std::nullopt;
            Entry out = std::move(*cached_);
            prefetch();
            return out;
        }

        // Single-pass input iterator. operator++ advances the parent Reader,
        // operator* returns the cached prefetched Entry. end() is a sentinel
        // that compares equal once the underlying stream is exhausted.
        class iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using value_type        = Entry;
            using difference_type   = std::ptrdiff_t;
            using pointer           = const Entry*;
            using reference         = const Entry&;

            iterator() = default;

            reference operator*() const  { return *parent_->cached_; }
            pointer   operator->() const { return &*parent_->cached_; }

            iterator& operator++() {
                parent_->prefetch();
                if (!parent_->cached_) parent_ = nullptr;
                return *this;
            }

            // Post-increment returns void to keep the input-iterator simple
            // and avoid copying the cached Entry.
            void operator++(int) { ++*this; }

            friend bool operator==(const iterator& a, const iterator& b) noexcept {
                return a.parent_ == b.parent_;
            }
            friend bool operator!=(const iterator& a, const iterator& b) noexcept {
                return !(a == b);
            }

        private:
            friend class Reader;
            explicit iterator(Reader* p) : parent_(p) {}
            Reader* parent_ = nullptr;
        };

        iterator begin() {
            return cached_ ? iterator(this) : iterator();
        }
        iterator end() noexcept { return iterator(); }

    private:
        // Keep these declared in the order they are constructed.
        detail::CompressedTrainingDataEntryReader inner_;
        std::optional<Entry>                      cached_;

        // Returns a constructed inner reader, or throws std::system_error if
        // the file cannot be opened. We probe the path with `std::ifstream`
        // first so we get a real errno, then hand the path to the codec.
        static detail::CompressedTrainingDataEntryReader
        open_or_throw(const std::filesystem::path& path) {
            {
                std::ifstream probe(path, std::ios::binary);
                if (!probe) {
                    const int err = errno ? errno : ENOENT;
                    throw std::system_error(
                        err, std::generic_category(),
                        "binpack::Reader: cannot open " + path.string());
                }
            }
            // The underlying codec wants `in` openmode, not the upstream
            // `app` default which has surprising read-position semantics.
            return detail::CompressedTrainingDataEntryReader(
                path.string(), std::ios::in | std::ios::binary);
        }

        void prefetch() {
            if (inner_.hasNext()) {
                cached_ = to_entry(inner_.next());
            } else {
                cached_.reset();
            }
        }
    };

    // -------------------------------------------------------------------------
    // Writer — sequential append of Entries to a .binpack file.
    //
    // Default OpenMode is `Truncate` (clean overwrite). Pass `Append` to grow
    // an existing shard. The destructor flushes any pending in-memory chunk.
    // `write()` throws format_error if the FEN or UCI move are malformed.
    // -------------------------------------------------------------------------
    class Writer {
    public:
        enum class OpenMode { Truncate, Append };

        explicit Writer(const std::filesystem::path& path,
                        OpenMode mode = OpenMode::Truncate)
            : inner_(path.string(), to_openmode(mode))
        {}

        Writer(const Writer&)            = delete;
        Writer& operator=(const Writer&) = delete;
        Writer(Writer&&)                 = delete;
        Writer& operator=(Writer&&)      = delete;

        void write(const Entry& e) {
            inner_.addTrainingDataEntry(to_raw(e));
        }

    private:
        detail::CompressedTrainingDataEntryWriter inner_;

        static std::ios::openmode to_openmode(OpenMode m) {
            // Both modes go via std::fstream, which the codec opens with
            // `binary | in | out | <mode>`. Truncate creates a fresh file;
            // append seeks to EOF on every write.
            return m == OpenMode::Append ? std::ios::app : std::ios::trunc;
        }
    };

}  // namespace binpack
