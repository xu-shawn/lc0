// binpack/detail.hpp — internal codec for the binpack training-data format.
//
// This file is the original `namespace binpack { ... }` from
// src/extra/nnue_data_binpack_format.h (Stockfish's training-data tools),
// wrapped in `namespace binpack::detail` so that the codec lives below the
// modern public API. Compared to the original, this file:
//   * throws `binpack::format_error` instead of `assert(false)` on a malformed
//     chunk header / oversized chunk;
//   * drops the iostream-coupled convert*/validate*/emit* utilities, which are
//     out of scope for a codec library.
//
// **NOT PUBLIC API** — work with `binpack::Reader` / `binpack::Writer` instead.

#pragma once

#include "chess.hpp"

#include <stdexcept>

namespace binpack {
    // Forward-declared here so the codec below can throw it; the real
    // definition (with public-API documentation) lives in binpack.hpp.
    class format_error : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };
}

namespace binpack { namespace detail {

// References to `chess::*` inside the codec resolve via inner-scope lookup
// to `binpack::detail::chess::*` (defined in chess.hpp).

    constexpr std::size_t KiB = 1024;
    constexpr std::size_t MiB = (1024*KiB);
    constexpr std::size_t GiB = (1024*MiB);

    constexpr std::size_t suggestedChunkSize = MiB;
    constexpr std::size_t maxMovelistSize = 10*KiB; // a safe upper bound
    constexpr std::size_t maxChunkSize = 100*MiB; // to prevent malformed files from causing huge allocations

    using namespace std::literals;

    namespace nodchip
    {
        // This namespace contains modified code from https://github.com/nodchip/Stockfish
        // which is released under GPL v3 license https://www.gnu.org/licenses/gpl-3.0.html

        using namespace std;

        struct StockfishMove
        {
            [[nodiscard]] static StockfishMove fromMove(chess::Move move)
            {
                StockfishMove sfm;

                sfm.m_raw = 0;

                unsigned moveFlag = 0;
                if (move.type == chess::MoveType::Promotion) moveFlag = 1;
                else if (move.type == chess::MoveType::EnPassant) moveFlag = 2;
                else if (move.type == chess::MoveType::Castle) moveFlag = 3;

                unsigned promotionIndex = 0;
                if (move.type == chess::MoveType::Promotion)
                {
                    promotionIndex = static_cast<int>(move.promotedPiece.type()) - static_cast<int>(chess::PieceType::Knight);
                }

                sfm.m_raw |= static_cast<std::uint16_t>(moveFlag);
                sfm.m_raw <<= 2;
                sfm.m_raw |= static_cast<std::uint16_t>(promotionIndex);
                sfm.m_raw <<= 6;
                sfm.m_raw |= static_cast<int>(move.from);
                sfm.m_raw <<= 6;
                sfm.m_raw |= static_cast<int>(move.to);

                return sfm;
            }

            [[nodiscard]] chess::Move toMove() const
            {
                const chess::Square to = static_cast<chess::Square>((m_raw & (0b111111 << 0) >> 0));
                const chess::Square from = static_cast<chess::Square>((m_raw & (0b111111 << 6)) >> 6);

                const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
                const chess::PieceType promotionType = static_cast<chess::PieceType>(static_cast<int>(chess::PieceType::Knight) + promotionIndex);

                const unsigned moveFlag = (m_raw & (0b11 << 14)) >> 14;
                chess::MoveType type = chess::MoveType::Normal;
                if (moveFlag == 1) type = chess::MoveType::Promotion;
                else if (moveFlag == 2) type = chess::MoveType::EnPassant;
                else if (moveFlag == 3) type = chess::MoveType::Castle;

                if (type == chess::MoveType::Promotion)
                {
                    const chess::Color stm = to.rank() == chess::rank8 ? chess::Color::White : chess::Color::Black;
                    return chess::Move{from, to, type, chess::Piece(promotionType, stm)};
                }

                return chess::Move{from, to, type};
            }

            [[nodiscard]] std::string toString() const
            {
                const chess::Square to = static_cast<chess::Square>((m_raw & (0b111111 << 0) >> 0));
                const chess::Square from = static_cast<chess::Square>((m_raw & (0b111111 << 6)) >> 6);

                const unsigned promotionIndex = (m_raw & (0b11 << 12)) >> 12;
                const chess::PieceType promotionType = static_cast<chess::PieceType>(static_cast<int>(chess::PieceType::Knight) + promotionIndex);

                std::string r;
                chess::parser_bits::appendSquareToString(from, r);
                chess::parser_bits::appendSquareToString(to, r);
                if (promotionType != chess::PieceType::None)
                {
                    r += chess::EnumTraits<chess::PieceType>::toChar(promotionType, chess::Color::Black);
                }

                return r;
            }

        private:
            std::uint16_t m_raw;
        };
        static_assert(sizeof(StockfishMove) == sizeof(std::uint16_t));

        struct PackedSfen
        {
            uint8_t data[32];
        };

        struct PackedSfenValue
        {
            // phase
            PackedSfen sfen;

            // Evaluation value returned from Learner::search()
            int16_t score;

            // PV first move
            // Used when finding the match rate with the teacher
            StockfishMove move;

            // Trouble of the phase from the initial phase.
            uint16_t gamePly;

            // 1 if the player on this side ultimately wins the game. -1 if you are losing.
            // 0 if a draw is reached.
            // The draw is in the teacher position generation command gensfen,
            // Only write if LEARN_GENSFEN_DRAW_RESULT is enabled.
            int8_t game_result;

            // When exchanging the file that wrote the teacher aspect with other people
            //Because this structure size is not fixed, pad it so that it is 40 bytes in any environment.
            uint8_t padding;

            // 32 + 2 + 2 + 2 + 1 + 1 = 40bytes
        };
        static_assert(sizeof(PackedSfenValue) == 40);
        // Class that handles bitstream

        // useful when doing aspect encoding
        struct BitStream
        {
            // Set the memory to store the data in advance.
            // Assume that memory is cleared to 0.
            void  set_data(uint8_t* data_) { data = data_; reset(); }

            // Get the pointer passed in set_data().
            uint8_t* get_data() const { return data; }

            // Get the cursor.
            int get_cursor() const { return bit_cursor; }

            // reset the cursor
            void reset() { bit_cursor = 0; }

            // Write 1bit to the stream.
            // If b is non-zero, write out 1. If 0, write 0.
            void write_one_bit(int b)
            {
                if (b)
                    data[bit_cursor / 8] |= 1 << (bit_cursor & 7);

                ++bit_cursor;
            }

            // Get 1 bit from the stream.
            int read_one_bit()
            {
                int b = (data[bit_cursor / 8] >> (bit_cursor & 7)) & 1;
                ++bit_cursor;

                return b;
            }

            // write n bits of data
            // Data shall be written out from the lower order of d.
            void write_n_bit(int d, int n)
            {
                for (int i = 0; i <n; ++i)
                    write_one_bit(d & (1 << i));
            }

            // read n bits of data
            // Reverse conversion of write_n_bit().
            int read_n_bit(int n)
            {
                int result = 0;
                for (int i = 0; i < n; ++i)
                    result |= read_one_bit() ? (1 << i) : 0;

                return result;
            }

        private:
            // Next bit position to read/write.
            int bit_cursor;

            // data entity
            uint8_t* data;
        };


        // Huffman coding
        // * is simplified from mini encoding to make conversion easier.
        //
        // Huffman Encoding
        //
        // Empty  xxxxxxx0
        // Pawn   xxxxx001 + 1 bit (Color)
        // Knight xxxxx011 + 1 bit (Color)
        // Bishop xxxxx101 + 1 bit (Color)
        // Rook   xxxxx111 + 1 bit (Color)
        // Queen   xxxx1001 + 1 bit (Color)
        //
        // Worst case:
        // - 32 empty squares    32 bits
        // - 30 pieces           150 bits
        // - 2 kings             12 bits
        // - castling rights     4 bits
        // - ep square           7 bits
        // - rule50              7 bits
        // - game ply            16 bits
        // - TOTAL               228 bits < 256 bits

        struct HuffmanedPiece
        {
            int code; // how it will be coded
            int bits; // How many bits do you have
        };

        // NOTE: Order adjusted for this library because originally NO_PIECE had index 0
        constexpr HuffmanedPiece huffman_table[] =
        {
            {0b0001,4}, // PAWN     1
            {0b0011,4}, // KNIGHT   3
            {0b0101,4}, // BISHOP   5
            {0b0111,4}, // ROOK     7
            {0b1001,4}, // QUEEN    9
            {-1,-1},    // KING - unused
            {0b0000,1}, // NO_PIECE 0
        };

        // Class for compressing/decompressing sfen
        // sfen can be packed to 256bit (32bytes) by Huffman coding.
        // This is proven by mini. The above is Huffman coding.
        //
        // Internal format = 1-bit turn + 7-bit king position *2 + piece on board (Huffman coding) + hand piece (Huffman coding)
        // Side to move (White = 0, Black = 1) (1bit)
        // White King Position (6 bits)
        // Black King Position (6 bits)
        // Huffman Encoding of the board
        // Castling availability (1 bit x 4)
        // En passant square (1 or 1 + 6 bits)
        // Rule 50 (6 bits)
        // Game play (8 bits)
        //
        // TODO(someone): Rename SFEN to FEN.
        //
        struct SfenPacker
        {
            // Pack sfen and store in data[32].
            void pack(const chess::Position& pos)
            {
                memset(data, 0, 32 /* 256bit */);
                stream.set_data(data);

                // turn
                // Side to move.
                stream.write_one_bit((int)(pos.sideToMove()));

                // 7-bit positions for leading and trailing balls
                // White king and black king, 6 bits for each.
                stream.write_n_bit(static_cast<int>(pos.kingSquare(chess::Color::White)), 6);
                stream.write_n_bit(static_cast<int>(pos.kingSquare(chess::Color::Black)), 6);

                // Write the pieces on the board other than the kings.
                for (chess::Rank r = chess::rank8; r >= chess::rank1; --r)
                {
                    for (chess::File f = chess::fileA; f <= chess::fileH; ++f)
                    {
                        chess::Piece pc = pos.pieceAt(chess::Square(f, r));
                        if (pc.type() == chess::PieceType::King)
                            continue;
                        write_board_piece_to_stream(pc);
                    }
                }

                // TODO(someone): Support chess960.
                auto cr = pos.castlingRights();
                stream.write_one_bit(contains(cr, chess::CastlingRights::WhiteKingSide));
                stream.write_one_bit(contains(cr, chess::CastlingRights::WhiteQueenSide));
                stream.write_one_bit(contains(cr, chess::CastlingRights::BlackKingSide));
                stream.write_one_bit(contains(cr, chess::CastlingRights::BlackQueenSide));

                if (pos.epSquare() == chess::Square::none()) {
                    stream.write_one_bit(0);
                }
                else {
                    stream.write_one_bit(1);
                    stream.write_n_bit(static_cast<int>(pos.epSquare()), 6);
                }

                stream.write_n_bit(pos.rule50Counter(), 6);

                stream.write_n_bit(pos.fullMove(), 8);

                // Write high bits of half move. This is a fix for the
                // limited range of half move counter.
                // This is backwards compatibile.
                stream.write_n_bit(pos.fullMove() >> 8, 8);

                // Write the highest bit of rule50 at the end. This is a backwards
                // compatibile fix for rule50 having only 6 bits stored.
                // This bit is just ignored by the old parsers.
                stream.write_n_bit(pos.rule50Counter() >> 6, 1);

                assert(stream.get_cursor() <= 256);
            }

            // sfen packed by pack() (256bit = 32bytes)
            // Or sfen to decode with unpack()
            uint8_t *data; // uint8_t[32];

            BitStream stream;

            // Output the board pieces to stream.
            void write_board_piece_to_stream(chess::Piece pc)
            {
                // piece type
                chess::PieceType pr = pc.type();
                auto c = huffman_table[static_cast<int>(pr)];
                stream.write_n_bit(c.code, c.bits);

                if (pc == chess::Piece::none())
                    return;

                // first and second flag
                stream.write_one_bit(static_cast<int>(pc.color()));
            }

            // Read one board piece from stream
            [[nodiscard]] chess::Piece read_board_piece_from_stream()
            {
                int pr = static_cast<int>(chess::PieceType::None);
                int code = 0, bits = 0;
                while (true)
                {
                    code |= stream.read_one_bit() << bits;
                    ++bits;

                    assert(bits <= 6);

                    for (pr = static_cast<int>(chess::PieceType::Pawn); pr <= static_cast<int>(chess::PieceType::None); ++pr)
                        if (huffman_table[pr].code == code
                            && huffman_table[pr].bits == bits)
                            goto Found;
                }
            Found:;
                if (pr == static_cast<int>(chess::PieceType::None))
                    return chess::Piece::none();

                // first and second flag
                chess::Color c = (chess::Color)stream.read_one_bit();

                return chess::Piece(static_cast<chess::PieceType>(pr), c);
            }
        };


        [[nodiscard]] inline chess::Position pos_from_packed_sfen(const PackedSfen& sfen)
        {
            SfenPacker packer;
            auto& stream = packer.stream;
            stream.set_data(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(&sfen)));

            chess::Position pos{};

            // Active color
            pos.setSideToMove((chess::Color)stream.read_one_bit());

            // First the position of the ball
            pos.place(chess::Piece(chess::PieceType::King, chess::Color::White), static_cast<chess::Square>(stream.read_n_bit(6)));
            pos.place(chess::Piece(chess::PieceType::King, chess::Color::Black), static_cast<chess::Square>(stream.read_n_bit(6)));

            // Piece placement
            for (chess::Rank r = chess::rank8; r >= chess::rank1; --r)
            {
                for (chess::File f = chess::fileA; f <= chess::fileH; ++f)
                {
                    auto sq = chess::Square(f, r);

                    // it seems there are already balls
                    chess::Piece pc;
                    if (pos.pieceAt(sq).type() != chess::PieceType::King)
                    {
                        assert(pos.pieceAt(sq) == chess::Piece::none());
                        pc = packer.read_board_piece_from_stream();
                    }
                    else
                    {
                        pc = pos.pieceAt(sq);
                    }

                    // There may be no pieces, so skip in that case.
                    if (pc == chess::Piece::none())
                        continue;

                    if (pc.type() != chess::PieceType::King)
                    {
                        pos.place(pc, sq);
                    }

                    assert(stream.get_cursor() <= 256);
                }
            }

            // Castling availability.
            chess::CastlingRights cr = chess::CastlingRights::None;
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::WhiteKingSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::WhiteQueenSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::BlackKingSide;
            }
            if (stream.read_one_bit()) {
                cr |= chess::CastlingRights::BlackQueenSide;
            }
            pos.setCastlingRights(cr);

            // En passant square. Ignore if no pawn capture is possible
            if (stream.read_one_bit()) {
                chess::Square ep_square = static_cast<chess::Square>(stream.read_n_bit(6));
                pos.setEpSquare(ep_square);
            }

            // Halfmove clock
            std::uint8_t rule50 = stream.read_n_bit(6);

            // Fullmove number
            std::uint16_t fullmove = stream.read_n_bit(8);

            // Fullmove number, high bits
            // This was added as a fix for fullmove clock
            // overflowing at 256. This change is backwards compatibile.
            fullmove |= stream.read_n_bit(8) << 8;

            // Read the highest bit of rule50. This was added as a fix for rule50
            // counter having only 6 bits stored.
            // In older entries this will just be a zero bit.
            rule50 |= stream.read_n_bit(1) << 6;

            pos.setFullMove(fullmove);
            pos.setRule50Counter(rule50);

            assert(stream.get_cursor() <= 256);

            return pos;
        }
    }

    struct CompressedTrainingDataFile
    {
        struct Header
        {
            std::uint32_t chunkSize;
        };

        CompressedTrainingDataFile(std::string path, std::ios_base::openmode om = std::ios_base::app) :
            m_path(std::move(path)),
            m_file(m_path, std::ios_base::binary | std::ios_base::in | std::ios_base::out | om)
        {
            // Necessary for MAC because app mode makes it put the reading
            // head at the end.
            m_file.seekg(0);
        }

        void append(const char* data, std::uint32_t size)
        {
            writeChunkHeader({size});
            m_file.write(data, size);
        }

        [[nodiscard]] bool hasNextChunk()
        {
            if (!m_file)
            {
                return false;
            }

            m_file.peek();
            return !m_file.eof();
        }

        [[nodiscard]] std::vector<unsigned char> readNextChunk()
        {
            auto size = readChunkHeader().chunkSize;
            std::vector<unsigned char> data(size);
            m_file.read(reinterpret_cast<char*>(data.data()), size);
            return data;
        }

    private:
        std::string m_path;
        std::fstream m_file;

        void writeChunkHeader(Header h)
        {
            unsigned char header[8];
            header[0] = 'B';
            header[1] = 'I';
            header[2] = 'N';
            header[3] = 'P';
            header[4] = h.chunkSize;
            header[5] = h.chunkSize >> 8;
            header[6] = h.chunkSize >> 16;
            header[7] = h.chunkSize >> 24;
            m_file.write(reinterpret_cast<const char*>(header), 8);
        }

        [[nodiscard]] Header readChunkHeader()
        {
            unsigned char header[8];
            m_file.read(reinterpret_cast<char*>(header), 8);
            if (header[0] != 'B' || header[1] != 'I' || header[2] != 'N' || header[3] != 'P')
            {
                throw format_error("binpack: chunk header magic mismatch (expected 'BINP')");
            }

            const std::uint32_t size =
                header[4]
                | (header[5] << 8)
                | (header[6] << 16)
                | (header[7] << 24);

            if (size > maxChunkSize)
            {
                throw format_error("binpack: chunk size exceeds maxChunkSize; file may be malformed");
            }

            return { size };
        }
    };

    [[nodiscard]] inline std::uint16_t signedToUnsigned(std::int16_t a)
    {
        std::uint16_t r;
        std::memcpy(&r, &a, sizeof(std::uint16_t));
        if (r & 0x8000)
        {
            r ^= 0x7FFF;
        }
        r = (r << 1) | (r >> 15);
        return r;
    }

    [[nodiscard]] inline std::int16_t unsignedToSigned(std::uint16_t r)
    {
        std::int16_t a;
        r = (r << 15) | (r >> 1);
        if (r & 0x8000)
        {
            r ^= 0x7FFF;
        }
        std::memcpy(&a, &r, sizeof(std::uint16_t));
        return a;
    }

    struct TrainingDataEntry
    {
        chess::Position pos;
        chess::Move move;
        std::int16_t score;
        std::uint16_t ply;
        std::int16_t result;

        [[nodiscard]] bool isValid() const
        {
            return pos.isMoveLegal(move);
        }

        [[nodiscard]] bool isCapturingMove() const
        {
            return pos.pieceAt(move.to) != chess::Piece::none() &&
                   pos.pieceAt(move.to).color() != pos.pieceAt(move.from).color(); // Exclude castling
        }

        [[nodiscard]] bool isInCheck() const
        {
            return pos.isCheck();
        }
    };

    [[nodiscard]] inline TrainingDataEntry packedSfenValueToTrainingDataEntry(const nodchip::PackedSfenValue& psv)
    {
        TrainingDataEntry ret;

        ret.pos = nodchip::pos_from_packed_sfen(psv.sfen);
        ret.move = psv.move.toMove();
        ret.score = psv.score;
        ret.ply = psv.gamePly;
        ret.result = psv.game_result;

        return ret;
    }

    [[nodiscard]] inline nodchip::PackedSfenValue trainingDataEntryToPackedSfenValue(const TrainingDataEntry& plain)
    {
        nodchip::PackedSfenValue ret;

        nodchip::SfenPacker sp;
        sp.data = reinterpret_cast<uint8_t*>(&ret.sfen);
        sp.pack(plain.pos);

        ret.score = plain.score;
        ret.move = nodchip::StockfishMove::fromMove(plain.move);
        ret.gamePly = plain.ply;
        ret.game_result = plain.result;
        ret.padding = 0xff; // for consistency with the .bin format.

        return ret;
    }

    [[nodiscard]] inline bool isContinuation(const TrainingDataEntry& lhs, const TrainingDataEntry& rhs)
    {
        return
            lhs.result == -rhs.result
            && lhs.ply + 1 == rhs.ply
            && lhs.pos.afterMove(lhs.move) == rhs.pos;
    }

    struct PackedTrainingDataEntry
    {
        unsigned char bytes[32];
    };

    [[nodiscard]] inline std::size_t usedBitsSafe(std::size_t value)
    {
        if (value == 0) return 0;
        return chess::util::usedBits(value - 1);
    }

    static constexpr std::size_t scoreVleBlockSize = 4;

    struct PackedMoveScoreListReader
    {
        TrainingDataEntry entry;
        std::uint16_t numPlies;
        unsigned char* movetext;

        PackedMoveScoreListReader(const TrainingDataEntry& entry_, unsigned char* movetext_, std::uint16_t numPlies_) :
            entry(entry_),
            numPlies(numPlies_),
            movetext(movetext_),
            m_lastScore(-entry_.score)
        {

        }

        [[nodiscard]] std::uint8_t extractBitsLE8(std::size_t count)
        {
            if (count == 0) return 0;

            if (m_readBitsLeft == 0)
            {
                m_readOffset += 1;
                m_readBitsLeft = 8;
            }

            const std::uint8_t byte = movetext[m_readOffset] << (8 - m_readBitsLeft);
            std::uint8_t bits = byte >> (8 - count);

            if (count > m_readBitsLeft)
            {
                const auto spillCount = count - m_readBitsLeft;
                bits |= movetext[m_readOffset + 1] >> (8 - spillCount);

                m_readBitsLeft += 8;
                m_readOffset += 1;
            }

            m_readBitsLeft -= count;

            return bits;
        }

        [[nodiscard]] std::uint16_t extractVle16(std::size_t blockSize)
        {
            auto mask = (1 << blockSize) - 1;
            std::uint16_t v = 0;
            std::size_t offset = 0;
            for(;;)
            {
                std::uint16_t block = extractBitsLE8(blockSize + 1);
                v |= ((block & mask) << offset);
                if (!(block >> blockSize))
                {
                    break;
                }

                offset += blockSize;
            }
            return v;
        }

        [[nodiscard]] TrainingDataEntry nextEntry()
        {
            entry.pos.doMove(entry.move);
            auto [move, score] = nextMoveScore(entry.pos);
            entry.move = move;
            entry.score = score;
            entry.ply += 1;
            entry.result = -entry.result;
            return entry;
        }

        [[nodiscard]] bool hasNext() const
        {
            return m_numReadPlies < numPlies;
        }

        [[nodiscard]] std::pair<chess::Move, std::int16_t> nextMoveScore(const chess::Position& pos)
        {
            chess::Move move;
            std::int16_t score;

            const chess::Color sideToMove = pos.sideToMove();
            const chess::Bitboard ourPieces = pos.piecesBB(sideToMove);
            const chess::Bitboard theirPieces = pos.piecesBB(!sideToMove);
            const chess::Bitboard occupied = ourPieces | theirPieces;

            const auto pieceId = extractBitsLE8(usedBitsSafe(ourPieces.count()));
            const auto from = chess::Square(chess::nthSetBitIndex(ourPieces.bits(), pieceId));

            const auto pt = pos.pieceAt(from).type();
            switch (pt)
            {
            case chess::PieceType::Pawn:
            {
                const chess::Rank promotionRank = pos.sideToMove() == chess::Color::White ? chess::rank7 : chess::rank2;
                const chess::Rank startRank = pos.sideToMove() == chess::Color::White ? chess::rank2 : chess::rank7;
                const auto forward = sideToMove == chess::Color::White ? chess::FlatSquareOffset(0, 1) : chess::FlatSquareOffset(0, -1);

                const chess::Square epSquare = pos.epSquare();

                chess::Bitboard attackTargets = theirPieces;
                if (epSquare != chess::Square::none())
                {
                    attackTargets |= epSquare;
                }

                chess::Bitboard destinations = chess::bb::pawnAttacks(chess::Bitboard::square(from), sideToMove) & attackTargets;

                const chess::Square sqForward = from + forward;
                if (!occupied.isSet(sqForward))
                {
                    destinations |= sqForward;
                    if (
                        from.rank() == startRank
                        && !occupied.isSet(sqForward + forward)
                        )
                    {
                        destinations |= sqForward + forward;
                    }
                }

                const auto destinationsCount = destinations.count();
                if (from.rank() == promotionRank)
                {
                    const auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount * 4ull));
                    const chess::Piece promotedPiece = chess::Piece(
                        chess::fromOrdinal<chess::PieceType>(ordinal(chess::PieceType::Knight) + (moveId % 4ull)),
                        sideToMove
                    );
                    const auto to = chess::Square(chess::nthSetBitIndex(destinations.bits(), moveId / 4ull));

                    move = chess::Move::promotion(from, to, promotedPiece);
                    break;
                }
                else
                {
                    auto moveId = extractBitsLE8(usedBitsSafe(destinationsCount));
                    const auto to = chess::Square(chess::nthSetBitIndex(destinations.bits(), moveId));
                    if (to == epSquare)
                    {
                        move = chess::Move::enPassant(from, to);
                        break;
                    }
                    else
                    {
                        move = chess::Move::normal(from, to);
                        break;
                    }
                }
            }
            case chess::PieceType::King:
            {
                const chess::CastlingRights ourCastlingRightsMask =
                    sideToMove == chess::Color::White
                    ? chess::CastlingRights::White
                    : chess::CastlingRights::Black;

                const chess::CastlingRights castlingRights = pos.castlingRights();

                const chess::Bitboard attacks = chess::bb::pseudoAttacks<chess::PieceType::King>(from) & ~ourPieces;
                const std::size_t attacksSize = attacks.count();
                const std::size_t numCastlings = chess::intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

                const auto moveId = extractBitsLE8(usedBitsSafe(attacksSize + numCastlings));

                if (moveId >= attacksSize)
                {
                    const std::size_t idx = moveId - attacksSize;

                    const chess::CastleType castleType =
                        idx == 0
                        && chess::contains(castlingRights, chess::CastlingTraits::castlingRights[sideToMove][chess::CastleType::Long])
                        ? chess::CastleType::Long
                        : chess::CastleType::Short;

                    move = chess::Move::castle(castleType, sideToMove);
                    break;
                }
                else
                {
                    auto to = chess::Square(chess::nthSetBitIndex(attacks.bits(), moveId));
                    move = chess::Move::normal(from, to);
                    break;
                }
                break;
            }
            default:
            {
                const chess::Bitboard attacks = chess::bb::attacks(pt, from, occupied) & ~ourPieces;
                const auto moveId = extractBitsLE8(usedBitsSafe(attacks.count()));
                auto to = chess::Square(chess::nthSetBitIndex(attacks.bits(), moveId));
                move = chess::Move::normal(from, to);
                break;
            }
            }

            score = m_lastScore + unsignedToSigned(extractVle16(scoreVleBlockSize));
            m_lastScore = -score;

            ++m_numReadPlies;

            return {move, score};
        }

        [[nodiscard]] std::size_t numReadBytes()
        {
            return m_readOffset + (m_readBitsLeft != 8);
        }

    private:
        std::size_t m_readBitsLeft = 8;
        std::size_t m_readOffset = 0;
        std::int16_t m_lastScore = 0;
        std::uint16_t m_numReadPlies = 0;
    };

    struct PackedMoveScoreList
    {
        std::uint16_t numPlies = 0;
        std::vector<unsigned char> movetext;

        [[nodiscard]] std::size_t numBytes() const
        {
            return movetext.size();
        }

        void clear(const TrainingDataEntry& e)
        {
            numPlies = 0;
            movetext.clear();
            m_bitsLeft = 0;
            m_lastScore = -e.score;
        }

        void addBitsLE8(std::uint8_t bits, std::size_t count)
        {
            if (count == 0) return;

            if (m_bitsLeft == 0)
            {
                movetext.emplace_back(bits << (8 - count));
                m_bitsLeft = 8;
            }
            else if (count <= m_bitsLeft)
            {
                movetext.back() |= bits << (m_bitsLeft - count);
            }
            else
            {
                const auto spillCount = count - m_bitsLeft;
                movetext.back() |= bits >> spillCount;
                movetext.emplace_back(bits << (8 - spillCount));
                m_bitsLeft += 8;
            }

            m_bitsLeft -= count;
        }

        void addBitsVle16(std::uint16_t v, std::size_t blockSize)
        {
            auto mask = (1 << blockSize) - 1;
            for(;;)
            {
                std::uint8_t block = (v & mask) | ((v > mask) << blockSize);
                addBitsLE8(block, blockSize + 1);
                v >>= blockSize;
                if (v == 0) break;
            }
        }


        void addMoveScore(const chess::Position& pos, chess::Move move, std::int16_t score)
        {
            const chess::Color sideToMove = pos.sideToMove();
            const chess::Bitboard ourPieces = pos.piecesBB(sideToMove);
            const chess::Bitboard theirPieces = pos.piecesBB(!sideToMove);
            const chess::Bitboard occupied = ourPieces | theirPieces;

            const std::uint8_t pieceId = (pos.piecesBB(sideToMove) & chess::bb::before(move.from)).count();
            std::size_t numMoves = 0;
            int moveId = 0;
            const auto pt = pos.pieceAt(move.from).type();
            switch (pt)
            {
            case chess::PieceType::Pawn:
            {
                const chess::Rank secondToLastRank = pos.sideToMove() == chess::Color::White ? chess::rank7 : chess::rank2;
                const chess::Rank startRank = pos.sideToMove() == chess::Color::White ? chess::rank2 : chess::rank7;
                const auto forward = sideToMove == chess::Color::White ? chess::FlatSquareOffset(0, 1) : chess::FlatSquareOffset(0, -1);

                const chess::Square epSquare = pos.epSquare();

                chess::Bitboard attackTargets = theirPieces;
                if (epSquare != chess::Square::none())
                {
                    attackTargets |= epSquare;
                }

                chess::Bitboard destinations = chess::bb::pawnAttacks(chess::Bitboard::square(move.from), sideToMove) & attackTargets;

                const chess::Square sqForward = move.from + forward;
                if (!occupied.isSet(sqForward))
                {
                    destinations |= sqForward;

                    if (
                        move.from.rank() == startRank
                        && !occupied.isSet(sqForward + forward)
                        )
                    {
                        destinations |= sqForward + forward;
                    }
                }

                moveId = (destinations & chess::bb::before(move.to)).count();
                numMoves = destinations.count();
                if (move.from.rank() == secondToLastRank)
                {
                    const auto promotionIndex = (ordinal(move.promotedPiece.type()) - ordinal(chess::PieceType::Knight));
                    moveId = moveId * 4 + promotionIndex;
                    numMoves *= 4;
                }

                break;
            }
            case chess::PieceType::King:
            {
                const chess::CastlingRights ourCastlingRightsMask =
                    sideToMove == chess::Color::White
                    ? chess::CastlingRights::White
                    : chess::CastlingRights::Black;

                const chess::CastlingRights castlingRights = pos.castlingRights();

                const chess::Bitboard attacks = chess::bb::pseudoAttacks<chess::PieceType::King>(move.from) & ~ourPieces;
                const auto attacksSize = attacks.count();
                const auto numCastlingRights = chess::intrin::popcount(ordinal(castlingRights & ourCastlingRightsMask));

                numMoves += attacksSize;
                numMoves += numCastlingRights;

                if (move.type == chess::MoveType::Castle)
                {
                    const auto longCastlingRights = chess::CastlingTraits::castlingRights[sideToMove][chess::CastleType::Long];

                    moveId = attacksSize - 1;

                    if (chess::contains(castlingRights, longCastlingRights))
                    {
                        // We have to add one no matter if it's the used one or not.
                        moveId += 1;
                    }

                    if (chess::CastlingTraits::moveCastlingType(move) == chess::CastleType::Short)
                    {
                        moveId += 1;
                    }
                }
                else
                {
                    moveId = (attacks & chess::bb::before(move.to)).count();
                }
                break;
            }
            default:
            {
                const chess::Bitboard attacks = chess::bb::attacks(pt, move.from, occupied) & ~ourPieces;

                moveId = (attacks & chess::bb::before(move.to)).count();
                numMoves = attacks.count();
            }
            }

            const std::size_t numPieces = ourPieces.count();
            addBitsLE8(pieceId, usedBitsSafe(numPieces));
            addBitsLE8(moveId, usedBitsSafe(numMoves));

            std::uint16_t scoreDelta = signedToUnsigned(score - m_lastScore);
            addBitsVle16(scoreDelta, scoreVleBlockSize);
            m_lastScore = -score;

            ++numPlies;
        }

    private:
        std::size_t m_bitsLeft = 0;
        std::int16_t m_lastScore = 0;
    };


    [[nodiscard]] inline PackedTrainingDataEntry packEntry(const TrainingDataEntry& plain)
    {
        PackedTrainingDataEntry packed;

        auto compressedPos = plain.pos.compress();
        auto compressedMove = plain.move.compress();

        static_assert(sizeof(compressedPos) + sizeof(compressedMove) + 6 == sizeof(PackedTrainingDataEntry));

        std::size_t offset = 0;
        compressedPos.writeToBigEndian(packed.bytes);
        offset += sizeof(compressedPos);
        compressedMove.writeToBigEndian(packed.bytes + offset);
        offset += sizeof(compressedMove);
        std::uint16_t pr = plain.ply | (signedToUnsigned(plain.result) << 14);
        packed.bytes[offset++] = signedToUnsigned(plain.score) >> 8;
        packed.bytes[offset++] = signedToUnsigned(plain.score);
        packed.bytes[offset++] = pr >> 8;
        packed.bytes[offset++] = pr;
        packed.bytes[offset++] = plain.pos.rule50Counter() >> 8;
        packed.bytes[offset++] = plain.pos.rule50Counter();

        return packed;
    }

    [[nodiscard]] inline TrainingDataEntry unpackEntry(const PackedTrainingDataEntry& packed)
    {
        TrainingDataEntry plain;

        std::size_t offset = 0;
        auto compressedPos = chess::CompressedPosition::readFromBigEndian(packed.bytes);
        plain.pos = compressedPos.decompress();
        offset += sizeof(compressedPos);
        auto compressedMove = chess::CompressedMove::readFromBigEndian(packed.bytes + offset);
        plain.move = compressedMove.decompress();
        offset += sizeof(compressedMove);
        plain.score = unsignedToSigned((packed.bytes[offset] << 8) | packed.bytes[offset+1]);
        offset += 2;
        std::uint16_t pr = (packed.bytes[offset] << 8) | packed.bytes[offset+1];
        plain.ply = pr & 0x3FFF;
        plain.pos.setPly(plain.ply);
        plain.result = unsignedToSigned(pr >> 14);
        offset += 2;
        plain.pos.setRule50Counter((packed.bytes[offset] << 8) | packed.bytes[offset+1]);

        return plain;
    }

    struct CompressedTrainingDataEntryWriter
    {
        static constexpr std::size_t chunkSize = suggestedChunkSize;

        CompressedTrainingDataEntryWriter(std::string path, std::ios_base::openmode om = std::ios_base::app) :
            m_outputFile(path, om),
            m_lastEntry{},
            m_movelist{},
            m_packedSize(0),
            m_packedEntries(chunkSize + maxMovelistSize),
            m_isFirst(true)
        {
            m_lastEntry.ply = 0xFFFF; // so it's never a continuation
            m_lastEntry.result = 0x7FFF;
        }

        void addTrainingDataEntry(const TrainingDataEntry& e)
        {
            bool isCont = isContinuation(m_lastEntry, e);
            if (isCont)
            {
                // add to movelist
                m_movelist.addMoveScore(e.pos, e.move, e.score);
            }
            else
            {
                if (!m_isFirst)
                {
                    writeMovelist();
                }

                if (m_packedSize >= chunkSize)
                {
                    m_outputFile.append(m_packedEntries.data(), m_packedSize);
                    m_packedSize = 0;
                }

                auto packed = packEntry(e);
                std::memcpy(m_packedEntries.data() + m_packedSize, &packed, sizeof(PackedTrainingDataEntry));
                m_packedSize += sizeof(PackedTrainingDataEntry);

                m_movelist.clear(e);

                m_isFirst = false;
            }

            m_lastEntry = e;
        }

        ~CompressedTrainingDataEntryWriter()
        {
            if (m_packedSize > 0)
            {
                if (!m_isFirst)
                {
                    writeMovelist();
                }

                m_outputFile.append(m_packedEntries.data(), m_packedSize);
                m_packedSize = 0;
            }
        }

    private:
        CompressedTrainingDataFile m_outputFile;
        TrainingDataEntry m_lastEntry;
        PackedMoveScoreList m_movelist;
        std::size_t m_packedSize;
        std::vector<char> m_packedEntries;
        bool m_isFirst;

        void writeMovelist()
        {
            m_packedEntries[m_packedSize++] = m_movelist.numPlies >> 8;
            m_packedEntries[m_packedSize++] = m_movelist.numPlies;
            if (m_movelist.numPlies > 0)
            {
                std::memcpy(m_packedEntries.data() + m_packedSize, m_movelist.movetext.data(), m_movelist.movetext.size());
                m_packedSize += m_movelist.movetext.size();
            }
        };
    };

    struct CompressedTrainingDataEntryReader
    {
        static constexpr std::size_t chunkSize = suggestedChunkSize;

        CompressedTrainingDataEntryReader(std::string path, std::ios_base::openmode om = std::ios_base::app) :
            m_inputFile(path, om),
            m_chunk(),
            m_movelistReader(std::nullopt),
            m_offset(0),
            m_isEnd(false)
        {
            if (!m_inputFile.hasNextChunk())
            {
                m_isEnd = true;
            }
            else
            {
                m_chunk = m_inputFile.readNextChunk();
            }
        }

        [[nodiscard]] bool hasNext()
        {
            return !m_isEnd;
        }

        [[nodiscard]] TrainingDataEntry next()
        {
            if (m_movelistReader.has_value())
            {
                const auto e = m_movelistReader->nextEntry();

                if (!m_movelistReader->hasNext())
                {
                    m_offset += m_movelistReader->numReadBytes();
                    m_movelistReader.reset();

                    fetchNextChunkIfNeeded();
                }

                return e;
            }

            PackedTrainingDataEntry packed;
            std::memcpy(&packed, m_chunk.data() + m_offset, sizeof(PackedTrainingDataEntry));
            m_offset += sizeof(PackedTrainingDataEntry);

            const std::uint16_t numPlies = (m_chunk[m_offset] << 8) | m_chunk[m_offset + 1];
            m_offset += 2;

            const auto e = unpackEntry(packed);

            if (numPlies > 0)
            {
                m_movelistReader.emplace(e, reinterpret_cast<unsigned char*>(m_chunk.data()) + m_offset, numPlies);
            }
            else
            {
                fetchNextChunkIfNeeded();
            }

            return e;
        }

    private:
        CompressedTrainingDataFile m_inputFile;
        std::vector<unsigned char> m_chunk;
        std::optional<PackedMoveScoreListReader> m_movelistReader;
        std::size_t m_offset;
        bool m_isEnd;

        void fetchNextChunkIfNeeded()
        {
            if (m_offset + sizeof(PackedTrainingDataEntry) + 2 > m_chunk.size())
            {
                if (m_inputFile.hasNextChunk())
                {
                    m_chunk = m_inputFile.readNextChunk();
                    m_offset = 0;
                }
                else
                {
                    m_isEnd = true;
                }
            }
        }
    };

}} // namespace binpack::detail
