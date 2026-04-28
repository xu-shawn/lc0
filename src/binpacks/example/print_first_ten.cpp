// print_first_ten — read up to 10 entries from a .binpack file, print each
// labeled record to stdout, and write the same entries to a fresh .binpack.
//
// Usage: ./print_first_ten [input.binpack [output.binpack]]
// Defaults: input=../../small.binpack, output=first_ten.binpack

#include <binpack/binpack.hpp>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_entry(std::size_t index, const binpack::Entry& e) {
    std::cout << "Entry " << index << ":\n"
              << "  fen:      " << e.fen      << '\n'
              << "  move_uci: " << e.move_uci << '\n'
              << "  score:    " << e.score    << '\n'
              << "  ply:      " << e.ply      << '\n'
              << "  result:   " << static_cast<int>(e.result) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const std::string input  = (argc > 1) ? argv[1] : "../../small.binpack";
    const std::string output = (argc > 2) ? argv[2] : "first_ten.binpack";

    try {
        binpack::Reader reader(input);
        binpack::Writer writer(output);

        constexpr std::size_t limit = 10;
        std::size_t printed = 0;

        for (const auto& entry : reader) {
            print_entry(printed, entry);
            writer.write(entry);
            if (++printed == limit) break;
        }

        std::cout << "\nWrote " << printed << " entries to " << output << '\n';
        return EXIT_SUCCESS;
    } catch (const binpack::format_error& ex) {
        std::cerr << "binpack format error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    } catch (const std::system_error& ex) {
        std::cerr << "I/O error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
