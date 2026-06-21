#include <binpack/binpack.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <input.binpack> <output.binpack> <count>\n";
        return 1;
    }
    const std::string in = argv[1], out = argv[2];
    const std::size_t n = std::stoull(argv[3]);
    binpack::Reader reader(in);
    binpack::Writer writer(out);
    std::size_t i = 0;
    for (const auto& e : reader) {
        writer.write(e);
        if (++i == n) break;
    }
    std::cout << "wrote " << i << " entries to " << out << "\n";
}
