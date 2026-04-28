CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
INCLUDES := -I../include

INPUT  ?= ../../small.binpack
OUTPUT ?= first_ten.binpack

.PHONY: all run clean

all: print_first_ten

print_first_ten: print_first_ten.cpp ../include/binpack/binpack.hpp ../include/binpack/detail.hpp ../include/binpack/chess.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ print_first_ten.cpp

run: print_first_ten
	./print_first_ten $(INPUT) $(OUTPUT)

clean:
	rm -f print_first_ten first_ten.binpack roundtrip.binpack
