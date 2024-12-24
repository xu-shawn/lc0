ifndef EXE
	EXE = lc0
endif

all:
	bash build.sh && mv build/release/lc0 $(EXE)
