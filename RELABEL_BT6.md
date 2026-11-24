# Relabeling binpacks with BT6

This branch relabels Stockfish training binpacks with the value of the Leela Chess Zero network
[BT6-1024x24x32h-swa-7000000](https://data.lczero.org/files/networks/contrib/BT6/). It uses `cuda-nx`, a CUDA
backend written for this network's architecture.

For each input binpack the relabeler writes two files with the same entries (positions, moves, results and skip marks
unchanged):
- **`out.binpack`** carries the score as centipawns, `cp = 660.6 q / (1 - 0.9751875 q^10)`.
- **`out.q.binpack`** carries the raw value as `q * 32000`.

Only the WDL value head is evaluated (q = W - L); policy is skipped.

## Requirements

- An NVIDIA GPU with tensor cores, sm_75 (Turing) or newer. The kernels were tuned on a Quadro RTX 6000; they use
  mma.m16n8k8 fp16 instructions, ldmatrix and up to 64 KB of shared memory per block.
- CUDA 12.x, gcc 11 or newer (the tree is C++20), meson and ninja.
- The network file `BT6-1024x24x32h-swa-7000000.pb.gz` (937 MB) from the link above.

## 1. Build

```sh
CUDA=/usr/local/cuda   # your CUDA root
CC=gcc-11 CXX=g++-11 meson setup build-nx --buildtype=release -Dplain_cuda=true -Dcudnn=false \
  -Dnative_cuda=false -Dcc_cuda=75 -Dblas=false -Dopenblas=false -Donednn=false -Ddnnl=false -Dopencl=false \
  -Dgtest=false -Drescorer=false -Donnx=false -Dcudnn_include=$CUDA/include -Dcudnn_libdirs=$CUDA/lib64 \
  -Dnvcc_ccbin=g++-11
ninja -C build-nx lc0
```

- `-Dcc_cuda` selects the GPU architecture, e.g. 86 for Ampere consumer cards or 89 for Ada.
- meson downloads its subprojects on the first setup.
- Optionally add `"-Dcpp_link_args=['-static-libstdc++','-static-libgcc']"` for a binary that runs on machines with an
  older libstdc++.

## 2. Relabel

```sh
export LC0_BACKEND=cuda-nx
./build-nx/lc0 BT6-1024x24x32h-swa-7000000.pb.gz in.binpack out.binpack 1024
```

The backend reads the weights from the network file and rearranges them into its own layouts at startup.

- **Batch size.** Pass 1024: it is the backend's batch, and the result cache only works at that size. The default is
  128.
- **Output.** A JSON summary goes to stdout: entries, skipped entries, history mode, seconds and positions/s.
- **History.** By default each position is evaluated with its real 8-position history, rebuilt from the binpack's game
  chains. Set `LC0_RELABEL_HISTORY=fen` for data without games, such as sampled search-tree positions.
- **Skip marks.** Entries carrying the skip mark of minimized binpacks (score 32002) pass through unevaluated.
- **Faster startup.** Reading the network file adds about 15 s per run. For many short jobs, write the rearranged
  weights once and load them instead:
  1. Run once with `LC0_NX_WRITE_BLOB=nx16.blob`, which writes the ~480 MB file `nx16.blob`. Alternatively, run
     `scripts/nx/nx_export.py`, which writes the same file.
  2. Make a copy of the network file without weights with `scripts/nx/strip_weights.py`. lc0 only reads the input
     format from it.
  3. Run with `LC0_NX_BLOB=nx16.blob` and that copy.
- **Splitting across GPUs.** Binpack chunks (`BINP` + u32 size + data) are independent. To use several GPUs, cut a file
  at chunk boundaries, relabel the parts separately and concatenate the outputs in order. Every part should hold whole
  chunks; parts of 24 MB take about 75 minutes on an RTX 6000.

| Variable | Effect |
|---|---|
| `LC0_RELABEL_HISTORY=fen` | Evaluate from the FEN alone (no history). |
| `LC0_RELABEL_QF32=file` | Also write the raw q of every entry as little-endian float32 (NaN for skipped entries). |
| `LC0_RELABEL_MAX_ENTRIES=N` | Stop after N entries. |
| `LC0_RELABEL_SYNC=1` | Turn off the pipeline: by default a second thread evaluates a batch while the main thread writes the previous one and reads the next. |
| `LC0_NX_CACHE_BITS=b` | Result cache with 2^b entries; the default 24 means 16M entries and 512 MB of host RAM. Identical inputs (all 112 planes) reuse their first result. 0 disables it. |
| `LC0_RELABEL_MIN_PLY=N` | Do not evaluate unmarked entries with ply < N. They get the skip mark in both outputs (NaN in q.f32) and stay in the chains for later positions' history. Off by default. |
| `LC0_RELABEL_FILTERED=1` | The same for entries whose move captures or whose side to move is in check, as nnue-pytorch's `filtered` drops them. Search-tree samples (ply 9999) are exempt. Off by default. |
| `LC0_NX_BLOB=file` | Load pre-arranged weights instead of reading them from the network file. |
| `LC0_NX_WRITE_BLOB=file` | Write the weights read from the network file in that format. |
| `LC0_NX_PROFILE=1`, `LC0_NX_EDGE_PROFILE=1` | Per-stage GPU times, printed every few batches. |

## Speed and accuracy

On a Quadro RTX 6000 (Turing, power-capped at 250 W), one batch of 1024 positions takes about 572 ms. In production
jobs that is about 1,700–1,750 positions/s per GPU on search-tree data and 2,150–2,250 entries/s on game binpacks.
Game data is faster per entry because skip-marked entries are not evaluated and repeated positions come from the cache.

Compared with ONNX Runtime fp32 on the original graph (30,000 positions each):

| Data | mean \|Δq\| | p99 | p99.9 | max | > 0.01 |
|---|---|---|---|---|---|
| search-tree positions, no history | 1.04e-4 | 0.0010 | 0.0038 | 0.10 | 8 |
| test80 game positions, chain history | 9.9e-5 | 0.0009 | 0.0032 | 0.0086 | 0 |

The few larger differences come from numerically chaotic positions, where any rounding is amplified by about 1000×.
`scripts/nx/nxref.py` is an fp64 NumPy reference of the value path, for checking other builds. The Python tools in
`scripts/nx` need `numpy`, `protobuf` and `net_pb2.py`; generate it with
`python3 -m grpc_tools.protoc -I proto --python_out=scripts/nx proto/net.proto`.

## How it works

- **Precision.** Mixed: fp16 operands for every GEMM and for attention, fp32 accumulation, and an fp32 residual stream
  (it reaches about ±22,600). Layer-norm statistics, logits, softmax and the door terms stay in fp32.
- **GEMMs.** Persistent tensor-core kernels with fused epilogues: SwiGLU, and an fp32 residual read-modify-write with
  whole-sector access and staggered block starts. A startup autotune picks the tile configuration.
- **Attention.**
  - A persistent kernel computes two heads per block, with Q/K/V and the attack-gate projections on tensor cores.
  - The E/M/G door terms are computed on tensor cores from the 16 fp16 edge features, pipelined one key chunk ahead.
  - The static pair stream is folded out. Its rms factor is packed into the attack word, and the pair term is
    `rp · (tz[rel] + Σ pz[c])`, from per-layer tables.
- **Edge stream.** Each update runs as a logit readback, then projections with a fused softmax, then batched cuBLAS
  triplet products, then a fused tensor-core FFN.
