// nibble's CUDA kernels, compiled offline into the committed nibble.fatbin.
// Normal cargo builds only embed that artifact; they need no CUDA toolkit.
//
// Rebuild from this directory with CUDA 13.4:
//
// nvcc -std=c++17 -O3 -use_fast_math --fatbin kernels.cu -o nibble.fatbin ^
//   -gencode arch=compute_75,code=sm_75 ^
//   -gencode arch=compute_80,code=sm_80 ^
//   -gencode arch=compute_86,code=sm_86 ^
//   -gencode arch=compute_89,code=sm_89 ^
//   -gencode arch=compute_90,code=sm_90 ^
//   -gencode arch=compute_100,code=sm_100 ^
//   -gencode arch=compute_120,code=sm_120 ^
//   -gencode arch=compute_120,code=compute_120
//
// (On POSIX shells use backslashes instead of Windows' caret continuations.
// On Windows nvcc needs MSVC's cl.exe: run it from an "x64 Native Tools"
// prompt, or after `vcvars64.bat` from the Visual Studio Build Tools.)
//
// The kernels reproduce ggml-cuda's rounding (b12675), so they are built as
// ggml-cuda is: with -use_fast_math (flush-to-zero, approximate division and
// square roots, and expf, sinf, cosf, powf and logf as their intrinsics).
// Where the order in which nvcc fuses a multiply-add decides the result, the
// source writes it with fmaf as nvcc fused ggml's expression; changing the
// compiler version can change that, so recheck the logits against
// llama.cpp's CUDA build after upgrading nvcc.

#include <math_constants.h>
#include "matvec.cu"
#include "ops.cu"
#include "attention.cu"
#include "delta.cu"
