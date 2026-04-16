# vector_math

A C++20 vector and matrix mathematics library with SIMD acceleration for x86/x64 (SSE/AVX) and ARM (NEON).

## 🚀 Part of the Campello Engine

This project is a module within the **Campello** ecosystem.

👉 Main repository: https://github.com/rusoleal/campello

Campello is a modular, composable game engine built as a collection of independent libraries.
Each module is designed to work standalone, but integrates seamlessly into the engine runtime.

## Features

- Generic vector and matrix templates (`Vec<T, N>`, `Mat<T, R, C>`)
- Typed specializations: `Vector2`, `Vector3`, `Vector4`, `Matrix2`, `Matrix3`, `Matrix4`
- SIMD-optimized `Matrix4f` (SSE) and `Matrix4d` (AVX) for performance-critical paths
- `Quaternion<T>` with Hamilton product, slerp, axis-angle, and vector rotation
- Rich vector API: dot, cross, distance, angle, reflect, lerp, clamp, floor/ceil/round, component-wise ops
- Rich matrix API: determinant, inverse, TRS compose/decompose helpers, row/column access
- Full Doxygen documentation on all public APIs
- Namespace: `systems::leal::vector_math`

## Requirements

- CMake 3.22.1+
- C++20 compiler (GCC, Clang, MSVC)

## Building

```bash
cmake -B build -S .
make -C build
```

### Tests

```bash
./launch_test.sh
# or
cmake -B build/test -S . -DVECTOR_MATH_BUILD_TEST=ON
make -C build/test
build/test/vector_math_test
```

### Benchmarks

```bash
./launch_benchmark.sh
# or
cmake -B build/benchmark -S . -DVECTOR_MATH_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
make -C build/benchmark
build/benchmark/vector_math_benchmark
```

## SIMD Support

Architecture is detected automatically at compile time:

| Architecture | Intrinsics | Types accelerated |
|---|---|---|
| x86/x64 | SSE / AVX (`-mavx`) | `Matrix4f` (SSE), `Matrix4d` (AVX) |
| AArch64 | NEON | `Matrix4f` (NEON), `Matrix4d` (NEON) |
| ARM 32-bit | — | `Matrix4f` (scalar fallback), `Matrix4d` (scalar fallback) |

`Vector4f` and `Matrix4f` use `alignas(16)` to satisfy SIMD alignment requirements.
