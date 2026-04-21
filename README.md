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
- SIMD-optimized float paths by default on x86/x64 (SSE) and optional AVX2-accelerated double paths on x86/x64
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

Enable AVX2 explicitly on x86/x64 when you want the wider double-precision SIMD paths:

```bash
cmake -B build -S . -DVECTOR_MATH_ENABLE_AVX2=ON
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
| x86/x64 | SSE by default, AVX2+FMA optional via `-DVECTOR_MATH_ENABLE_AVX2=ON` | `Vector4f`, `Vector3f`, `Matrix4f` by default; `Vector4d`, `Vector3d`, `Matrix4d` use AVX when enabled |
| AArch64 | NEON | `Matrix4f` (NEON), `Matrix4d` (NEON) |
| ARM 32-bit | — | `Matrix4f` (scalar fallback), `Matrix4d` (scalar fallback) |

`Vector3f`/`Vector4f`/`Matrix4f` use `alignas(16)`. `Vector3d`/`Vector4d`/`Matrix4d` use `alignas(32)` for AVX-capable layouts.
