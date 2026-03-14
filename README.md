# vector_math

A C++17 vector and matrix mathematics library with SIMD acceleration for x86/x64 (SSE/AVX) and ARM (NEON).

## Features

- Generic vector and matrix templates (`Vec<T, N>`, `Mat<T, R, C>`)
- Typed specializations: `Vector2`, `Vector3`, `Vector4`, `Matrix2`, `Matrix3`, `Matrix4`
- SIMD-optimized `Matrix4f` (SSE) and `Matrix4d` (AVX) for performance-critical paths
- `Quaternion<T>` for rotations
- Namespace: `systems::leal::vector_math`

## Requirements

- CMake 3.22.1+
- C++17 compiler (GCC, Clang, MSVC)

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
| ARM | NEON | scalar fallback (in progress) |

`Vector4f` and `Matrix4f` use `alignas(16)` to satisfy SIMD alignment requirements.
