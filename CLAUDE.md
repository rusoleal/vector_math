# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build (library only)
```bash
cmake -B build -S .
make -C build
```

### Build and run tests
```bash
./launch_test.sh
# Or manually:
cmake -B build/test -S . -DVECTOR_MATH_BUILD_TEST=ON
make -C build/test
build/test/vector_math_test
```

### Run a single test
```bash
build/test/vector_math_test --gtest_filter="TestSuiteName.TestName"
```

### Build and run benchmarks
```bash
./launch_benchmark.sh
# Or manually (Release mode for accurate perf):
cmake -B build/benchmark -S . -DVECTOR_MATH_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
make -C build/benchmark
build/benchmark/vector_math_benchmark
```

## Architecture

**Namespace:** `systems::leal::vector_math`

**Headers in `inc/vector_math/`; implementations in `src/`.**

### Type hierarchy

- `Vec<DATA_TYPE, SIZE>` and `Mat<DATA_TYPE, ROWS, COLS>` — generic templates in `vec.hpp` / `mat.hpp`
- `Vector2<T>`, `Vector3<T>`, `Vector4<T>` and `Matrix2<T>`, `Matrix3<T>`, `Matrix4<T>` — typed specializations
- `Vector4f`, `Vector4d` — aligned (`alignas(16)`) float/double vector4, SIMD-ready
- `Matrix4f`, `Matrix4d` — aligned float/double 4×4 matrices with SIMD implementations in `src/matrix4f.cpp` and `src/matrix4d.cpp`
- `Quaternion<T>` — rotation quaternion in `quaternion.hpp`

### SIMD strategy (common.hpp)

Architecture is detected at compile time:
- `__VECTOR_MATH_ARCH_X86_X64` → `<immintrin.h>` (SSE/AVX). AVX is force-enabled via `-mavx` / `/arch:AVX` in CMake.
- `__VECTOR_MATH_ARCH_ARM` → `<arm_neon.h>` (NEON)

`Matrix4f` uses SSE 128-bit intrinsics (4×float). `Matrix4d` uses AVX 256-bit intrinsics (4×double). ARM paths currently fall back to scalar operations.

### Known issues

- `matrix4d` AVX implementation is broken (commit `f7bf612`). The scalar fallback is used on ARM; the AVX path may mix `_mm256_add_ps` (32-bit) with `_mm256_add_pd` (64-bit) incorrectly.

### Dependencies (auto-fetched by CMake via FetchContent)

- Google Test v1.17.0 — unit tests
- Google Benchmark v1.9.4 — benchmarks
- GLM — used in benchmarks for comparison
