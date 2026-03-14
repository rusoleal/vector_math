# Changelog

## [0.1.0] - 2026-03-14

### Fixed
- `Matrix4d` AVX intrinsics: wrong register type `__m256` (8×float) replaced with `__m256d` (4×double) throughout
- `Matrix4d` matrix multiply: `_mm256_add_ps` typo corrected to `_mm256_add_pd`
- `Matrix4d` matrix-vector multiply: `_mm256_hadd_pd` reduction now correctly combines 128-bit lanes using `_mm256_extractf128_pd` + `_mm_add_pd`, fixing wrong results on x86/x64
- `Vec` `operator*(scalar, vec)`: friend declaration did not match external template definition, causing a linker error; fixed by defining it inline in the class body
- `Vec` `operator/(vec, scalar)`: declared as both a member and a free friend function, causing call ambiguity; removed the duplicate free function
- `Matrix4` `scale` and `perspective`: `Matrix4<T>(0)` was ambiguous between value and pointer constructor; fixed to `Matrix4<T>(DATA_TYPE(0))`
- `Matrix4` `ortho`: called non-existent `CreateOrthoOffCenter`; corrected to `orthoOffCenter`
- `Matrix4` `ortho`: passed width/height arguments to `orthoOffCenter` in wrong positions, causing `left == right` (division by zero); fixed argument order

### Added
- ARM NEON implementation for `Matrix4f` multiply (scalar fallback removed for float path)
- `Matrix4d` and `Matrix4f` SIMD-accelerated multiply operators
- `alignas(16)` on `Matrix4f` and `Vector4f` for correct SSE alignment
- `Matrix4` transpose operation
- Benchmarks with Release mode build support and GLM comparison

### Changed
- SIMD architecture detection consolidated in `common.hpp`

---

## [0.0.2] - prior

### Added
- Google Test and Google Benchmark integration via CMake FetchContent
- `Matrix4` benchmark suite
- GitHub Actions CI for Ubuntu, Windows, and macOS

### Fixed
- `cmath` include and SIMD conditional compilation

---

## [0.0.1] - initial release

- Generic `Vec<T, N>` and `Mat<T, R, C>` templates
- `Vector2`, `Vector3`, `Vector4`, `Matrix2`, `Matrix3`, `Matrix4` typed specializations
- `Quaternion<T>`
- CMake build system with C++17
