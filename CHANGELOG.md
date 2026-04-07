# Changelog

## [0.3.5] - 2026-04-07

### Fixed
- **`Matrix4f` ARMv7 build failure** — the ARM NEON paths in `matrix4f.hpp` used AArch64-only intrinsics (`vfmaq_laneq_f32`, `vpaddq_f32`) unconditionally for all ARM targets. Added `&& (defined(__arm64__) || defined(__aarch64__))` guards to both the matrix–matrix and matrix–vector multiply operators, mirroring the existing pattern in `matrix4d.hpp`. ARMv7 (`armeabi-v7a`) now falls back to the scalar implementation.

### Added
- **Android CI** — new `android-build` job in `cmake-multi-platform.yml` cross-compiles the library and test suite for all four Android ABIs (`armeabi-v7a`, `arm64-v8a`, `x86`, `x86_64`) in both `Release` and `Debug` using the NDK toolchain (`android-21`). Tests are compiled but not executed (no Android runtime on the host runner).

---

## [0.3.4] - 2026-03-18

### Fixed
- **`M_PI` still undeclared on MSVC in `test/main.cpp`** — `#define _USE_MATH_DEFINES` + `#include <cmath>` were placed *after* `#include <gtest/gtest.h>`, which pulls in `<cmath>` transitively before the define was seen. Moved the define and `<cmath>` above the gtest include so MSVC exposes `M_PI` before the include-guard prevents a second parse.

---

## [0.3.3] - 2026-03-18

### Fixed
- **`_USE_MATH_DEFINES` propagated to library headers** — `common.hpp` and `vec.hpp` now define `_USE_MATH_DEFINES` before `<cmath>` under `_WIN32`, so any consumer compiling on MSVC gets `M_PI` and other POSIX math constants without needing to set the macro themselves. Previously this guard was only applied in `test/main.cpp`.

---

## [0.3.2] - 2026-03-18

### Fixed
- **Windows CI (`M_PI` undeclared)** — added `#define _USE_MATH_DEFINES` before `<cmath>` under `_WIN32` so MSVC exposes `M_PI` and other POSIX math constants in `test/main.cpp`

---

## [0.3.1] - 2026-03-18

### Fixed
- **`Matrix4f` ARM NEON matrix–matrix multiply** — `this` rows and `rhs` rows were swapped in the accumulation loop, producing wrong results on AArch64 (Apple Silicon / `ubuntu-24.04-arm`). Fixed to match the x86 SSE logic: load `rhs` rows as vectors and broadcast scalars from each row of `this`.

---

## [0.3.0] - 2026-03-18

### Added
- **`Vector4d` SIMD arithmetic** — all operators now have AVX (x86) and NEON (AArch64) paths instead of falling back to the scalar `Vector4<double>` base class:
  - `operator+` / `operator-`: `_mm256_add/sub_pd` · `vaddq/vsubq_f64`
  - unary `operator-`: `_mm256_xor_pd` with sign-bit mask · `vnegq_f64`
  - `operator*(scalar)` / `operator/(scalar)`: `_mm256_mul/div_pd` · `vmulq_n_f64`
  - `dot()` (instance + static): AVX hadd trick (`_mm256_hadd_pd` + 128-bit extract) · `vpaddq_f64` + `vaddvq_f64`
- **`Matrix4d` SIMD add / subtract / negate / scalar-multiply** — new overrides replace the 16-element scalar loop with 4 AVX 256-bit ops (one per row) or 8 NEON 128-bit ops:
  - `operator+(Matrix4d)` / `operator-(Matrix4d)`: `_mm256_add/sub_pd` · `vaddq/vsubq_f64`
  - unary `operator-()`: `_mm256_xor_pd` · `vnegq_f64`
  - `operator*(double)`: `_mm256_mul_pd` + `_mm256_set1_pd` · `vmulq_n_f64`
- **FMA3 in `Matrix4d` mat×mat and mat×vec** (x86) — `mul + add` pairs replaced with `_mm256_fmadd_pd` when compiled with `-mfma` (`__FMA__` defined); reduces 7 instructions to 4 per accumulation step
- **`Matrix4f * Vector4f` ARM NEON** — previously fell back to scalar; now uses two `vpaddq_f32` passes to compute all four dot products simultaneously
- **`-mfma` compiler flag** added to the x86 CMake path (gcc/clang: `-mavx -mfma`; MSVC: `/arch:AVX2`)
- Benchmarks for all new operations: `BM_Vector4dSIMDAdd`, `BM_Vector4dSIMDScalarMultiply`, `BM_Vector4dSIMDDot`, `BM_Matrix4dSIMDAdd`, `BM_Matrix4dSIMDScalarMultiply`, `BM_Matrix4fByVectorGeneric`, with scalar and GLM baselines
- **AArch64 NEON** full implementation for `Matrix4d` matrix–matrix and matrix–vector multiply (`float64x2_t`, `vfmaq_f64`, `vpaddq_f64`)
- `Matrix4d::lookAt` optimized inline override — avoids generic `Vec<>` loop overhead and intermediate temporaries
- CMake **install support**: `GNUInstallDirs`, `CMakePackageConfigHelpers`, package config files (`vector_mathConfig.cmake`, `vector_mathConfigVersion.cmake`), and `INSTALL_INTERFACE` include paths
- Benchmark suite expanded: `Matrix4f`, `Matrix4d`, `Quaternion`, and GLM comparison benchmarks; removed dummy `BM_StringCreation`

### Changed
- `Matrix4d` and `Matrix4f` implementations moved from `.cpp` translation units to **header-only inline** methods — `src/matrix4d.cpp` and `src/matrix4f.cpp` removed
- `Matrix4::identity()` now returns a cached `static const` instance (computed once via IIFE) instead of allocating a local array on every call
- CI: added **`ubuntu-24.04-arm`** runner (AArch64 NEON coverage); added **Debug** build type alongside Release; `ctest` now runs with `--output-on-failure`

---

## [0.2.0] - 2026-03-17

### Added

**Vec\<T, N\> (base class)**
- `dot(rhs)` — instance and static dot product
- `operator*(const Vec&)` — component-wise (Hadamard) multiplication
- `distanceTo(other)` / `distanceToSquared(other)` — Euclidean distance
- `setZero()` / `setFrom(other)` — mutation helpers
- `absolute()` — in-place per-component `abs`
- `floor()` / `ceil()` / `round()` — in-place per-component rounding
- `clamp(min, max)` / `clamp(minVec, maxVec)` — scalar and per-component clamping
- `static min(a, b)` / `static max(a, b)` — component-wise min/max
- `static lerp(a, b, t)` — linear interpolation

**Vector2\<T\>**
- Named accessors `x()` / `y()` returning references
- `dot()` instance + static
- `cross(other)` — 2D scalar cross product
- `angleTo(other)` — unsigned angle in radians [0, π]
- `angleToSigned(other)` — signed angle in radians (-π, π]
- `reflect(normal)` / `reflected(normal)` — reflection about a normal
- Typed `operator+`, `-`, `*`, `/` overrides (return `Vector2` instead of `Vec`)

**Vector3\<T\>**
- Named accessors `x()` / `y()` / `z()` returning references
- `angleTo(other)` — unsigned angle in radians
- `angleToSigned(other, normal)` — signed angle with reference normal
- `reflect(normal)` / `reflected(normal)` — reflection about a normal
- `normalizeInto(out)` — normalized copy into an output parameter
- Typed `operator+`, `-`, `*`, `/` overrides (return `Vector3`)

**Vector4\<T\>**
- Named accessors `x()` / `y()` / `z()` / `w()` returning references
- `dot()` instance + static
- `setValues(x, y, z, w)` — convenience setter
- Typed operator overrides (return `Vector4`)

**Quaternion\<T\>**
- `static identity()` — no-rotation quaternion (0, 0, 0, 1)
- `static axisAngle(axis, angle)` — construct from axis-angle
- `static slerp(a, b, t)` — spherical linear interpolation
- `operator*` — Hamilton product (quaternion composition)
- `operator+` / `operator-` / unary `operator-` / `operator*(scalar)`
- `setAxisAngle(axis, angle)`
- `conjugate()` / `conjugated()`
- `inverse()` / `inverted()`
- `rotate(v&)` / `rotated(v)` — rotate a `Vector3` by this quaternion
- `axis()` — extract normalized rotation axis
- `angle()` — extract rotation angle in radians

**Matrix4\<T\>**
- `operator+` / `operator-` / unary `operator-`
- `trace()` — sum of diagonal elements
- `determinant()` — 4×4 determinant via cofactor expansion
- `invert()` — in-place inversion; returns `false` if singular
- `inverted()` — returns an inverted copy
- `getRow(i)` / `setRow(i, v)` — row access by index
- `getColumn(i)` / `setColumn(i, v)` — column access by index
- `getTranslation()` / `setTranslation(v)` — translation component
- `getScale()` — extract TRS scale factors (column lengths)
- `transform3(v)` — transform a point (w = 1, includes translation)
- `rotate3(v)` — transform a direction (w = 0, no translation)
- `static compose(translation, rotation, scale)` — build a TRS matrix

### Fixed
- `Matrix4::rotateZ` had identical code to `rotateY` (wrong column indices); corrected to rotate in the X–Y plane
- `Matrix4::rotateX`, `rotateY`, `rotateZ`: local variables `cos`/`sin` shadowed standard library functions; renamed to `cosA`/`sinA`

### Changed
- All public headers now carry Doxygen `///` documentation covering every class, constructor, operator, and method

---

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
