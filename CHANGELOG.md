# Changelog

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
