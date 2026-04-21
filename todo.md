# SIMD Optimization Roadmap

> Analysis of `vector_math` for SIMD acceleration opportunities on **x86/x64 (SSE/AVX)** and **ARM NEON (AArch64)**.

---

## Legend

| Priority | Meaning |
|----------|---------|
| **P0** | High impact, straightforward implementation, hot paths |
| **P1** | Good impact, moderate complexity |
| **P2** | Nice-to-have, complex or limited speedup |

---

## P0 — Critical Gaps (Highest Impact)

### 1. `Vector4f` — Add Full SIMD Overrides
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/vector4f.hpp` with SSE/NEON/scalar paths for arithmetic and `dot`.  
**Why:** 4 floats fit perfectly in one 128-bit SSE/NEON register. This is the most glaring omission.

| Method | x86/x64 (SSE) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator+` | `_mm_add_ps` | `vaddq_f32` |
| `operator-` | `_mm_sub_ps` | `vsubq_f32` |
| `operator-()` | `_mm_xor_ps` (sign flip) | `vnegq_f32` |
| `operator* scalar` | `_mm_mul_ps` + broadcast | `vmulq_n_f32` |
| `operator/ scalar` | `_mm_div_ps` or `*rcp` | `vmulq_n_f32` (precompute `1/scalar`) |
| `dot` | `_mm_mul_ps` → `_mm_hadd_ps` (SSE3) or shuffle+add | `vmulq_f32` → `vaddvq_f32` (AArch64) |
| `lengthSquared` | reuse `dot` implementation | reuse `dot` implementation |
| `normalize` / `normalized` | SIMD `lengthSquared` + scalar `sqrt` + SIMD `* invLen` | same pattern |

**Files:** `inc/vector_math/vector4f.hpp`
**Reference pattern:** `vector4d.hpp`

---

### 2. `Matrix4f` — Add Missing Component-Wise SIMD Operators
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/matrix4f.hpp` for `+`, `-`, unary `-`, and `* scalar` with SSE/NEON/scalar paths.  
**Why:** Matrix4d already has these; Matrix4f should be consistent. 16 floats = 4 rows of SSE/NEON ops.

| Method | x86/x64 (SSE) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator+` | 4× `_mm_add_ps` | 4× `vaddq_f32` |
| `operator-` | 4× `_mm_sub_ps` | 4× `vsubq_f32` |
| `operator-()` | 4× `_mm_xor_ps` | 4× `vnegq_f32` |
| `operator* scalar` | 4× `_mm_mul_ps` (broadcast scalar) | 4× `vmulq_n_f32` |

**Files:** `inc/vector_math/matrix4f.hpp`
**Reference pattern:** `matrix4d.hpp`

---

### 3. `Matrix4f::lookAt` — Scalar-Fast Override
**Status:** Done  
**Current state:** Implemented as a scalar-fast override in `inc/vector_math/matrix4f.hpp`.  
**Why:** `Matrix4d` has an optimized inline scalar override that avoids all generic `Vec<>` loop overhead. Same optimization applies here.

**Approach:** Port the `Matrix4d::lookAt` scalar-fast implementation to `Matrix4f`.

**Files:** `inc/vector_math/matrix4f.hpp`

---

## P1 — Significant New SIMD Types

### 4. `Vector3f` — New 16-byte Aligned SIMD Type
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/vector3f.hpp` as a padded 16-byte type with SSE/NEON/scalar paths.  
**Why:** `Vector3` is the workhorse of 3D graphics. 3 floats fit in a 128-bit register with 1 unused lane (set to 0).

**New file:** `inc/vector_math/vector3f.hpp`

| Method | x86/x64 (SSE) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator+` | `_mm_add_ps`, mask/store 3 lanes | `vaddq_f32`, store 3 lanes |
| `operator-` | `_mm_sub_ps`, mask/store 3 lanes | `vsubq_f32`, store 3 lanes |
| `operator-()` | `_mm_xor_ps` | `vnegq_f32` |
| `operator* scalar` | `_mm_mul_ps` + broadcast | `vmulq_n_f32` |
| `operator/ scalar` | `_mm_div_ps` or `*rcp` | `vmulq_n_f32` (precompute `1/scalar`) |
| `dot` | `_mm_mul_ps` → horizontal sum of 3 lanes | `vmulq_f32` → `vaddvq_f32` (ignore 4th) |
| `cross` | `_mm_sub_ps` + `_mm_mul_ps` + shuffles | `vextq_f32` + `vmulq_f32` + `vsubq_f32` |
| `lengthSquared` | reuse `dot` | reuse `dot` |
| `normalize` / `normalized` | SIMD `lengthSquared` + scalar `sqrt` + SIMD `* invLen` | same pattern |
| `reflect` / `reflected` | SIMD `dot` + broadcast + fused sub-mul | same pattern |

**Note:** Must be careful with the 4th lane. Load 3 + zero the 4th, or use unaligned loads/stores for 3 elements. NEON `vld1q_f32` from 3 floats is safe if the 4th is don't-care and masked on store.

**Reference patterns:** `vector4f.hpp`, `vector4d.hpp`

---

### 5. `Vector3d` — New 32-byte Aligned SIMD Type
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/vector3d.hpp` as a padded 32-byte type with AVX/NEON/scalar paths; AVX2-only cross-product shuffle path is guarded.  
**Why:** Same rationale as `Vector3f`. 3 doubles fit in 256-bit AVX with 1 unused lane, or 2× NEON `float64x2_t` (ignoring the 4th element of the 2nd register).

**New file:** `inc/vector_math/vector3d.hpp`

| Method | x86/x64 (AVX) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator+` | `_mm256_add_pd`, mask/store 3 lanes | 2× `vaddq_f64`, store 3 lanes |
| `operator-` | `_mm256_sub_pd`, mask/store 3 lanes | 2× `vsubq_f64`, store 3 lanes |
| `operator-()` | `_mm256_xor_pd` | 2× `vnegq_f64` |
| `operator* scalar` | `_mm256_mul_pd` + broadcast | 2× `vmulq_n_f64` |
| `operator/ scalar` | `_mm256_div_pd` or `*rcp` (no fast rcp for double) | 2× `vmulq_n_f64` (precompute `1/scalar`) |
| `dot` | `_mm256_mul_pd` → `_mm256_hadd_pd` + extract/add | 2× `vmulq_f64` → `vpaddq_f64` → `vaddvq_f64` |
| `cross` | shuffles + `_mm256_sub_pd` + `_mm256_mul_pd` | 2-register cross with manual element swizzles |
| `lengthSquared` | reuse `dot` | reuse `dot` |
| `normalize` / `normalized` | SIMD `lengthSquared` + scalar `sqrt` + SIMD `* invLen` | same pattern |

**Note:** For AVX, can load 4 doubles (from a 32-byte aligned buffer) and use the lower 3, or use `_mm256_maskstore_pd` if AVX2 is available. Fallback: scalar for the 3rd element.

---

### 6. `Quaternionf` — New 16-byte Aligned SIMD Type
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/quaternionf.hpp` with typed float specialization, SIMD-backed component ops/dot/conjugation, and `Vector3f` rotation overloads.  
**Why:** Quaternions are 4 floats = perfect SSE/NEON fit. The Hamilton product (`operator*`) is 16 multiplies + 12 adds — SIMD helps significantly.

**New file:** `inc/vector_math/quaternionf.hpp`

| Method | x86/x64 (SSE) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator+` | `_mm_add_ps` | `vaddq_f32` |
| `operator-` | `_mm_sub_ps` | `vsubq_f32` |
| `operator-()` | `_mm_xor_ps` | `vnegq_f32` |
| `operator* scalar` | `_mm_mul_ps` + broadcast | `vmulq_n_f32` |
| `operator* (Hamilton)` | shuffles + `_mm_add/sub/mul_ps` | `vextq_f32` / `vrev64q_f32` + `vmulq_f32` + `vaddq_f32` / `vsubq_f32` |
| `conjugated` | `_mm_xor_ps` (flip xyz signs) | `vnegq_f32` on lower 3 lanes, or mask |
| `dot` | `_mm_mul_ps` → horizontal sum | `vmulq_f32` → `vaddvq_f32` |
| `lengthSquared` | reuse `dot` | reuse `dot` |
| `inverse` / `inverted` | SIMD `conjugate` + SIMD `* invLenSqr` | same pattern |
| `rotated(Vector3f)` | Can use SIMD for the `qv × v` and `qv × t` cross products if Vector3f SIMD exists | same |

**Note:** The Hamilton product is the big win here. Standard formula:
```
t1 = w1*w2 - x1*x2 - y1*y2 - z1*z2  (w)
t2 = w1*x2 + x1*w2 + y1*z2 - z1*y2  (x)
t3 = w1*y2 - x1*z2 + y1*w2 + z1*x2  (y)
t4 = w1*z2 + x1*y2 - y1*x2 + z1*w2  (z)
```
This decomposes into 4 SIMD dot-products with sign-flipped shuffles.

---

### 7. `Quaterniond` — New 32-byte Aligned SIMD Type
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/quaterniond.hpp` with typed double specialization, SIMD-backed component ops/dot/conjugation where available, and `Vector3d` rotation overloads.  
**Why:** Same as `Quaternionf` but for double precision. Inherits `Vector4d` component-wise ops, but Hamilton product and `rotated` are custom scalar.

**New file:** `inc/vector_math/quaterniond.hpp`

| Method | x86/x64 (AVX) | AArch64 (NEON) |
|--------|---------------|----------------|
| `operator* (Hamilton)` | AVX shuffles + `_mm256_add/sub/mul_pd` | 2× `float64x2_t` + manual swizzles + `vaddq_f64` / `vsubq_f64` |
| `conjugated` | `_mm256_xor_pd` (flip xyz signs, keep w) | 2× `vnegq_f64` with lane mask |
| `rotated(Vector3d)` | AVX cross products if Vector3d SIMD exists | same |
| `inverse` / `inverted` | reuse `Vector4d` dot + SIMD conjugate + scale | same |

---

## P2 — Extensions & Nice-to-Haves

### 8. `Vector4f` / `Vector4d` — `lengthSquared`, `normalize`, `distanceToSquared`
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/vector4f.hpp` and `inc/vector_math/vector4d.hpp` using the SIMD `dot` path plus scalar `sqrt` for normalization.  
**Why:** Hot paths in graphics. Can reuse the SIMD `dot` implementation already planned above.

| Method | SIMD Approach |
|--------|---------------|
| `lengthSquared` | SIMD `dot(self, self)` |
| `distanceToSquared` | SIMD `dot(a-b, a-b)` |
| `normalize` / `normalized` | SIMD `lengthSquared` → scalar `sqrt` → SIMD `* invLength` |

**Files:** `inc/vector_math/vector4f.hpp`, `inc/vector_math/vector4d.hpp`

---

### 9. `Matrix4f::compose` / `Matrix4d::compose` — SIMD Column Scaling
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/matrix4f.hpp` and `inc/vector_math/matrix4d.hpp` with SIMD scaling of the first three rows of the rotation basis before writing translation.  
**Why:** 3 columns × 4 components = 12 multiplies. Can be done with 3 SSE/AVX rows or 3 NEON `float32x4_t` / 6 `float64x2_t` multiplies.

**Approach:** Override `compose` in `Matrix4f` and `Matrix4d` to scale columns 0-2 with SIMD broadcast-multiplies instead of scalar element-wise.

**Files:** `inc/vector_math/matrix4f.hpp`, `inc/vector_math/matrix4d.hpp`

---

### 10. `Vec` — SIMD-Aware Generic Helpers (Conditional)
**Status:** Done via concrete SIMD types  
**Current state:** SIMD helper overrides were implemented in the concrete 4-wide SIMD types (`Vector4f`, `Vector4d`) for `lerp`, `min`, `max`, `clamp`, and `absolute`; the generic `Vec<>` template remains scalar by design.  
**Why:** For `SIZE == 4` and `DATA_TYPE == float/double`, these could dispatch to SIMD internally without creating new types.

**Candidates:**
- `lerp` — 1 SIMD sub + 1 SIMD mul + 1 SIMD add for 4-wide
- `min` / `max` — `_mm_min/max_ps` / `vminmaxq_f32`
- `clamp` — 2× `min`/`max`
- `absolute` — `_mm_and_ps` (clear sign bit) / `vabsq_f32`

**Risk:** Adds complexity to generic templates. May be cleaner to only implement in the concrete SIMD types (`Vector4f`, `Vector4d`, etc.) rather than the generic `Vec<>` base.

**Outcome:** Implemented in concrete SIMD types only, matching the recommendation.

---

### 11. `Vector2f` / `Vector2d` — New Aligned SIMD Types
**Status:** Done  
**Current state:** Implemented in `inc/vector_math/vector2f.hpp` and `inc/vector_math/vector2d.hpp` as aligned 2D SIMD-friendly types with arithmetic, dot, cross, normalize, and reflect helpers.  
**Why:** 2 floats/doubles = half a 128-bit register. SIMD benefit is marginal for such small types unless doing batch operations. Horizontal operations (`dot`, `cross`/`perp`) are trivial scalar.

**Verdict:** Low priority, but now implemented for API completeness and benchmark coverage.

---

## Not Recommended for SIMD

| Method / Type | Reason |
|---------------|--------|
| `Matrix4::determinant` | Complex scalar data flow with many cross-products; SIMD shuffles would be cumbersome and likely slower due to data dependencies. |
| `Matrix4::invert` | Same as determinant — scalar algorithm with heavy data dependencies. |
| `angleTo` / `angleToSigned` | Dominated by scalar `acos` and `sqrt`; SIMD for the vector ops gives minimal overall gain. |
| `Matrix2` / `Matrix3` | Too small; scalar is fine. |
| `slerp` | Dominated by scalar `acos`, `sin`, `sqrt`. |
| `axisAngle` | Dominated by scalar `sin` / `cos`. |
| `perspective`, `ortho`, `rotateX/Y/Z` | Construct only a few elements; scalar is optimal. |

---

## Testing & Benchmarking Checklist

For every SIMD implementation added:

- [x] **Numerical correctness:** GoogleTest coverage added for the implemented SIMD types and overrides, comparing SIMD result against generic scalar result with `FLOAT_EPS` / `DOUBLE_EPS`.
- [x] **Alignment safety:** `alignas(16)` / `alignas(32)` enforced for the implemented float and double SIMD types, with constructor and size/alignment tests added.
- [x] **NEON 32-bit fallback:** Double-precision SIMD types fall back to scalar on non-AArch64 ARM.
- [x] **Benchmark:** Added `google/benchmark` cases comparing SIMD type vs generic type vs GLM for representative vector and matrix paths.
- [x] **Include headers:** `vector_math.hpp` updated to expose the new SIMD vector and quaternion types.

---

## Summary Table

| # | Task | Type | x86/64 | NEON | Effort | Impact |
|---|------|------|--------|------|--------|--------|
| 1 | `Vector4f` SIMD ops | Done | SSE | ✅ | Low | **High** |
| 2 | `Matrix4f` component-wise ops | Done | SSE | ✅ | Low | **High** |
| 3 | `Matrix4f::lookAt` scalar fast | Done | Scalar | Scalar | Low | Medium |
| 4 | `Vector3f` new SIMD type | Done | SSE | ✅ | Medium | **High** |
| 5 | `Vector3d` new SIMD type | Done | AVX | ✅ | Medium | Medium |
| 6 | `Quaternionf` new SIMD type | Done | SSE | ✅ | Medium | **High** |
| 7 | `Quaterniond` new SIMD type | Done | AVX | ✅ | Medium | Medium |
| 8 | `Vector4f/d` length/normalize | Done | SSE/AVX | ✅ | Low | Medium |
| 9 | `Matrix4f/d::compose` SIMD | Done | SSE/AVX | ✅ | Low | Low |
| 10 | `Vec` generic SIMD helpers | Done via concrete types | SSE/AVX | ✅ | Medium | Low |
| 11 | `Vector2f/d` SIMD | Done | SSE2 / scalar | ✅ | Low | Low |
