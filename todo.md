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
**Current state:** Only `alignas(16)`; falls back to scalar `Vector4<float>` for all ops.  
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
**Current state:** Only `operator*(Matrix4f)` and `operator*(Vector4f)` are SIMD-accelerated. `+`, `-`, `-()`, `* scalar` fall back to generic scalar loops.  
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
**Current state:** Uses generic `Matrix4<float>::lookAt` which builds intermediate `Vector3`s and calls `identity()`.  
**Why:** `Matrix4d` has an optimized inline scalar override that avoids all generic `Vec<>` loop overhead. Same optimization applies here.

**Approach:** Port the `Matrix4d::lookAt` scalar-fast implementation to `Matrix4f`.

**Files:** `inc/vector_math/matrix4f.hpp`

---

## P1 — Significant New SIMD Types

### 4. `Vector3f` — New 16-byte Aligned SIMD Type
**Current state:** No SIMD specialization for 3D float vectors.  
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
**Current state:** No SIMD specialization for 3D double vectors.  
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
**Current state:** `Quaternion<float>` inherits `Vector4<float>` — all scalar.  
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
**Current state:** `Quaternion<double>` inherits `Vector4<double>` — all scalar.  
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
**Current state:** Inherited from `Vec<>` — scalar loops.  
**Why:** Hot paths in graphics. Can reuse the SIMD `dot` implementation already planned above.

| Method | SIMD Approach |
|--------|---------------|
| `lengthSquared` | SIMD `dot(self, self)` |
| `distanceToSquared` | SIMD `dot(a-b, a-b)` |
| `normalize` / `normalized` | SIMD `lengthSquared` → scalar `sqrt` → SIMD `* invLength` |

**Files:** `inc/vector_math/vector4f.hpp`, `inc/vector_math/vector4d.hpp`

---

### 9. `Matrix4f::compose` / `Matrix4d::compose` — SIMD Column Scaling
**Current state:** Scalar loop scaling the first 3 columns by scale vector components.  
**Why:** 3 columns × 4 components = 12 multiplies. Can be done with 3 SSE/AVX rows or 3 NEON `float32x4_t` / 6 `float64x2_t` multiplies.

**Approach:** Override `compose` in `Matrix4f` and `Matrix4d` to scale columns 0-2 with SIMD broadcast-multiplies instead of scalar element-wise.

**Files:** `inc/vector_math/matrix4f.hpp`, `inc/vector_math/matrix4d.hpp`

---

### 10. `Vec` — SIMD-Aware Generic Helpers (Conditional)
**Current state:** All `Vec<DATA_TYPE, SIZE>` helpers are scalar loops.  
**Why:** For `SIZE == 4` and `DATA_TYPE == float/double`, these could dispatch to SIMD internally without creating new types.

**Candidates:**
- `lerp` — 1 SIMD sub + 1 SIMD mul + 1 SIMD add for 4-wide
- `min` / `max` — `_mm_min/max_ps` / `vminmaxq_f32`
- `clamp` — 2× `min`/`max`
- `absolute` — `_mm_and_ps` (clear sign bit) / `vabsq_f32`

**Risk:** Adds complexity to generic templates. May be cleaner to only implement in the concrete SIMD types (`Vector4f`, `Vector4d`, etc.) rather than the generic `Vec<>` base.

**Recommendation:** Implement in concrete SIMD types only. Mark as **P2**.

---

### 11. `Vector2f` / `Vector2d` — New Aligned SIMD Types
**Current state:** No SIMD specialization.  
**Why:** 2 floats/doubles = half a 128-bit register. SIMD benefit is marginal for such small types unless doing batch operations. Horizontal operations (`dot`, `cross`/`perp`) are trivial scalar.

**Verdict:** Low priority. A compiler with auto-vectorization usually handles 2-wide scalar code well. **Defer.**

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

- [ ] **Numerical correctness:** Add GoogleTest cases comparing SIMD result against generic scalar result with appropriate epsilon (`FLOAT_EPS`, `DOUBLE_EPS`).
- [ ] **Alignment safety:** Ensure `alignas(16)` for float types and `alignas(32)` for double types are respected in constructors, stack allocation, and member storage.
- [ ] **NEON 32-bit fallback:** ARMv7 lacks double-precision NEON. Ensure `Vector3d`, `Vector4d`, `Matrix4d`, `Quaterniond` fall back to scalar on `__arm__` (non-`__aarch64__`).
- [ ] **Benchmark:** Add `google/benchmark` cases comparing SIMD type vs generic type vs GLM (where applicable).
- [ ] **Include headers:** Update `vector_math.hpp` (or main include) to expose new types.

---

## Summary Table

| # | Task | Type | x86/64 | NEON | Effort | Impact |
|---|------|------|--------|------|--------|--------|
| 1 | `Vector4f` SIMD ops | Missing | SSE | ✅ | Low | **High** |
| 2 | `Matrix4f` component-wise ops | Missing | SSE | ✅ | Low | **High** |
| 3 | `Matrix4f::lookAt` scalar fast | Missing | Scalar | Scalar | Low | Medium |
| 4 | `Vector3f` new SIMD type | New | SSE | ✅ | Medium | **High** |
| 5 | `Vector3d` new SIMD type | New | AVX | ✅ | Medium | Medium |
| 6 | `Quaternionf` new SIMD type | New | SSE | ✅ | Medium | **High** |
| 7 | `Quaterniond` new SIMD type | New | AVX | ✅ | Medium | Medium |
| 8 | `Vector4f/d` length/normalize | Extension | SSE/AVX | ✅ | Low | Medium |
| 9 | `Matrix4f/d::compose` SIMD | Extension | SSE/AVX | ✅ | Low | Low |
| 10 | `Vec` generic SIMD helpers | Extension | SSE/AVX | ✅ | Medium | Low |
| 11 | `Vector2f/d` SIMD | Defer | — | — | — | Low |
