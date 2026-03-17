#pragma once

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4f.hpp>

namespace systems::leal::vector_math
{
    /// 16-byte aligned float 4×4 matrix with SIMD-accelerated operations.
    ///
    /// Inherits the full Matrix4<float> API. The `operator*` overloads below
    /// replace the generic implementations with architecture-specific paths:
    /// - **x86/x86-64**: SSE 128-bit intrinsics (`_mm_load_ps`, `_mm_mul_ps`, etc.)
    /// - **ARM**: NEON intrinsics (`vld1q_f32`, `vfmaq_laneq_f32`, etc.)
    /// - **Other**: falls back to the generic scalar implementation.
    ///
    /// @note Both the matrix and the Vector4f operand must be 16-byte aligned.
    class alignas(16) Matrix4f : public Matrix4<float> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Matrix4f();

        /// Constructs all elements from a single @p value.
        Matrix4f(float value);

        /// Constructs from a 16-element row-major C array.
        Matrix4f(float buffer[16]);

        // ── SIMD-accelerated operators ────────────────────────────────────────

        /// Matrix–matrix multiplication using SIMD intrinsics.
        Matrix4f operator*(const Matrix4f &rhs) const;

        /// Matrix–Vector4f multiplication using SIMD intrinsics.
        Vector4f operator*(const Vector4f &rhs) const;

        // ── Static factories ──────────────────────────────────────────────────

        /// Returns a 4×4 identity matrix.
        static Matrix4f identity();

    };
}
