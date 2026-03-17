#pragma once

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4d.hpp>

namespace systems::leal::vector_math
{
    /// 32-byte aligned double 4×4 matrix with SIMD-accelerated operations.
    ///
    /// Inherits the full Matrix4<double> API. The `operator*` overloads below
    /// replace the generic implementations with architecture-specific paths:
    /// - **x86/x86-64**: AVX 256-bit intrinsics (`_mm256_load_pd`, `_mm256_mul_pd`, etc.)
    /// - **ARM**: falls back to the generic scalar implementation.
    ///
    /// @note Both the matrix and the Vector4d operand must be 32-byte aligned.
    class alignas(32) Matrix4d : public Matrix4<double> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Matrix4d();

        /// Constructs all elements from a single @p value.
        Matrix4d(double value);

        /// Constructs from a 16-element row-major C array.
        Matrix4d(double buffer[16]);

        // ── SIMD-accelerated operators ────────────────────────────────────────

        /// Matrix–matrix multiplication using SIMD intrinsics.
        Matrix4d operator*(const Matrix4d &rhs) const;

        /// Matrix–Vector4d multiplication using SIMD intrinsics.
        Vector4d operator*(const Vector4d &rhs) const;

        // ── Static factories ──────────────────────────────────────────────────

        /// Returns a 4×4 identity matrix.
        static Matrix4d identity();

    };
}
