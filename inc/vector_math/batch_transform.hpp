#pragma once

#include <cstddef>
#include <vector_math/matrix4f.hpp>
#include <vector_math/matrix4d.hpp>

namespace systems::leal::vector_math {

    /// Batch matrix-vector transform with runtime SIMD dispatch.
    ///
    /// Computes `out[i] = m * in[i]` for i in [0, count). The implementation
    /// is selected at runtime: on x86/x64, an AVX2+FMA3 path is used when the
    /// CPU supports it; otherwise a scalar fallback runs. On other platforms
    /// the scalar path is used.
    ///
    /// @param m      4×4 transform matrix.
    /// @param in     Input array of `count` vectors (may alias `out`).
    /// @param out    Output array of `count` vectors.
    /// @param count  Number of vectors to transform.
    void transformVectors(const Matrix4f& m, const Vector4f* in, Vector4f* out, std::size_t count);

    /// Batch matrix-vector transform with runtime SIMD dispatch (double precision).
    /// @see transformVectors(const Matrix4f&, const Vector4f*, Vector4f*, std::size_t)
    void transformVectors(const Matrix4d& m, const Vector4d* in, Vector4d* out, std::size_t count);

} // namespace systems::leal::vector_math
