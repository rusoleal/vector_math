#pragma once

#include <vector_math/vector4.hpp>

namespace systems::leal::vector_math {

    /// 16-byte aligned float 4D vector.
    ///
    /// Identical to Vector4<float> but with `alignas(16)` to satisfy SSE/NEON load
    /// requirements. Use this type when passing vectors to Matrix4f operations.
    class alignas(16) Vector4f:public Vector4<float> {
    public:
        /// Default-constructs all elements to zero.
        Vector4f() = default;

        /// Constructs from explicit x, y, z, w values.
        Vector4f(float x, float y, float z, float w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

    };

}
