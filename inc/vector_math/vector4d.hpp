#pragma once

#include <vector_math/vector4.hpp>

namespace systems::leal::vector_math {

    /// 32-byte aligned double 4D vector.
    ///
    /// Identical to Vector4<double> but with `alignas(32)` to satisfy AVX load
    /// requirements. Use this type when passing vectors to Matrix4d operations.
    class alignas(32) Vector4d:public Vector4<double> {
    public:
        /// Default-constructs all elements to zero.
        Vector4d() = default;

        /// Constructs from explicit x, y, z, w values.
        Vector4d(double x, double y, double z, double w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

    };

}
