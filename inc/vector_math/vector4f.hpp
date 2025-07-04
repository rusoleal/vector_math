#pragma once

#include <vector_math/vector4.hpp>

namespace systems::leal::vector_math {

    class alignas(16) Vector4f:public Vector4<float> {
    public:
        Vector4f() = default;
        Vector4f(float x, float y, float z, float w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

    };

}
