#pragma once

#include <vector_math/vector4.hpp>

namespace systems::leal::vector_math {

    class alignas(32) Vector4d:public Vector4<double> {
    public:
        Vector4d() = default;
        Vector4d(double x, double y, double z, double w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

    };

}
