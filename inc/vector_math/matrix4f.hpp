#pragma once

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4f.hpp>

namespace systems::leal::vector_math
{
    class alignas(16) Matrix4f : public Matrix4<float> {
    public:
        //////////////////
        // constructors //
        //////////////////
        Matrix4f();
        Matrix4f(float value);
        Matrix4f(float buffer[16]);

        ///////////////
        // operators //
        ///////////////
        Matrix4f operator*(const Matrix4f &rhs) const;
        Vector4f operator*(const Vector4f &rhs) const;

        /////////////
        // statics //
        /////////////
        static Matrix4f identity();

    };
}