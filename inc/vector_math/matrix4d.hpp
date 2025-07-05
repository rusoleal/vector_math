#pragma once

#include <vector_math/common.hpp>
#include <vector_math/matrix4.hpp>
#include <vector_math/vector4d.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
#endif  


namespace systems::leal::vector_math
{
    class alignas(32) Matrix4d : public Matrix4<double> {
    public:
        //////////////////
        // constructors //
        //////////////////
        Matrix4d();
        Matrix4d(double value);
        Matrix4d(double buffer[16]);

        ///////////////
        // operators //
        ///////////////
        Matrix4d operator*(const Matrix4d &rhs) const;
        Vector4d operator*(const Vector4d &rhs) const;

        /////////////
        // statics //
        /////////////
        static Matrix4d identity();

    };
}