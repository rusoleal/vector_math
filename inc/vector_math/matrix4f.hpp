#pragma once

#if defined(__x86__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    #define __VECTOR_MATH_ARCH_X86_X64
    #undef __VECTOR_MATH_ARCH_ARM
#endif

#if defined(__arm__) || defined(__arm64__)
    #undef __VECTOR_MATH_ARCH_X86_X64
    #define __VECTOR_MATH_ARCH_ARM
#endif

#ifdef __VECTOR_MATH_ARCH_X86_X64
    // do x64 stuff   
    #include <intrin.h>
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    // do arm stuff
#endif  

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4f.hpp>

namespace systems::leal::vector_math
{
    class alignas(float) Matrix4f : public Matrix4<float> {
    public:
        //////////////////
        // constructors //
        //////////////////
        Matrix4f():Matrix4<float>() {}
        Matrix4f(float value):Matrix4<float>(value) {}
        Matrix4f(float buffer[16]):Matrix4<float>(buffer) {}

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

    Matrix4f Matrix4f::operator*(const Matrix4f &rhs) const {
        #ifdef __VECTOR_MATH_ARCH_X86_X64
            Matrix4f toReturn;
            auto transposed = rhs.transpose();
            alignas(float) float result[4];

            const float *plhs = this->data;
            for (int c=0; c<4; c++) {
                const float *prhs = transposed.data;
                __m128 x = _mm_load_ps(plhs);
                for (int r=0; r<4; r++) {
                    __m128 y = _mm_load_ps(prhs);
                    __m128 z =_mm_mul_ps(x,y);
                    _mm_store_ps(result, z);

                    float value=0;
                    for (int e=0; e<4; e++) {
                        value += result[e];
                    }
                    toReturn.data[4*r + c] = value;

                    prhs += 4;
                }
                plhs += 4;
            }
            return toReturn;
        #elif defined(__VECTOR_MATH_ARCH_ARM)
            auto toReturn = ((Matrix4<float> *)this)->operator*(rhs);
            return *(Matrix4f *)&toReturn;
        #endif  
    }

    Vector4f Matrix4f::operator*(const Vector4f &rhs) const {
        #ifdef __VECTOR_MATH_ARCH_X86_X64
            Vector4f toReturn;
            alignas(float) float result[4];

            const float *plhs = this->data;
            const float *prhs = rhs.data;
            __m128 y = _mm_load_ps(prhs);
            for (int r=0; r<4; r++) {
                __m128 x = _mm_load_ps(plhs);
                __m128 z =_mm_mul_ps(x,y);
                _mm_store_ps(result, z);

                float value=0;
                for (int e=0; e<4; e++) {
                    value += result[e];
                }
                toReturn.data[r] = value;

                plhs += 4;
            }
            return toReturn;
        #elif defined(__VECTOR_MATH_ARCH_ARM)
            auto toReturn = ((Matrix4<float> *)this)->operator*(rhs);
            return *(Vector4f *)&toReturn;
        #endif
    }

    Matrix4f Matrix4f::identity()
    {
        float data[] = {
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,
        };
        return Matrix4f(data);
    }

}