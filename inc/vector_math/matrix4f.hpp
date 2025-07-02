#pragma once

#include <intrin.h>
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
        //auto toReturn = ((Mat<float,4,4> *)this)->operator*(rhs);

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

        //__m128 y = _mm_load_ps(rhs.data);
        //__m128 z =_mm_mul_ps(x,y);
        //__m256 z = _mm256_add_ps(x,y);

        //alignas(float) float result[8];
        //_mm256_store_ps(result, z);


        return toReturn;
    }

    Vector4f Matrix4f::operator*(const Vector4f &rhs) const {
        //auto toReturn = ((Mat<float,4,4> *)this)->operator*(rhs);

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

        //__m128 y = _mm_load_ps(rhs.data);
        //__m128 z =_mm_mul_ps(x,y);
        //__m256 z = _mm256_add_ps(x,y);

        //alignas(float) float result[8];
        //_mm256_store_ps(result, z);


        return toReturn;
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