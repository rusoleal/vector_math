#include <vector_math/matrix4f.hpp>

using namespace systems::leal::vector_math;

Matrix4f::Matrix4f():Matrix4<float>() {
}

Matrix4f::Matrix4f(float value):Matrix4<float>(value) {
}

Matrix4f::Matrix4f(float buffer[16]):Matrix4<float>(buffer) {
}

Matrix4f Matrix4f::operator*(const Matrix4f &rhs) const {
    #ifdef __VECTOR_MATH_ARCH_X86_X64
        //printf("1\n");
        Matrix4f toReturn;
        __m128 row1 = _mm_load_ps(&rhs.data[0]);
        __m128 row2 = _mm_load_ps(&rhs.data[4]);
        __m128 row3 = _mm_load_ps(&rhs.data[8]);
        __m128 row4 = _mm_load_ps(&rhs.data[12]);
        for(int i=0; i<4; i++) {
            __m128 brod1 = _mm_set1_ps(this->data[4*i + 0]);
            __m128 brod2 = _mm_set1_ps(this->data[4*i + 1]);
            __m128 brod3 = _mm_set1_ps(this->data[4*i + 2]);
            __m128 brod4 = _mm_set1_ps(this->data[4*i + 3]);
            __m128 row = _mm_add_ps(
                        _mm_add_ps(
                            _mm_mul_ps(brod1, row1),
                            _mm_mul_ps(brod2, row2)),
                        _mm_add_ps(
                            _mm_mul_ps(brod3, row3),
                            _mm_mul_ps(brod4, row4)));
            _mm_store_ps(&toReturn.data[4*i], row);
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
        __m128 row1 = _mm_load_ps(&this->data[0]);
        __m128 row2 = _mm_load_ps(&this->data[4]);
        __m128 row3 = _mm_load_ps(&this->data[8]);
        __m128 row4 = _mm_load_ps(&this->data[12]);
        __m128 vector = _mm_load_ps(rhs.data);
        __m128 r1 = _mm_mul_ps(row1, vector);
        __m128 r2 = _mm_mul_ps(row2, vector);
        __m128 r3 = _mm_mul_ps(row3, vector);
        __m128 r4 = _mm_mul_ps(row4, vector);
        __m128 result = _mm_hadd_ps(
            _mm_hadd_ps(r1,r2),
            _mm_hadd_ps(r3,r4)
        );
        _mm_store_ps(toReturn.data, result);
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

