#include <vector_math/common.hpp>
#include <vector_math/matrix4d.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif  

using namespace systems::leal::vector_math;

Matrix4d::Matrix4d():Matrix4<double>() {
}

Matrix4d::Matrix4d(double value):Matrix4<double>(value) {
}

Matrix4d::Matrix4d(double buffer[16]):Matrix4<double>(buffer) {
}

Matrix4d Matrix4d::operator*(const Matrix4d &rhs) const {
    #ifdef __VECTOR_MATH_ARCH_X86_X64
        //printf("1\n");
        Matrix4d toReturn;
        __m256 row1 = _mm256_load_pd(&rhs.data[0]);
        __m256 row2 = _mm256_load_pd(&rhs.data[4]);
        __m256 row3 = _mm256_load_pd(&rhs.data[8]);
        __m256 row4 = _mm256_load_pd(&rhs.data[12]);
        for(int i=0; i<4; i++) {
            __m256 brod1 = _mm256_set1_pd(this->data[4*i + 0]);
            __m256 brod2 = _mm256_set1_pd(this->data[4*i + 1]);
            __m256 brod3 = _mm256_set1_pd(this->data[4*i + 2]);
            __m256 brod4 = _mm256_set1_pd(this->data[4*i + 3]);
            __m256 row = _mm256_add_pd(
                        _mm256_add_ps(
                            _mm256_mul_pd(brod1, row1),
                            _mm256_mul_pd(brod2, row2)),
                        _mm256_add_pd(
                            _mm256_mul_pd(brod3, row3),
                            _mm256_mul_pd(brod4, row4)));
            _mm256_store_pd(&toReturn.data[4*i], row);
        }
        return toReturn;        
    #elif defined(__VECTOR_MATH_ARCH_ARM)
        auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
        return *(Matrix4d *)&toReturn;
    #endif  
}

Vector4d Matrix4d::operator*(const Vector4d &rhs) const {
    #ifdef __VECTOR_MATH_ARCH_X86_X64
        Vector4d toReturn;
        __m256 row1 = _mm256_load_pd(&this->data[0]);
        __m256 row2 = _mm256_load_pd(&this->data[4]);
        __m256 row3 = _mm256_load_pd(&this->data[8]);
        __m256 row4 = _mm256_load_pd(&this->data[12]);
        __m256 vector = _mm256_load_pd(rhs.data);
        __m256 r1 = _mm256_mul_pd(row1, vector);
        __m256 r2 = _mm256_mul_pd(row2, vector);
        __m256 r3 = _mm256_mul_pd(row3, vector);
        __m256 r4 = _mm256_mul_pd(row4, vector);
        __m256 result = _mm256_hadd_pd(
            _mm256_hadd_pd(r1,r2),
            _mm256_hadd_pd(r3,r4)
        );
        _mm256_store_pd(toReturn.data, result);
        return toReturn;        
    #elif defined(__VECTOR_MATH_ARCH_ARM)
        auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
        return *(Vector4d *)&toReturn;
    #endif
}

Matrix4d Matrix4d::identity()
{
    double data[] = {
        1.0,0.0,0.0,0.0,
        0.0,1.0,0.0,0.0,
        0.0,0.0,1.0,0.0,
        0.0,0.0,0.0,1.0,
    };
    return Matrix4d(data);
}

