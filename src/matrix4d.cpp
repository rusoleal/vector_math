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
        Matrix4d toReturn;
        __m256d row1 = _mm256_load_pd(&rhs.data[0]);
        __m256d row2 = _mm256_load_pd(&rhs.data[4]);
        __m256d row3 = _mm256_load_pd(&rhs.data[8]);
        __m256d row4 = _mm256_load_pd(&rhs.data[12]);
        for(int i=0; i<4; i++) {
            __m256d brod1 = _mm256_set1_pd(this->data[4*i + 0]);
            __m256d brod2 = _mm256_set1_pd(this->data[4*i + 1]);
            __m256d brod3 = _mm256_set1_pd(this->data[4*i + 2]);
            __m256d brod4 = _mm256_set1_pd(this->data[4*i + 3]);
            __m256d row = _mm256_add_pd(
                        _mm256_add_pd(
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
        __m256d row1 = _mm256_load_pd(&this->data[0]);
        __m256d row2 = _mm256_load_pd(&this->data[4]);
        __m256d row3 = _mm256_load_pd(&this->data[8]);
        __m256d row4 = _mm256_load_pd(&this->data[12]);
        __m256d vector = _mm256_load_pd(rhs.data);
        __m256d r1 = _mm256_mul_pd(row1, vector);
        __m256d r2 = _mm256_mul_pd(row2, vector);
        __m256d r3 = _mm256_mul_pd(row3, vector);
        __m256d r4 = _mm256_mul_pd(row4, vector);
        // _mm256_hadd_pd operates per 128-bit lane, so we need to combine lanes manually
        __m256d a12 = _mm256_hadd_pd(r1, r2); // [r1[0]+r1[1], r2[0]+r2[1], r1[2]+r1[3], r2[2]+r2[3]]
        __m256d a34 = _mm256_hadd_pd(r3, r4); // [r3[0]+r3[1], r4[0]+r4[1], r3[2]+r3[3], r4[2]+r4[3]]
        __m128d sum12 = _mm_add_pd(_mm256_castpd256_pd128(a12), _mm256_extractf128_pd(a12, 1)); // [r1sum, r2sum]
        __m128d sum34 = _mm_add_pd(_mm256_castpd256_pd128(a34), _mm256_extractf128_pd(a34, 1)); // [r3sum, r4sum]
        __m256d result = _mm256_set_m128d(sum34, sum12);
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

