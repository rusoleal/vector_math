#pragma once

#include <vector_math/vector3.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 32-byte aligned double 3D vector with an internal padded 4th lane for SIMD.
    ///
    /// Public semantics match Vector3<double>; the extra lane is always kept at zero.
    class alignas(32) Vector3d {
    public:
        double data[4];

        /// Default-constructs all components to zero.
        Vector3d() : data{0.0, 0.0, 0.0, 0.0} {}

        /// Broadcast constructor.
        explicit Vector3d(double value) : data{value, value, value, 0.0} {}

        /// Constructs from explicit x, y, z values.
        Vector3d(double x, double y, double z) : data{x, y, z, 0.0} {}

        /// Constructs from the generic Vector3<double>.
        Vector3d(const Vector3<double>& other) : data{other.data[0], other.data[1], other.data[2], 0.0} {}

        /// Converts to the generic Vector3<double>.
        operator Vector3<double>() const { return Vector3<double>(data[0], data[1], data[2]); }

        double& x() { return data[0]; }
        const double& x() const { return data[0]; }
        double& y() { return data[1]; }
        const double& y() const { return data[1]; }
        double& z() { return data[2]; }
        const double& z() const { return data[2]; }

        bool operator==(const Vector3d& rhs) const {
            return data[0] == rhs.data[0] && data[1] == rhs.data[1] && data[2] == rhs.data[2];
        }

        bool operator!=(const Vector3d& rhs) const { return !(*this == rhs); }

        inline Vector3d operator+(const Vector3d& rhs) const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_add_pd(_mm256_load_pd(data), _mm256_load_pd(rhs.data)));
                result.data[3] = 0.0;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector3d result;
                vst1q_f64(&result.data[0], vaddq_f64(vld1q_f64(&data[0]), vld1q_f64(&rhs.data[0])));
                vst1q_f64(&result.data[2], vaddq_f64(vld1q_f64(&data[2]), vld1q_f64(&rhs.data[2])));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2]);
            #endif
        }

        inline Vector3d operator-(const Vector3d& rhs) const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_sub_pd(_mm256_load_pd(data), _mm256_load_pd(rhs.data)));
                result.data[3] = 0.0;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector3d result;
                vst1q_f64(&result.data[0], vsubq_f64(vld1q_f64(&data[0]), vld1q_f64(&rhs.data[0])));
                vst1q_f64(&result.data[2], vsubq_f64(vld1q_f64(&data[2]), vld1q_f64(&rhs.data[2])));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2]);
            #endif
        }

        inline Vector3d operator-() const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_xor_pd(_mm256_load_pd(data), _mm256_set1_pd(-0.0)));
                result.data[3] = 0.0;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector3d result;
                vst1q_f64(&result.data[0], vnegq_f64(vld1q_f64(&data[0])));
                vst1q_f64(&result.data[2], vnegq_f64(vld1q_f64(&data[2])));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(-data[0], -data[1], -data[2]);
            #endif
        }

        inline Vector3d operator*(double scalar) const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_mul_pd(_mm256_load_pd(data), _mm256_set1_pd(scalar)));
                result.data[3] = 0.0;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector3d result;
                vst1q_f64(&result.data[0], vmulq_n_f64(vld1q_f64(&data[0]), scalar));
                vst1q_f64(&result.data[2], vmulq_n_f64(vld1q_f64(&data[2]), scalar));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(data[0] * scalar, data[1] * scalar, data[2] * scalar);
            #endif
        }

        friend inline Vector3d operator*(double scalar, const Vector3d& v) { return v * scalar; }

        inline Vector3d operator/(double scalar) const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_div_pd(_mm256_load_pd(data), _mm256_set1_pd(scalar)));
                result.data[3] = 0.0;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector3d result;
                double inv = 1.0 / scalar;
                vst1q_f64(&result.data[0], vmulq_n_f64(vld1q_f64(&data[0]), inv));
                vst1q_f64(&result.data[2], vmulq_n_f64(vld1q_f64(&data[2]), inv));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(data[0] / scalar, data[1] / scalar, data[2] / scalar);
            #endif
        }

        inline double dot(const Vector3d& rhs) const {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                __m256d mul = _mm256_mul_pd(_mm256_load_pd(data), _mm256_load_pd(rhs.data));
                __m256d hadd = _mm256_hadd_pd(mul, mul);
                __m128d lo = _mm256_castpd256_pd128(hadd);
                __m128d hi = _mm256_extractf128_pd(hadd, 1);
                return _mm_cvtsd_f64(_mm_add_pd(lo, hi));
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                float64x2_t prod_lo = vmulq_f64(vld1q_f64(&data[0]), vld1q_f64(&rhs.data[0]));
                float64x2_t prod_hi = vmulq_f64(vld1q_f64(&data[2]), vld1q_f64(&rhs.data[2]));
                return vaddvq_f64(vpaddq_f64(prod_lo, prod_hi));
            #else
                return data[0]*rhs.data[0] + data[1]*rhs.data[1] + data[2]*rhs.data[2];
            #endif
        }

        static inline double dot(const Vector3d& lhs, const Vector3d& rhs) { return lhs.dot(rhs); }

        inline double lengthSquared() const { return dot(*this); }

        inline double length() const { return std::sqrt(lengthSquared()); }

        inline void normalize() {
            double lenSqr = lengthSquared();
            if (isZero<double>(lenSqr)) return;
            *this = *this * (1.0 / std::sqrt(lenSqr));
        }

        inline Vector3d normalized() const {
            Vector3d result = *this;
            result.normalize();
            return result;
        }

        static inline Vector3d cross(const Vector3d& lhs, const Vector3d& rhs) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX2__)
                __m256d a = _mm256_load_pd(lhs.data);
                __m256d b = _mm256_load_pd(rhs.data);
                __m256d a_yzx = _mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 0, 2, 1));
                __m256d b_zxy = _mm256_permute4x64_pd(b, _MM_SHUFFLE(3, 1, 0, 2));
                __m256d a_zxy = _mm256_permute4x64_pd(a, _MM_SHUFFLE(3, 1, 0, 2));
                __m256d b_yzx = _mm256_permute4x64_pd(b, _MM_SHUFFLE(3, 0, 2, 1));
                Vector3d result;
                _mm256_store_pd(result.data, _mm256_sub_pd(_mm256_mul_pd(a_yzx, b_zxy), _mm256_mul_pd(a_zxy, b_yzx)));
                result.data[3] = 0.0;
                return result;
            #else
                return Vector3d(
                    (lhs.data[1] * rhs.data[2]) - (lhs.data[2] * rhs.data[1]),
                    (lhs.data[2] * rhs.data[0]) - (lhs.data[0] * rhs.data[2]),
                    (lhs.data[0] * rhs.data[1]) - (lhs.data[1] * rhs.data[0])
                );
            #endif
        }

        inline void reflect(const Vector3d& normal) {
            *this = *this - normal * (2.0 * dot(normal));
        }

        inline Vector3d reflected(const Vector3d& normal) const {
            Vector3d result = *this;
            result.reflect(normal);
            return result;
        }

        inline void normalizeInto(Vector3d& out) const {
            out = normalized();
        }
    };

}
