#pragma once

#include <vector_math/vector2.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 16-byte aligned double 2D vector.
    class alignas(16) Vector2d {
    public:
        double data[2];

        Vector2d() : data{0.0, 0.0} {}
        explicit Vector2d(double value) : data{value, value} {}
        Vector2d(double x, double y) : data{x, y} {}
        Vector2d(const Vector2<double>& other) : data{other.data[0], other.data[1]} {}

        operator Vector2<double>() const { return Vector2<double>(data[0], data[1]); }

        double& x() { return data[0]; }
        const double& x() const { return data[0]; }
        double& y() { return data[1]; }
        const double& y() const { return data[1]; }

        bool operator==(const Vector2d& rhs) const { return data[0] == rhs.data[0] && data[1] == rhs.data[1]; }
        bool operator!=(const Vector2d& rhs) const { return !(*this == rhs); }

        inline Vector2d operator+(const Vector2d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2d result;
                _mm_store_pd(result.data, _mm_add_pd(_mm_load_pd(data), _mm_load_pd(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector2d result;
                vst1q_f64(result.data, vaddq_f64(vld1q_f64(data), vld1q_f64(rhs.data)));
                return result;
            #else
                return Vector2d(data[0] + rhs.data[0], data[1] + rhs.data[1]);
            #endif
        }

        inline Vector2d operator-(const Vector2d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2d result;
                _mm_store_pd(result.data, _mm_sub_pd(_mm_load_pd(data), _mm_load_pd(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector2d result;
                vst1q_f64(result.data, vsubq_f64(vld1q_f64(data), vld1q_f64(rhs.data)));
                return result;
            #else
                return Vector2d(data[0] - rhs.data[0], data[1] - rhs.data[1]);
            #endif
        }

        inline Vector2d operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2d result;
                _mm_store_pd(result.data, _mm_xor_pd(_mm_load_pd(data), _mm_set1_pd(-0.0)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector2d result;
                vst1q_f64(result.data, vnegq_f64(vld1q_f64(data)));
                return result;
            #else
                return Vector2d(-data[0], -data[1]);
            #endif
        }

        inline Vector2d operator*(double scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2d result;
                _mm_store_pd(result.data, _mm_mul_pd(_mm_load_pd(data), _mm_set1_pd(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector2d result;
                vst1q_f64(result.data, vmulq_n_f64(vld1q_f64(data), scalar));
                return result;
            #else
                return Vector2d(data[0] * scalar, data[1] * scalar);
            #endif
        }

        friend inline Vector2d operator*(double scalar, const Vector2d& v) { return v * scalar; }

        inline Vector2d operator/(double scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2d result;
                _mm_store_pd(result.data, _mm_div_pd(_mm_load_pd(data), _mm_set1_pd(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector2d result;
                double inv = 1.0 / scalar;
                vst1q_f64(result.data, vmulq_n_f64(vld1q_f64(data), inv));
                return result;
            #else
                return Vector2d(data[0] / scalar, data[1] / scalar);
            #endif
        }

        inline double dot(const Vector2d& rhs) const { return data[0] * rhs.data[0] + data[1] * rhs.data[1]; }
        static inline double dot(const Vector2d& lhs, const Vector2d& rhs) { return lhs.dot(rhs); }
        inline double cross(const Vector2d& rhs) const { return data[0] * rhs.data[1] - data[1] * rhs.data[0]; }
        inline double lengthSquared() const { return dot(*this); }
        inline double length() const { return std::sqrt(lengthSquared()); }

        inline void normalize() {
            double lenSqr = lengthSquared();
            if (isZero<double>(lenSqr)) return;
            *this = *this * (1.0 / std::sqrt(lenSqr));
        }

        inline Vector2d normalized() const {
            Vector2d result = *this;
            result.normalize();
            return result;
        }

        inline void reflect(const Vector2d& normal) {
            *this = *this - normal * (2.0 * dot(normal));
        }

        inline Vector2d reflected(const Vector2d& normal) const {
            Vector2d result = *this;
            result.reflect(normal);
            return result;
        }
    };

}
