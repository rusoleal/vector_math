#pragma once

#include <vector_math/vector2.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 16-byte aligned float 2D vector with padded SIMD lanes.
    class alignas(16) Vector2f {
    public:
        float data[4];

        Vector2f() : data{0.0f, 0.0f, 0.0f, 0.0f} {}
        explicit Vector2f(float value) : data{value, value, 0.0f, 0.0f} {}
        Vector2f(float x, float y) : data{x, y, 0.0f, 0.0f} {}
        Vector2f(const Vector2<float>& other) : data{other.data[0], other.data[1], 0.0f, 0.0f} {}

        operator Vector2<float>() const { return Vector2<float>(data[0], data[1]); }

        float& x() { return data[0]; }
        const float& x() const { return data[0]; }
        float& y() { return data[1]; }
        const float& y() const { return data[1]; }

        bool operator==(const Vector2f& rhs) const { return data[0] == rhs.data[0] && data[1] == rhs.data[1]; }
        bool operator!=(const Vector2f& rhs) const { return !(*this == rhs); }

        inline Vector2f operator+(const Vector2f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2f result;
                _mm_store_ps(result.data, _mm_add_ps(_mm_load_ps(data), _mm_load_ps(rhs.data)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector2f result;
                vst1q_f32(result.data, vaddq_f32(vld1q_f32(data), vld1q_f32(rhs.data)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #else
                return Vector2f(data[0] + rhs.data[0], data[1] + rhs.data[1]);
            #endif
        }

        inline Vector2f operator-(const Vector2f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2f result;
                _mm_store_ps(result.data, _mm_sub_ps(_mm_load_ps(data), _mm_load_ps(rhs.data)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector2f result;
                vst1q_f32(result.data, vsubq_f32(vld1q_f32(data), vld1q_f32(rhs.data)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #else
                return Vector2f(data[0] - rhs.data[0], data[1] - rhs.data[1]);
            #endif
        }

        inline Vector2f operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2f result;
                _mm_store_ps(result.data, _mm_xor_ps(_mm_load_ps(data), _mm_set1_ps(-0.0f)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector2f result;
                vst1q_f32(result.data, vnegq_f32(vld1q_f32(data)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #else
                return Vector2f(-data[0], -data[1]);
            #endif
        }

        inline Vector2f operator*(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2f result;
                _mm_store_ps(result.data, _mm_mul_ps(_mm_load_ps(data), _mm_set1_ps(scalar)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector2f result;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(data), scalar));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #else
                return Vector2f(data[0] * scalar, data[1] * scalar);
            #endif
        }

        friend inline Vector2f operator*(float scalar, const Vector2f& v) { return v * scalar; }

        inline Vector2f operator/(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector2f result;
                _mm_store_ps(result.data, _mm_div_ps(_mm_load_ps(data), _mm_set1_ps(scalar)));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector2f result;
                float inv = 1.0f / scalar;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(data), inv));
                result.data[2] = result.data[3] = 0.0f;
                return result;
            #else
                return Vector2f(data[0] / scalar, data[1] / scalar);
            #endif
        }

        inline float dot(const Vector2f& rhs) const {
            return data[0] * rhs.data[0] + data[1] * rhs.data[1];
        }

        static inline float dot(const Vector2f& lhs, const Vector2f& rhs) { return lhs.dot(rhs); }

        inline float cross(const Vector2f& rhs) const { return data[0] * rhs.data[1] - data[1] * rhs.data[0]; }

        inline float lengthSquared() const { return dot(*this); }
        inline float length() const { return std::sqrt(lengthSquared()); }

        inline void normalize() {
            float lenSqr = lengthSquared();
            if (isZero<float>(lenSqr)) return;
            *this = *this * (1.0f / std::sqrt(lenSqr));
        }

        inline Vector2f normalized() const {
            Vector2f result = *this;
            result.normalize();
            return result;
        }

        inline void reflect(const Vector2f& normal) {
            *this = *this - normal * (2.0f * dot(normal));
        }

        inline Vector2f reflected(const Vector2f& normal) const {
            Vector2f result = *this;
            result.reflect(normal);
            return result;
        }
    };

}
