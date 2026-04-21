#pragma once

#include <vector_math/vector3.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 16-byte aligned float 3D vector with an internal padded 4th lane for SIMD.
    ///
    /// Public semantics match Vector3<float>; the extra lane is always kept at zero.
    class alignas(16) Vector3f {
    public:
        float data[4];

        /// Default-constructs all components to zero.
        Vector3f() : data{0.0f, 0.0f, 0.0f, 0.0f} {}

        /// Broadcast constructor.
        explicit Vector3f(float value) : data{value, value, value, 0.0f} {}

        /// Constructs from explicit x, y, z values.
        Vector3f(float x, float y, float z) : data{x, y, z, 0.0f} {}

        /// Constructs from the generic Vector3<float>.
        Vector3f(const Vector3<float>& other) : data{other.data[0], other.data[1], other.data[2], 0.0f} {}

        /// Converts to the generic Vector3<float>.
        operator Vector3<float>() const { return Vector3<float>(data[0], data[1], data[2]); }

        float& x() { return data[0]; }
        const float& x() const { return data[0]; }
        float& y() { return data[1]; }
        const float& y() const { return data[1]; }
        float& z() { return data[2]; }
        const float& z() const { return data[2]; }

        bool operator==(const Vector3f& rhs) const {
            return data[0] == rhs.data[0] && data[1] == rhs.data[1] && data[2] == rhs.data[2];
        }

        bool operator!=(const Vector3f& rhs) const { return !(*this == rhs); }

        inline Vector3f operator+(const Vector3f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector3f result;
                _mm_store_ps(result.data, _mm_add_ps(_mm_load_ps(data), _mm_load_ps(rhs.data)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector3f result;
                vst1q_f32(result.data, vaddq_f32(vld1q_f32(data), vld1q_f32(rhs.data)));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(data[0] + rhs.data[0], data[1] + rhs.data[1], data[2] + rhs.data[2]);
            #endif
        }

        inline Vector3f operator-(const Vector3f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector3f result;
                _mm_store_ps(result.data, _mm_sub_ps(_mm_load_ps(data), _mm_load_ps(rhs.data)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector3f result;
                vst1q_f32(result.data, vsubq_f32(vld1q_f32(data), vld1q_f32(rhs.data)));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(data[0] - rhs.data[0], data[1] - rhs.data[1], data[2] - rhs.data[2]);
            #endif
        }

        inline Vector3f operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector3f result;
                _mm_store_ps(result.data, _mm_xor_ps(_mm_load_ps(data), _mm_set1_ps(-0.0f)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector3f result;
                vst1q_f32(result.data, vnegq_f32(vld1q_f32(data)));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(-data[0], -data[1], -data[2]);
            #endif
        }

        inline Vector3f operator*(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector3f result;
                _mm_store_ps(result.data, _mm_mul_ps(_mm_load_ps(data), _mm_set1_ps(scalar)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector3f result;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(data), scalar));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(data[0] * scalar, data[1] * scalar, data[2] * scalar);
            #endif
        }

        friend inline Vector3f operator*(float scalar, const Vector3f& v) { return v * scalar; }

        inline Vector3f operator/(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector3f result;
                _mm_store_ps(result.data, _mm_div_ps(_mm_load_ps(data), _mm_set1_ps(scalar)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector3f result;
                float inv = 1.0f / scalar;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(data), inv));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(data[0] / scalar, data[1] / scalar, data[2] / scalar);
            #endif
        }

        inline float dot(const Vector3f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 mul = _mm_mul_ps(_mm_load_ps(data), _mm_load_ps(rhs.data));
                #ifdef __SSE3__
                    __m128 sum = _mm_hadd_ps(mul, mul);
                    sum = _mm_hadd_ps(sum, sum);
                    return _mm_cvtss_f32(sum);
                #else
                    __m128 shuf = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
                    __m128 sums = _mm_add_ps(mul, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    return _mm_cvtss_f32(sums);
                #endif
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                return vaddvq_f32(vmulq_f32(vld1q_f32(data), vld1q_f32(rhs.data)));
            #else
                return data[0]*rhs.data[0] + data[1]*rhs.data[1] + data[2]*rhs.data[2];
            #endif
        }

        static inline float dot(const Vector3f& lhs, const Vector3f& rhs) { return lhs.dot(rhs); }

        inline float lengthSquared() const { return dot(*this); }

        inline float length() const { return std::sqrt(lengthSquared()); }

        inline void normalize() {
            float lenSqr = lengthSquared();
            if (isZero<float>(lenSqr)) return;
            *this = *this * (1.0f / std::sqrt(lenSqr));
        }

        inline Vector3f normalized() const {
            Vector3f result = *this;
            result.normalize();
            return result;
        }

        static inline Vector3f cross(const Vector3f& lhs, const Vector3f& rhs) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 a = _mm_load_ps(lhs.data);
                __m128 b = _mm_load_ps(rhs.data);
                __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
                __m128 b_zxy = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2));
                __m128 a_zxy = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2));
                __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
                Vector3f result;
                _mm_store_ps(result.data, _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx)));
                result.data[3] = 0.0f;
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                float32x4_t a = vld1q_f32(lhs.data);
                float32x4_t b = vld1q_f32(rhs.data);
                float32x4_t a_yzx = vextq_f32(a, a, 1);
                float32x4_t b_yzx = vextq_f32(b, b, 1);
                float32x4_t a_zxy = vextq_f32(a, a, 2);
                float32x4_t b_zxy = vextq_f32(b, b, 2);
                Vector3f result;
                vst1q_f32(result.data, vsubq_f32(vmulq_f32(a_yzx, b_zxy), vmulq_f32(a_zxy, b_yzx)));
                result.data[3] = 0.0f;
                return result;
            #else
                return Vector3f(
                    (lhs.data[1] * rhs.data[2]) - (lhs.data[2] * rhs.data[1]),
                    (lhs.data[2] * rhs.data[0]) - (lhs.data[0] * rhs.data[2]),
                    (lhs.data[0] * rhs.data[1]) - (lhs.data[1] * rhs.data[0])
                );
            #endif
        }

        inline void reflect(const Vector3f& normal) {
            *this = *this - normal * (2.0f * dot(normal));
        }

        inline Vector3f reflected(const Vector3f& normal) const {
            Vector3f result = *this;
            result.reflect(normal);
            return result;
        }

        inline void normalizeInto(Vector3f& out) const {
            out = normalized();
        }
    };

}
