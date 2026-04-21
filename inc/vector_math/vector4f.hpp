#pragma once

#include <vector_math/vector4.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 16-byte aligned float 4D vector.
    ///
    /// Identical to Vector4<float> but with `alignas(16)` to satisfy SSE/NEON load
    /// requirements. Use this type when passing vectors to Matrix4f operations.
    class alignas(16) Vector4f:public Vector4<float> {
    public:
        /// Default-constructs all elements to zero.
        Vector4f() = default;

        /// Constructs from explicit x, y, z, w values.
        Vector4f(float x, float y, float z, float w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

        /// Component-wise addition.
        inline Vector4f operator+(const Vector4f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data,
                    _mm_add_ps(_mm_load_ps(this->data), _mm_load_ps(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data,
                    vaddq_f32(vld1q_f32(this->data), vld1q_f32(rhs.data)));
                return result;
            #else
                return Vector4f(
                    this->data[0] + rhs.data[0], this->data[1] + rhs.data[1],
                    this->data[2] + rhs.data[2], this->data[3] + rhs.data[3]
                );
            #endif
        }

        /// Component-wise subtraction.
        inline Vector4f operator-(const Vector4f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data,
                    _mm_sub_ps(_mm_load_ps(this->data), _mm_load_ps(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data,
                    vsubq_f32(vld1q_f32(this->data), vld1q_f32(rhs.data)));
                return result;
            #else
                return Vector4f(
                    this->data[0] - rhs.data[0], this->data[1] - rhs.data[1],
                    this->data[2] - rhs.data[2], this->data[3] - rhs.data[3]
                );
            #endif
        }

        /// Negation.
        inline Vector4f operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data,
                    _mm_xor_ps(_mm_load_ps(this->data), _mm_set1_ps(-0.0f)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data, vnegq_f32(vld1q_f32(this->data)));
                return result;
            #else
                return Vector4f(-this->data[0], -this->data[1], -this->data[2], -this->data[3]);
            #endif
        }

        /// Scalar multiplication: `v * s`.
        inline Vector4f operator*(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data,
                    _mm_mul_ps(_mm_load_ps(this->data), _mm_set1_ps(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(this->data), scalar));
                return result;
            #else
                return Vector4f(
                    this->data[0] * scalar, this->data[1] * scalar,
                    this->data[2] * scalar, this->data[3] * scalar
                );
            #endif
        }

        /// Scalar multiplication (commutative): `s * v`.
        friend inline Vector4f operator*(float scalar, const Vector4f& v) { return v * scalar; }

        /// Scalar division.
        inline Vector4f operator/(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data,
                    _mm_div_ps(_mm_load_ps(this->data), _mm_set1_ps(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                float inv = 1.0f / scalar;
                vst1q_f32(result.data, vmulq_n_f32(vld1q_f32(this->data), inv));
                return result;
            #else
                return Vector4f(
                    this->data[0] / scalar, this->data[1] / scalar,
                    this->data[2] / scalar, this->data[3] / scalar
                );
            #endif
        }

        // dot() inherited from Vector4<float> — compiler auto-vectorizes scalar better than hand-written SSE hadd
        /// Returns the squared Euclidean length.
        inline float lengthSquared() const { return dot(*this); }

        /// In-place normalization using the SIMD dot path.
        inline void normalize() {
            float lenSqr = lengthSquared();
            if (isZero<float>(lenSqr)) return;
            *this = *this * (1.0f / std::sqrt(lenSqr));
        }

        /// Returns a normalized copy.
        inline Vector4f normalized() const {
            Vector4f result = *this;
            result.normalize();
            return result;
        }

        /// Returns the squared Euclidean distance to @p other.
        inline float distanceToSquared(const Vector4f& other) const {
            Vector4f delta = *this - other;
            return delta.dot(delta);
        }

        /// Replaces each component with its absolute value.
        inline void absolute() {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 value = _mm_load_ps(this->data);
                __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
                _mm_store_ps(this->data, _mm_and_ps(value, mask));
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                vst1q_f32(this->data, vabsq_f32(vld1q_f32(this->data)));
            #else
                this->data[0] = std::abs(this->data[0]);
                this->data[1] = std::abs(this->data[1]);
                this->data[2] = std::abs(this->data[2]);
                this->data[3] = std::abs(this->data[3]);
            #endif
        }

        /// Clamps each component to [minVal, maxVal].
        inline void clamp(float minVal, float maxVal) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 value = _mm_load_ps(this->data);
                __m128 minV = _mm_set1_ps(minVal);
                __m128 maxV = _mm_set1_ps(maxVal);
                _mm_store_ps(this->data, _mm_min_ps(_mm_max_ps(value, minV), maxV));
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                float32x4_t value = vld1q_f32(this->data);
                float32x4_t minV = vdupq_n_f32(minVal);
                float32x4_t maxV = vdupq_n_f32(maxVal);
                vst1q_f32(this->data, vminq_f32(vmaxq_f32(value, minV), maxV));
            #else
                this->data[0] = this->data[0] < minVal ? minVal : (this->data[0] > maxVal ? maxVal : this->data[0]);
                this->data[1] = this->data[1] < minVal ? minVal : (this->data[1] > maxVal ? maxVal : this->data[1]);
                this->data[2] = this->data[2] < minVal ? minVal : (this->data[2] > maxVal ? maxVal : this->data[2]);
                this->data[3] = this->data[3] < minVal ? minVal : (this->data[3] > maxVal ? maxVal : this->data[3]);
            #endif
        }

        /// Clamps each component independently to [minVec[i], maxVec[i]].
        inline void clamp(const Vector4f& minVec, const Vector4f& maxVec) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 value = _mm_load_ps(this->data);
                __m128 minV = _mm_load_ps(minVec.data);
                __m128 maxV = _mm_load_ps(maxVec.data);
                _mm_store_ps(this->data, _mm_min_ps(_mm_max_ps(value, minV), maxV));
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                float32x4_t value = vld1q_f32(this->data);
                float32x4_t minV = vld1q_f32(minVec.data);
                float32x4_t maxV = vld1q_f32(maxVec.data);
                vst1q_f32(this->data, vminq_f32(vmaxq_f32(value, minV), maxV));
            #else
                this->data[0] = this->data[0] < minVec.data[0] ? minVec.data[0] : (this->data[0] > maxVec.data[0] ? maxVec.data[0] : this->data[0]);
                this->data[1] = this->data[1] < minVec.data[1] ? minVec.data[1] : (this->data[1] > maxVec.data[1] ? maxVec.data[1] : this->data[1]);
                this->data[2] = this->data[2] < minVec.data[2] ? minVec.data[2] : (this->data[2] > maxVec.data[2] ? maxVec.data[2] : this->data[2]);
                this->data[3] = this->data[3] < minVec.data[3] ? minVec.data[3] : (this->data[3] > maxVec.data[3] ? maxVec.data[3] : this->data[3]);
            #endif
        }

        /// Returns the component-wise minimum.
        static inline Vector4f min(const Vector4f& a, const Vector4f& b) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data, _mm_min_ps(_mm_load_ps(a.data), _mm_load_ps(b.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data, vminq_f32(vld1q_f32(a.data), vld1q_f32(b.data)));
                return result;
            #else
                return Vector4f(
                    a.data[0] < b.data[0] ? a.data[0] : b.data[0],
                    a.data[1] < b.data[1] ? a.data[1] : b.data[1],
                    a.data[2] < b.data[2] ? a.data[2] : b.data[2],
                    a.data[3] < b.data[3] ? a.data[3] : b.data[3]
                );
            #endif
        }

        /// Returns the component-wise maximum.
        static inline Vector4f max(const Vector4f& a, const Vector4f& b) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                _mm_store_ps(result.data, _mm_max_ps(_mm_load_ps(a.data), _mm_load_ps(b.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                vst1q_f32(result.data, vmaxq_f32(vld1q_f32(a.data), vld1q_f32(b.data)));
                return result;
            #else
                return Vector4f(
                    a.data[0] > b.data[0] ? a.data[0] : b.data[0],
                    a.data[1] > b.data[1] ? a.data[1] : b.data[1],
                    a.data[2] > b.data[2] ? a.data[2] : b.data[2],
                    a.data[3] > b.data[3] ? a.data[3] : b.data[3]
                );
            #endif
        }

        /// Linear interpolation: returns a + t * (b - a).
        static inline Vector4f lerp(const Vector4f& a, const Vector4f& b, float t) {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4f result;
                __m128 av = _mm_load_ps(a.data);
                __m128 bv = _mm_load_ps(b.data);
                __m128 tv = _mm_set1_ps(t);
                _mm_store_ps(result.data, _mm_add_ps(av, _mm_mul_ps(tv, _mm_sub_ps(bv, av))));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Vector4f result;
                float32x4_t av = vld1q_f32(a.data);
                float32x4_t bv = vld1q_f32(b.data);
                float32x4_t tv = vdupq_n_f32(t);
                vst1q_f32(result.data, vaddq_f32(av, vmulq_f32(tv, vsubq_f32(bv, av))));
                return result;
            #else
                return Vector4f(
                    a.data[0] + t * (b.data[0] - a.data[0]),
                    a.data[1] + t * (b.data[1] - a.data[1]),
                    a.data[2] + t * (b.data[2] - a.data[2]),
                    a.data[3] + t * (b.data[3] - a.data[3])
                );
            #endif
        }

    };

}
