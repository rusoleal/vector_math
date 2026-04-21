#pragma once

#include <vector_math/quaternion.hpp>
#include <vector_math/vector3f.hpp>
#include <vector_math/vector4f.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 16-byte aligned float quaternion with SIMD-accelerated hot paths.
    class alignas(16) Quaternionf : public Quaternion<float> {
    public:
        using Quaternion<float>::rotate;
        using Quaternion<float>::rotated;

        Quaternionf() = default;

        Quaternionf(float x, float y, float z, float w) : Quaternion<float>(x, y, z, w) {}

        Quaternionf(const Quaternion<float>& q) : Quaternion<float>(q.data[0], q.data[1], q.data[2], q.data[3]) {}

        static Quaternionf identity() { return Quaternionf(0.0f, 0.0f, 0.0f, 1.0f); }

        static Quaternionf axisAngle(const Vector3<float>& axis, float angle) {
            float halfAngle = angle * 0.5f;
            float sinHalf = std::sin(halfAngle);
            return Quaternionf(
                axis.data[0] * sinHalf,
                axis.data[1] * sinHalf,
                axis.data[2] * sinHalf,
                std::cos(halfAngle)
            );
        }

        static Quaternionf axisAngle(const Vector3f& axis, float angle) {
            return axisAngle(Vector3<float>(axis), angle);
        }

        static Quaternionf slerp(const Quaternionf& a, const Quaternionf& b, float t) {
            return Quaternionf(Quaternion<float>::slerp(a, b, t));
        }

        inline Quaternionf operator+(const Quaternionf& rhs) const {
            Vector4f lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4f rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            auto result = lhsV + rhsV;
            return Quaternionf(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaternionf operator-(const Quaternionf& rhs) const {
            Vector4f lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4f rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            auto result = lhsV - rhsV;
            return Quaternionf(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaternionf operator-() const {
            Vector4f v(this->data[0], this->data[1], this->data[2], this->data[3]);
            auto result = -v;
            return Quaternionf(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaternionf operator*(float scalar) const {
            Vector4f v(this->data[0], this->data[1], this->data[2], this->data[3]);
            auto result = v * scalar;
            return Quaternionf(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        friend inline Quaternionf operator*(float scalar, const Quaternionf& q) { return q * scalar; }

        /// Quaternion composition (Hamilton product): applies @p rhs first, then this.
        inline Quaternionf operator*(const Quaternionf& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                alignas(16) float rhsBuffer[4] = { rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3] };
                __m128 r = _mm_load_ps(rhsBuffer);
                __m128 w = _mm_mul_ps(_mm_set1_ps(this->data[3]), r);

                __m128 xTerm = _mm_mul_ps(
                    _mm_set1_ps(this->data[0]),
                    _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(0, 1, 2, 3)),
                               _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f))
                );
                __m128 yTerm = _mm_mul_ps(
                    _mm_set1_ps(this->data[1]),
                    _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(1, 0, 3, 2)),
                               _mm_set_ps(-1.0f, -1.0f, 1.0f, 1.0f))
                );
                __m128 zTerm = _mm_mul_ps(
                    _mm_set1_ps(this->data[2]),
                    _mm_mul_ps(_mm_shuffle_ps(r, r, _MM_SHUFFLE(2, 3, 0, 1)),
                               _mm_set_ps(-1.0f, 1.0f, 1.0f, -1.0f))
                );

                __m128 resultVec = _mm_add_ps(_mm_add_ps(w, xTerm), _mm_add_ps(yTerm, zTerm));
                alignas(16) float out[4];
                _mm_store_ps(out, resultVec);
                return Quaternionf(out[0], out[1], out[2], out[3]);
            #else
                return Quaternionf(
                    this->data[3]*rhs.data[0] + this->data[0]*rhs.data[3] + this->data[1]*rhs.data[2] - this->data[2]*rhs.data[1],
                    this->data[3]*rhs.data[1] - this->data[0]*rhs.data[2] + this->data[1]*rhs.data[3] + this->data[2]*rhs.data[0],
                    this->data[3]*rhs.data[2] + this->data[0]*rhs.data[1] - this->data[1]*rhs.data[0] + this->data[2]*rhs.data[3],
                    this->data[3]*rhs.data[3] - this->data[0]*rhs.data[0] - this->data[1]*rhs.data[1] - this->data[2]*rhs.data[2]
                );
            #endif
        }

        inline float dot(const Quaternionf& rhs) const {
            Vector4f lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4f rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            return lhsV.dot(rhsV);
        }

        static inline float dot(const Quaternionf& lhs, const Quaternionf& rhs) { return lhs.dot(rhs); }

        inline float lengthSquared() const { return dot(*this); }

        inline void conjugate() {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                alignas(16) float buffer[4] = { this->data[0], this->data[1], this->data[2], this->data[3] };
                __m128 v = _mm_load_ps(buffer);
                __m128 signMask = _mm_set_ps(0.0f, -0.0f, -0.0f, -0.0f);
                _mm_store_ps(buffer, _mm_xor_ps(v, signMask));
                this->data[0] = buffer[0];
                this->data[1] = buffer[1];
                this->data[2] = buffer[2];
                this->data[3] = buffer[3];
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                float32x4_t signs = { -1.0f, -1.0f, -1.0f, 1.0f };
                alignas(16) float buffer[4] = { this->data[0], this->data[1], this->data[2], this->data[3] };
                vst1q_f32(buffer, vmulq_f32(vld1q_f32(buffer), signs));
                this->data[0] = buffer[0];
                this->data[1] = buffer[1];
                this->data[2] = buffer[2];
                this->data[3] = buffer[3];
            #else
                this->data[0] = -this->data[0];
                this->data[1] = -this->data[1];
                this->data[2] = -this->data[2];
            #endif
        }

        inline Quaternionf conjugated() const {
            Quaternionf q = *this;
            q.conjugate();
            return q;
        }

        inline void inverse() {
            float lenSqr = lengthSquared();
            if (isZero<float>(lenSqr)) return;
            Quaternionf q = conjugated() * (1.0f / lenSqr);
            this->data[0] = q.data[0];
            this->data[1] = q.data[1];
            this->data[2] = q.data[2];
            this->data[3] = q.data[3];
        }

        inline Quaternionf inverted() const {
            Quaternionf q = *this;
            q.inverse();
            return q;
        }

        inline Vector3f rotated(const Vector3f& v) const {
            Vector3f qv(this->data[0], this->data[1], this->data[2]);
            float qw = this->data[3];
            Vector3f t = Vector3f::cross(qv, v) * 2.0f;
            return v + (t * qw) + Vector3f::cross(qv, t);
        }

        inline void rotate(Vector3f& v) const {
            v = rotated(v);
        }
    };

}
