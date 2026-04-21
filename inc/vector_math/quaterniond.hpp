#pragma once

#include <vector_math/quaternion.hpp>
#include <vector_math/vector3d.hpp>
#include <vector_math/vector4d.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 32-byte aligned double quaternion with SIMD-accelerated component-wise hot paths.
    class alignas(32) Quaterniond : public Quaternion<double> {
    public:
        using Quaternion<double>::rotate;
        using Quaternion<double>::rotated;

        Quaterniond() = default;

        Quaterniond(double x, double y, double z, double w) : Quaternion<double>(x, y, z, w) {}

        Quaterniond(const Quaternion<double>& q) : Quaternion<double>(q.data[0], q.data[1], q.data[2], q.data[3]) {}

        static Quaterniond identity() { return Quaterniond(0.0, 0.0, 0.0, 1.0); }

        static Quaterniond axisAngle(const Vector3<double>& axis, double angle) {
            double halfAngle = angle * 0.5;
            double sinHalf = std::sin(halfAngle);
            return Quaterniond(
                axis.data[0] * sinHalf,
                axis.data[1] * sinHalf,
                axis.data[2] * sinHalf,
                std::cos(halfAngle)
            );
        }

        static Quaterniond axisAngle(const Vector3d& axis, double angle) {
            return axisAngle(Vector3<double>(axis), angle);
        }

        static Quaterniond slerp(const Quaterniond& a, const Quaterniond& b, double t) {
            return Quaterniond(Quaternion<double>::slerp(a, b, t));
        }

        inline Quaterniond operator+(const Quaterniond& rhs) const {
            Vector4d lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4d rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            auto result = lhsV + rhsV;
            return Quaterniond(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaterniond operator-(const Quaterniond& rhs) const {
            Vector4d lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4d rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            auto result = lhsV - rhsV;
            return Quaterniond(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaterniond operator-() const {
            Vector4d v(this->data[0], this->data[1], this->data[2], this->data[3]);
            auto result = -v;
            return Quaterniond(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        inline Quaterniond operator*(double scalar) const {
            Vector4d v(this->data[0], this->data[1], this->data[2], this->data[3]);
            auto result = v * scalar;
            return Quaterniond(result.data[0], result.data[1], result.data[2], result.data[3]);
        }

        friend inline Quaterniond operator*(double scalar, const Quaterniond& q) { return q * scalar; }

        /// Quaternion composition (Hamilton product): applies @p rhs first, then this.
        inline Quaterniond operator*(const Quaterniond& rhs) const {
            return Quaterniond(
                this->data[3]*rhs.data[0] + this->data[0]*rhs.data[3] + this->data[1]*rhs.data[2] - this->data[2]*rhs.data[1],
                this->data[3]*rhs.data[1] - this->data[0]*rhs.data[2] + this->data[1]*rhs.data[3] + this->data[2]*rhs.data[0],
                this->data[3]*rhs.data[2] + this->data[0]*rhs.data[1] - this->data[1]*rhs.data[0] + this->data[2]*rhs.data[3],
                this->data[3]*rhs.data[3] - this->data[0]*rhs.data[0] - this->data[1]*rhs.data[1] - this->data[2]*rhs.data[2]
            );
        }

        inline double dot(const Quaterniond& rhs) const {
            Vector4d lhsV(this->data[0], this->data[1], this->data[2], this->data[3]);
            Vector4d rhsV(rhs.data[0], rhs.data[1], rhs.data[2], rhs.data[3]);
            return lhsV.dot(rhsV);
        }

        static inline double dot(const Quaterniond& lhs, const Quaterniond& rhs) { return lhs.dot(rhs); }

        inline double lengthSquared() const { return dot(*this); }

        inline void conjugate() {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                alignas(32) double buffer[4] = { this->data[0], this->data[1], this->data[2], this->data[3] };
                __m256d v = _mm256_load_pd(buffer);
                __m256d signMask = _mm256_set_pd(0.0, -0.0, -0.0, -0.0);
                _mm256_store_pd(buffer, _mm256_xor_pd(v, signMask));
                this->data[0] = buffer[0];
                this->data[1] = buffer[1];
                this->data[2] = buffer[2];
                this->data[3] = buffer[3];
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                alignas(32) double buffer[4] = { this->data[0], this->data[1], this->data[2], this->data[3] };
                float64x2_t signLo = { -1.0, -1.0 };
                float64x2_t signHi = { -1.0, 1.0 };
                vst1q_f64(&buffer[0], vmulq_f64(vld1q_f64(&buffer[0]), signLo));
                vst1q_f64(&buffer[2], vmulq_f64(vld1q_f64(&buffer[2]), signHi));
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

        inline Quaterniond conjugated() const {
            Quaterniond q = *this;
            q.conjugate();
            return q;
        }

        inline void inverse() {
            double lenSqr = lengthSquared();
            if (isZero<double>(lenSqr)) return;
            Quaterniond q = conjugated() * (1.0 / lenSqr);
            this->data[0] = q.data[0];
            this->data[1] = q.data[1];
            this->data[2] = q.data[2];
            this->data[3] = q.data[3];
        }

        inline Quaterniond inverted() const {
            Quaterniond q = *this;
            q.inverse();
            return q;
        }

        inline Vector3d rotated(const Vector3d& v) const {
            Vector3d qv(this->data[0], this->data[1], this->data[2]);
            double qw = this->data[3];
            Vector3d t = Vector3d::cross(qv, v) * 2.0;
            return v + (t * qw) + Vector3d::cross(qv, t);
        }

        inline void rotate(Vector3d& v) const {
            v = rotated(v);
        }
    };

}
