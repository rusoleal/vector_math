#pragma once

#include<vector_math/vector3.hpp>
#include<vector_math/vector4.hpp>

namespace systems::leal::vector_math {

    /// Unit quaternion representing a 3D rotation.
    ///
    /// Stored as (x, y, z, w) where x/y/z are the imaginary (vector) part and
    /// w is the scalar (real) part. Inherits all Vector4<DATA_TYPE> operations.
    ///
    /// Convention: `q = w + xi + yj + zk`.
    /// For a rotation of @p angle radians around unit axis @p n:
    ///   `x = n.x*sin(angle/2)`, `y = n.y*sin(angle/2)`, `z = n.z*sin(angle/2)`, `w = cos(angle/2)`.
    template<class DATA_TYPE>
    class Quaternion:public Vector4<DATA_TYPE> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs to a zero quaternion.
        /// Use Quaternion::identity() for a valid no-rotation quaternion.
        Quaternion() = default;

        /// Constructs directly from components (x, y, z, w).
        Quaternion(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z, DATA_TYPE w)
            :Vector4<DATA_TYPE>(x,y,z,w) {}

        // ── Static factories ──────────────────────────────────────────────────

        /// Returns the identity quaternion (0, 0, 0, 1) — represents no rotation.
        static Quaternion<DATA_TYPE> identity() {
            return Quaternion<DATA_TYPE>(0, 0, 0, 1);
        }

        /// Creates a quaternion from a normalized @p axis and rotation @p angle (radians).
        /// @pre @p axis must be unit length.
        static Quaternion<DATA_TYPE> axisAngle(const Vector3<DATA_TYPE>& axis, DATA_TYPE angle) {
            DATA_TYPE halfAngle = angle * DATA_TYPE(0.5);
            DATA_TYPE sinHalf = std::sin(halfAngle);
            return Quaternion<DATA_TYPE>(
                axis.data[0] * sinHalf,
                axis.data[1] * sinHalf,
                axis.data[2] * sinHalf,
                std::cos(halfAngle)
            );
        }

        /// Spherical linear interpolation between @p a and @p b.
        /// @param t  Blend factor in [0, 1]; 0 returns @p a, 1 returns @p b.
        /// Takes the shortest arc; handles antipodal quaternions automatically.
        static Quaternion<DATA_TYPE> slerp(const Quaternion<DATA_TYPE>& a, const Quaternion<DATA_TYPE>& b, DATA_TYPE t) {
            DATA_TYPE cosHalfTheta = a.data[0]*b.data[0] + a.data[1]*b.data[1]
                                   + a.data[2]*b.data[2] + a.data[3]*b.data[3];

            Quaternion<DATA_TYPE> bAdjusted = b;
            if (cosHalfTheta < DATA_TYPE(0)) {
                bAdjusted.data[0] = -b.data[0];
                bAdjusted.data[1] = -b.data[1];
                bAdjusted.data[2] = -b.data[2];
                bAdjusted.data[3] = -b.data[3];
                cosHalfTheta = -cosHalfTheta;
            }

            if (cosHalfTheta >= DATA_TYPE(1)) return a;

            DATA_TYPE halfTheta = std::acos(cosHalfTheta);
            DATA_TYPE sinHalfTheta = std::sqrt(DATA_TYPE(1) - cosHalfTheta * cosHalfTheta);

            if (std::abs(sinHalfTheta) < DATA_TYPE(0.001)) {
                return Quaternion<DATA_TYPE>(
                    a.data[0] + t*(bAdjusted.data[0]-a.data[0]),
                    a.data[1] + t*(bAdjusted.data[1]-a.data[1]),
                    a.data[2] + t*(bAdjusted.data[2]-a.data[2]),
                    a.data[3] + t*(bAdjusted.data[3]-a.data[3])
                );
            }

            DATA_TYPE ratioA = std::sin((DATA_TYPE(1)-t) * halfTheta) / sinHalfTheta;
            DATA_TYPE ratioB = std::sin(t * halfTheta) / sinHalfTheta;
            return Quaternion<DATA_TYPE>(
                a.data[0]*ratioA + bAdjusted.data[0]*ratioB,
                a.data[1]*ratioA + bAdjusted.data[1]*ratioB,
                a.data[2]*ratioA + bAdjusted.data[2]*ratioB,
                a.data[3]*ratioA + bAdjusted.data[3]*ratioB
            );
        }

        // ── Operators ─────────────────────────────────────────────────────────

        /// Quaternion composition (Hamilton product): applies @p rhs rotation first, then this.
        Quaternion<DATA_TYPE> operator*(const Quaternion<DATA_TYPE>& rhs) const {
            return Quaternion<DATA_TYPE>(
                this->data[3]*rhs.data[0] + this->data[0]*rhs.data[3] + this->data[1]*rhs.data[2] - this->data[2]*rhs.data[1],
                this->data[3]*rhs.data[1] - this->data[0]*rhs.data[2] + this->data[1]*rhs.data[3] + this->data[2]*rhs.data[0],
                this->data[3]*rhs.data[2] + this->data[0]*rhs.data[1] - this->data[1]*rhs.data[0] + this->data[2]*rhs.data[3],
                this->data[3]*rhs.data[3] - this->data[0]*rhs.data[0] - this->data[1]*rhs.data[1] - this->data[2]*rhs.data[2]
            );
        }

        /// Component-wise addition (used for blending, not a rotation operation).
        Quaternion<DATA_TYPE> operator+(const Quaternion<DATA_TYPE>& rhs) const {
            return Quaternion<DATA_TYPE>(
                this->data[0]+rhs.data[0], this->data[1]+rhs.data[1],
                this->data[2]+rhs.data[2], this->data[3]+rhs.data[3]
            );
        }

        /// Component-wise subtraction.
        Quaternion<DATA_TYPE> operator-(const Quaternion<DATA_TYPE>& rhs) const {
            return Quaternion<DATA_TYPE>(
                this->data[0]-rhs.data[0], this->data[1]-rhs.data[1],
                this->data[2]-rhs.data[2], this->data[3]-rhs.data[3]
            );
        }

        /// Negates all four components (same rotation, opposite hemisphere).
        Quaternion<DATA_TYPE> operator-() const {
            return Quaternion<DATA_TYPE>(-this->data[0], -this->data[1], -this->data[2], -this->data[3]);
        }

        /// Scales all four components by @p scalar.
        Quaternion<DATA_TYPE> operator*(DATA_TYPE scalar) const {
            return Quaternion<DATA_TYPE>(
                this->data[0]*scalar, this->data[1]*scalar,
                this->data[2]*scalar, this->data[3]*scalar
            );
        }

        // ── Methods ───────────────────────────────────────────────────────────

        /// Sets this quaternion to the rotation described by @p axis and @p angle (radians).
        /// @pre @p axis must be unit length.
        void setAxisAngle(const Vector3<DATA_TYPE>& axis, DATA_TYPE angle) {
            *this = axisAngle(axis, angle);
        }

        /// Conjugates this quaternion in-place: negates x, y, z; keeps w.
        /// For a unit quaternion, the conjugate equals the inverse.
        void conjugate() {
            this->data[0] = -this->data[0];
            this->data[1] = -this->data[1];
            this->data[2] = -this->data[2];
        }

        /// Returns a conjugated copy without modifying this quaternion.
        Quaternion<DATA_TYPE> conjugated() const {
            return Quaternion<DATA_TYPE>(-this->data[0], -this->data[1], -this->data[2], this->data[3]);
        }

        /// Inverts this quaternion in-place: `q^-1 = conjugate(q) / |q|²`.
        /// No-op if the quaternion has zero length.
        void inverse() {
            DATA_TYPE lenSqr = this->lengthSquared();
            if (isZero<DATA_TYPE>(lenSqr)) return;
            conjugate();
            DATA_TYPE invLenSqr = DATA_TYPE(1) / lenSqr;
            this->data[0] *= invLenSqr;
            this->data[1] *= invLenSqr;
            this->data[2] *= invLenSqr;
            this->data[3] *= invLenSqr;
        }

        /// Returns an inverted copy without modifying this quaternion.
        Quaternion<DATA_TYPE> inverted() const {
            auto q = *this;
            q.inverse();
            return q;
        }

        /// Returns @p v rotated by this quaternion.
        /// Uses `v' = v + w*t + cross(qv, t)` where `t = 2*cross(qv, v)`.
        /// @pre This quaternion should be unit length for a pure rotation.
        Vector3<DATA_TYPE> rotated(const Vector3<DATA_TYPE>& v) const {
            Vector3<DATA_TYPE> qv(this->data[0], this->data[1], this->data[2]);
            DATA_TYPE qw = this->data[3];
            auto crossQvV = Vector3<DATA_TYPE>::cross(qv, v);
            Vector3<DATA_TYPE> t(crossQvV.data[0]*DATA_TYPE(2), crossQvV.data[1]*DATA_TYPE(2), crossQvV.data[2]*DATA_TYPE(2));
            auto t2 = Vector3<DATA_TYPE>::cross(qv, t);
            return Vector3<DATA_TYPE>(
                v.data[0] + qw*t.data[0] + t2.data[0],
                v.data[1] + qw*t.data[1] + t2.data[1],
                v.data[2] + qw*t.data[2] + t2.data[2]
            );
        }

        /// Rotates @p v in-place by this quaternion.
        void rotate(Vector3<DATA_TYPE>& v) const {
            v = rotated(v);
        }

        /// Returns the normalized rotation axis.
        /// Returns a zero vector if this quaternion has no rotation (w ≈ ±1).
        Vector3<DATA_TYPE> axis() const {
            Vector3<DATA_TYPE> axisVec(this->data[0], this->data[1], this->data[2]);
            axisVec.normalize();
            return axisVec;
        }

        /// Returns the rotation angle in radians derived from the w component. Range [0, 2π].
        DATA_TYPE angle() const {
            DATA_TYPE w = this->data[3];
            w = w < DATA_TYPE(-1) ? DATA_TYPE(-1) : (w > DATA_TYPE(1) ? DATA_TYPE(1) : w);
            return DATA_TYPE(2) * std::acos(w);
        }

    };

}
