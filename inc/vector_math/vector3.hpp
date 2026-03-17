#pragma once

#include<vector_math/vec.hpp>

namespace systems::leal::vector_math {

    /// 3D vector of type @p DATA_TYPE.
    ///
    /// Inherits all Vec<DATA_TYPE,3> operations.
    /// Components: `data[0]` = x, `data[1]` = y, `data[2]` = z.
    template<class DATA_TYPE>
    class Vector3:public Vec<DATA_TYPE,3> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Vector3() = default;

        /// Constructs all elements from a single @p value (broadcast/splat).
        Vector3(DATA_TYPE value);

        /// Constructs from explicit x, y, z values.
        Vector3(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z);

        // ── Named accessors ───────────────────────────────────────────────────

        DATA_TYPE& x() { return this->data[0]; }              ///< Reference to the x component.
        const DATA_TYPE& x() const { return this->data[0]; } ///< Const reference to x.
        DATA_TYPE& y() { return this->data[1]; }              ///< Reference to the y component.
        const DATA_TYPE& y() const { return this->data[1]; } ///< Const reference to y.
        DATA_TYPE& z() { return this->data[2]; }              ///< Reference to the z component.
        const DATA_TYPE& z() const { return this->data[2]; } ///< Const reference to z.

        // ── Operators (typed overrides returning Vector3) ─────────────────────

        /// Component-wise addition.
        Vector3<DATA_TYPE> operator+(const Vector3<DATA_TYPE>& other) const;
        /// Component-wise subtraction.
        Vector3<DATA_TYPE> operator-(const Vector3<DATA_TYPE>& other) const;
        /// Negation.
        Vector3<DATA_TYPE> operator-() const;
        /// Scalar multiplication: `v * s`.
        Vector3<DATA_TYPE> operator*(DATA_TYPE scalar) const;
        /// Scalar multiplication (commutative): `s * v`.
        friend Vector3<DATA_TYPE> operator*(DATA_TYPE scalar, const Vector3<DATA_TYPE>& v) {
            return v * scalar;
        }
        /// Scalar division.
        Vector3<DATA_TYPE> operator/(DATA_TYPE scalar) const;

        // ── Methods ───────────────────────────────────────────────────────────

        /// Returns the cross product `lhs × rhs` as a new Vector3.
        static Vector3<DATA_TYPE> cross(const Vector3<DATA_TYPE>& lhs, const Vector3<DATA_TYPE>& rhs);

        /// Returns the dot product `this · rhs`.
        DATA_TYPE dot(const Vector3<DATA_TYPE>& rhs) const;

        /// Static version: returns `lhs · rhs`.
        static DATA_TYPE dot(const Vector3<DATA_TYPE>& lhs, const Vector3<DATA_TYPE>& rhs);

        /// Returns the unsigned angle in radians between this vector and @p other. Range [0, π].
        DATA_TYPE angleTo(const Vector3<DATA_TYPE>& other) const;

        /// Returns the signed angle in radians from this vector to @p other,
        /// measured around @p normal using the right-hand rule. Range (-π, π].
        DATA_TYPE angleToSigned(const Vector3<DATA_TYPE>& other, const Vector3<DATA_TYPE>& normal) const;

        /// Reflects this vector in-place about @p normal (which must be normalized).
        /// Formula: `v = v - 2 * dot(v, n) * n`.
        void reflect(const Vector3<DATA_TYPE>& normal);

        /// Returns a copy of this vector reflected about @p normal.
        Vector3<DATA_TYPE> reflected(const Vector3<DATA_TYPE>& normal) const;

        /// Writes the normalized copy of this vector into @p out.
        void normalizeInto(Vector3<DATA_TYPE>& out) const;

    };

    template<class DATA_TYPE>
    Vector3<DATA_TYPE>::Vector3(DATA_TYPE value):Vec<DATA_TYPE,3>(value) {}

    template<class DATA_TYPE>
    Vector3<DATA_TYPE>::Vector3(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z) {
        this->data[0] = x;
        this->data[1] = y;
        this->data[2] = z;
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator+(const Vector3<DATA_TYPE>& other) const {
        return Vector3<DATA_TYPE>(
            this->data[0] + other.data[0],
            this->data[1] + other.data[1],
            this->data[2] + other.data[2]
        );
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator-(const Vector3<DATA_TYPE> &other) const {
        return Vector3<DATA_TYPE>(
            this->data[0] - other.data[0],
            this->data[1] - other.data[1],
            this->data[2] - other.data[2]
        );
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator-() const {
        return Vector3<DATA_TYPE>(-this->data[0], -this->data[1], -this->data[2]);
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator*(DATA_TYPE scalar) const {
        return Vector3<DATA_TYPE>(
            this->data[0] * scalar,
            this->data[1] * scalar,
            this->data[2] * scalar
        );
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator/(DATA_TYPE scalar) const {
        return Vector3<DATA_TYPE>(
            this->data[0] / scalar,
            this->data[1] / scalar,
            this->data[2] / scalar
        );
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::cross(const Vector3<DATA_TYPE> &lhs, const Vector3<DATA_TYPE> &rhs) {
        return {
            (lhs.data[1] * rhs.data[2]) - (lhs.data[2] * rhs.data[1]),
            (lhs.data[2] * rhs.data[0]) - (lhs.data[0] * rhs.data[2]),
            (lhs.data[0] * rhs.data[1]) - (lhs.data[1] * rhs.data[0])
        };
    }

    template<class DATA_TYPE>
    DATA_TYPE Vector3<DATA_TYPE>::dot(const Vector3<DATA_TYPE> &rhs) const {
        return (this->data[0] * rhs.data[0]) + (this->data[1] * rhs.data[1]) + (this->data[2] * rhs.data[2]);
    }

    template<class DATA_TYPE>
    DATA_TYPE Vector3<DATA_TYPE>::dot(const Vector3<DATA_TYPE> &lhs, const Vector3<DATA_TYPE> &rhs) {
        return lhs.dot(rhs);
    }

    template<class DATA_TYPE>
    DATA_TYPE Vector3<DATA_TYPE>::angleTo(const Vector3<DATA_TYPE>& other) const {
        DATA_TYPE lenProduct = this->length() * other.length();
        if (isZero<DATA_TYPE>(lenProduct)) return DATA_TYPE(0);
        DATA_TYPE cosAngle = dot(other) / lenProduct;
        cosAngle = cosAngle < DATA_TYPE(-1) ? DATA_TYPE(-1) : (cosAngle > DATA_TYPE(1) ? DATA_TYPE(1) : cosAngle);
        return std::acos(cosAngle);
    }

    template<class DATA_TYPE>
    DATA_TYPE Vector3<DATA_TYPE>::angleToSigned(const Vector3<DATA_TYPE>& other, const Vector3<DATA_TYPE>& normal) const {
        DATA_TYPE angle = angleTo(other);
        auto c = cross(*this, other);
        return dot(c, normal) < DATA_TYPE(0) ? -angle : angle;
    }

    template<class DATA_TYPE>
    void Vector3<DATA_TYPE>::reflect(const Vector3<DATA_TYPE>& normal) {
        DATA_TYPE d = DATA_TYPE(2) * dot(normal);
        this->data[0] -= d * normal.data[0];
        this->data[1] -= d * normal.data[1];
        this->data[2] -= d * normal.data[2];
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::reflected(const Vector3<DATA_TYPE>& normal) const {
        auto result = *this;
        result.reflect(normal);
        return result;
    }

    template<class DATA_TYPE>
    void Vector3<DATA_TYPE>::normalizeInto(Vector3<DATA_TYPE>& out) const {
        out = *this;
        out.normalize();
    }

}
