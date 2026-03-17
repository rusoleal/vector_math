#pragma once

#include<vector_math/vec.hpp>

namespace systems::leal::vector_math {

    /// 2D vector of type @p DATA_TYPE.
    ///
    /// Inherits all Vec<DATA_TYPE,2> operations.
    /// Components are stored as `data[0]` (x) and `data[1]` (y).
    template<class DATA_TYPE>
    class Vector2:public Vec<DATA_TYPE,2> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Vector2() = default;

        /// Constructs from explicit x, y values.
        Vector2(DATA_TYPE x, DATA_TYPE y) {
            this->data[0] = x;
            this->data[1] = y;
        }

        // ── Named accessors ───────────────────────────────────────────────────

        DATA_TYPE& x() { return this->data[0]; }         ///< Reference to the x component.
        const DATA_TYPE& x() const { return this->data[0]; } ///< Const reference to x.
        DATA_TYPE& y() { return this->data[1]; }         ///< Reference to the y component.
        const DATA_TYPE& y() const { return this->data[1]; } ///< Const reference to y.

        // ── Operators (typed overrides returning Vector2) ─────────────────────

        /// Component-wise addition.
        Vector2<DATA_TYPE> operator+(const Vector2<DATA_TYPE>& other) const {
            return Vector2<DATA_TYPE>(this->data[0] + other.data[0], this->data[1] + other.data[1]);
        }
        /// Component-wise subtraction.
        Vector2<DATA_TYPE> operator-(const Vector2<DATA_TYPE>& other) const {
            return Vector2<DATA_TYPE>(this->data[0] - other.data[0], this->data[1] - other.data[1]);
        }
        /// Negation.
        Vector2<DATA_TYPE> operator-() const {
            return Vector2<DATA_TYPE>(-this->data[0], -this->data[1]);
        }
        /// Scalar multiplication: `v * s`.
        Vector2<DATA_TYPE> operator*(DATA_TYPE scalar) const {
            return Vector2<DATA_TYPE>(this->data[0] * scalar, this->data[1] * scalar);
        }
        /// Scalar multiplication (commutative): `s * v`.
        friend Vector2<DATA_TYPE> operator*(DATA_TYPE scalar, const Vector2<DATA_TYPE>& v) {
            return v * scalar;
        }
        /// Scalar division.
        Vector2<DATA_TYPE> operator/(DATA_TYPE scalar) const {
            return Vector2<DATA_TYPE>(this->data[0] / scalar, this->data[1] / scalar);
        }

        // ── Methods ───────────────────────────────────────────────────────────

        /// Returns the dot product `this · rhs`.
        DATA_TYPE dot(const Vector2<DATA_TYPE>& rhs) const {
            return this->data[0] * rhs.data[0] + this->data[1] * rhs.data[1];
        }
        /// Static version: returns `lhs · rhs`.
        static DATA_TYPE dot(const Vector2<DATA_TYPE>& lhs, const Vector2<DATA_TYPE>& rhs) {
            return lhs.dot(rhs);
        }

        /// Returns the z-component of the 3D cross product (a scalar in 2D):
        /// `x * other.y - y * other.x`. Positive if @p other is counterclockwise from this.
        DATA_TYPE cross(const Vector2<DATA_TYPE>& other) const {
            return this->data[0] * other.data[1] - this->data[1] * other.data[0];
        }

        /// Returns the unsigned angle in radians between this vector and @p other. Range [0, π].
        DATA_TYPE angleTo(const Vector2<DATA_TYPE>& other) const {
            DATA_TYPE lenProduct = this->length() * other.length();
            if (isZero<DATA_TYPE>(lenProduct)) return DATA_TYPE(0);
            DATA_TYPE cosAngle = dot(other) / lenProduct;
            cosAngle = cosAngle < DATA_TYPE(-1) ? DATA_TYPE(-1) : (cosAngle > DATA_TYPE(1) ? DATA_TYPE(1) : cosAngle);
            return std::acos(cosAngle);
        }

        /// Returns the signed angle in radians from this vector to @p other.
        /// Positive = counterclockwise. Range (-π, π].
        DATA_TYPE angleToSigned(const Vector2<DATA_TYPE>& other) const {
            DATA_TYPE angle = angleTo(other);
            return cross(other) < DATA_TYPE(0) ? -angle : angle;
        }

        /// Reflects this vector in-place about @p normal (which must be normalized).
        /// Formula: `v = v - 2 * dot(v, n) * n`.
        void reflect(const Vector2<DATA_TYPE>& normal) {
            DATA_TYPE d = DATA_TYPE(2) * dot(normal);
            this->data[0] -= d * normal.data[0];
            this->data[1] -= d * normal.data[1];
        }

        /// Returns a copy of this vector reflected about @p normal.
        Vector2<DATA_TYPE> reflected(const Vector2<DATA_TYPE>& normal) const {
            auto result = *this;
            result.reflect(normal);
            return result;
        }

    };

}
