#pragma once

#include<vector_math/vec.hpp>

namespace systems::leal::vector_math {

    /// 4D vector of type @p DATA_TYPE.
    ///
    /// Inherits all Vec<DATA_TYPE,4> operations.
    /// Components: `data[0]` = x, `data[1]` = y, `data[2]` = z, `data[3]` = w.
    template<class DATA_TYPE>
    class Vector4:public Vec<DATA_TYPE,4> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Vector4() = default;

        /// Constructs from explicit x, y, z, w values.
        Vector4(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z, DATA_TYPE w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

        // ── Named accessors ───────────────────────────────────────────────────

        DATA_TYPE& x() { return this->data[0]; }              ///< Reference to the x component.
        const DATA_TYPE& x() const { return this->data[0]; } ///< Const reference to x.
        DATA_TYPE& y() { return this->data[1]; }              ///< Reference to the y component.
        const DATA_TYPE& y() const { return this->data[1]; } ///< Const reference to y.
        DATA_TYPE& z() { return this->data[2]; }              ///< Reference to the z component.
        const DATA_TYPE& z() const { return this->data[2]; } ///< Const reference to z.
        DATA_TYPE& w() { return this->data[3]; }              ///< Reference to the w component.
        const DATA_TYPE& w() const { return this->data[3]; } ///< Const reference to w.

        // ── Operators (typed overrides returning Vector4) ─────────────────────

        /// Component-wise addition.
        Vector4<DATA_TYPE> operator+(const Vector4<DATA_TYPE>& other) const {
            return Vector4<DATA_TYPE>(
                this->data[0] + other.data[0], this->data[1] + other.data[1],
                this->data[2] + other.data[2], this->data[3] + other.data[3]
            );
        }
        /// Component-wise subtraction.
        Vector4<DATA_TYPE> operator-(const Vector4<DATA_TYPE>& other) const {
            return Vector4<DATA_TYPE>(
                this->data[0] - other.data[0], this->data[1] - other.data[1],
                this->data[2] - other.data[2], this->data[3] - other.data[3]
            );
        }
        /// Negation.
        Vector4<DATA_TYPE> operator-() const {
            return Vector4<DATA_TYPE>(-this->data[0], -this->data[1], -this->data[2], -this->data[3]);
        }
        /// Scalar multiplication: `v * s`.
        Vector4<DATA_TYPE> operator*(DATA_TYPE scalar) const {
            return Vector4<DATA_TYPE>(
                this->data[0] * scalar, this->data[1] * scalar,
                this->data[2] * scalar, this->data[3] * scalar
            );
        }
        /// Scalar multiplication (commutative): `s * v`.
        friend Vector4<DATA_TYPE> operator*(DATA_TYPE scalar, const Vector4<DATA_TYPE>& v) {
            return v * scalar;
        }
        /// Scalar division.
        Vector4<DATA_TYPE> operator/(DATA_TYPE scalar) const {
            return Vector4<DATA_TYPE>(
                this->data[0] / scalar, this->data[1] / scalar,
                this->data[2] / scalar, this->data[3] / scalar
            );
        }

        // ── Methods ───────────────────────────────────────────────────────────

        /// Returns the dot product `this · rhs`.
        DATA_TYPE dot(const Vector4<DATA_TYPE>& rhs) const {
            return this->data[0]*rhs.data[0] + this->data[1]*rhs.data[1]
                 + this->data[2]*rhs.data[2] + this->data[3]*rhs.data[3];
        }
        /// Static version: returns `lhs · rhs`.
        static DATA_TYPE dot(const Vector4<DATA_TYPE>& lhs, const Vector4<DATA_TYPE>& rhs) {
            return lhs.dot(rhs);
        }

        /// Sets all four components at once.
        void setValues(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z, DATA_TYPE w) {
            this->data[0] = x; this->data[1] = y; this->data[2] = z; this->data[3] = w;
        }

    };

}
