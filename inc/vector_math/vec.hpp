#pragma once

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <stdint.h>
#include <cstdlib>
#include <vector_math/common.hpp>

namespace systems::leal::vector_math
{

    /// Generic fixed-size vector of @p SIZE elements of type @p DATA_TYPE.
    ///
    /// Stored as a plain array `data[SIZE]` with natural alignment.
    /// All specializations (Vector2, Vector3, Vector4, …) derive from this class
    /// and inherit its full operator and method set.
    template <class DATA_TYPE, uint32_t SIZE>
    class alignas(DATA_TYPE) Vec
    {
    public:
        /// Element type alias.
        typedef DATA_TYPE DataType;

        /// Compile-time size constant accessible as `Vec::Size`.
        enum Params { Size = SIZE };

        /// Raw element storage. Laid out as a contiguous C array.
        DATA_TYPE data[SIZE];

        // ── Constructors ────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Vec();

        /// Constructs all elements from a single @p value (broadcast/splat).
        Vec(DATA_TYPE value);

        // ── Comparison ──────────────────────────────────────────────────────

        /// Returns true if every element compares equal.
        bool operator==(const Vec<DATA_TYPE, SIZE> &other) const;

        /// Returns true if any element differs.
        bool operator!=(const Vec<DATA_TYPE, SIZE> &other) const;

        // ── Arithmetic operators ─────────────────────────────────────────────

        /// Component-wise addition.
        Vec<DATA_TYPE, SIZE> operator+(const Vec<DATA_TYPE, SIZE> &other) const;

        /// Component-wise subtraction.
        Vec<DATA_TYPE, SIZE> operator-(const Vec<DATA_TYPE, SIZE> &other) const;

        /// Negation: returns `-(*this)` component-wise.
        Vec<DATA_TYPE, SIZE> operator-() const;

        /// Scalar multiplication: `v * s`.
        Vec<DATA_TYPE, SIZE> operator*(DATA_TYPE scalar) const;

        /// Scalar multiplication (commutative): `s * v`.
        friend Vec<DATA_TYPE, SIZE> operator*(DATA_TYPE scalar, const Vec<DATA_TYPE, SIZE> &lhs)
        {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a = 0; a < SIZE; a++)
                toReturn.data[a] = lhs.data[a] * scalar;
            return toReturn;
        }

        /// Scalar division: `v / s`.
        Vec<DATA_TYPE, SIZE> operator/(DATA_TYPE scalar) const;

        /// Component-wise multiplication: `v * u` (Hadamard product).
        Vec<DATA_TYPE, SIZE> operator*(const Vec<DATA_TYPE, SIZE>& other) const;

        // ── Length / normalization ───────────────────────────────────────────

        /// Returns the squared Euclidean length (avoids a square-root call).
        DATA_TYPE lengthSquared() const;

        /// Returns the Euclidean length: `sqrt(lengthSquared())`.
        DATA_TYPE length() const;

        /// Normalizes this vector in-place. No-op if the vector is zero.
        void normalize();

        /// Returns a normalized copy without modifying this vector.
        Vec<DATA_TYPE, SIZE> normalized() const;

        // ── Dot product ──────────────────────────────────────────────────────

        /// Returns the dot product of this vector and @p rhs.
        DATA_TYPE dot(const Vec<DATA_TYPE, SIZE>& rhs) const;

        /// Static version: returns `lhs · rhs`.
        static DATA_TYPE dot(const Vec<DATA_TYPE, SIZE>& lhs, const Vec<DATA_TYPE, SIZE>& rhs);

        // ── Distance ────────────────────────────────────────────────────────

        /// Returns the squared Euclidean distance to @p other (cheaper than distanceTo).
        DATA_TYPE distanceToSquared(const Vec<DATA_TYPE, SIZE>& other) const;

        /// Returns the Euclidean distance to @p other.
        DATA_TYPE distanceTo(const Vec<DATA_TYPE, SIZE>& other) const;

        // ── Mutation helpers ─────────────────────────────────────────────────

        /// Sets all elements to zero.
        void setZero();

        /// Copies all elements from @p other into this vector.
        void setFrom(const Vec<DATA_TYPE, SIZE>& other);

        /// Replaces each element with its absolute value.
        void absolute();

        /// Replaces each element with `floor(element)`.
        void floor();

        /// Replaces each element with `ceil(element)`.
        void ceil();

        /// Replaces each element with `round(element)` (round-half-away-from-zero).
        void round();

        /// Clamps each element to `[minVal, maxVal]`.
        void clamp(DATA_TYPE minVal, DATA_TYPE maxVal);

        /// Clamps each element to `[minVec[i], maxVec[i]]` independently.
        void clamp(const Vec<DATA_TYPE, SIZE>& minVec, const Vec<DATA_TYPE, SIZE>& maxVec);

        // ── Static factories ─────────────────────────────────────────────────

        /// Returns a new vector whose elements are `min(a[i], b[i])`.
        static Vec<DATA_TYPE, SIZE> min(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b);

        /// Returns a new vector whose elements are `max(a[i], b[i])`.
        static Vec<DATA_TYPE, SIZE> max(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b);

        /// Linear interpolation: returns `a + t * (b - a)`.
        /// @param t  Blend factor; 0 returns @p a, 1 returns @p b.
        static Vec<DATA_TYPE, SIZE> lerp(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b, DATA_TYPE t);

    };

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE>::Vec()
    {
        for (uint32_t a = 0; a < this->Size; a++)
        {
            data[a] = 0;
        }
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE>::Vec(DATA_TYPE value)
    {
        for (uint32_t a = 0; a < this->Size; a++)
        {
            data[a] = value;
        }
    }

    template <class DATA_TYPE, uint32_t SIZE>
    bool Vec<DATA_TYPE, SIZE>::operator==(const Vec<DATA_TYPE, SIZE> &other) const
    {
        for (uint32_t a = 0; a < this->Size; a++)
        {
            if (data[a] != other.data[a])
                return false;
        }
        return true;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    bool Vec<DATA_TYPE, SIZE>::operator!=(const Vec<DATA_TYPE, SIZE> &other) const
    {
        return !operator==(other);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator+(const Vec<DATA_TYPE, SIZE> &other) const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (uint32_t a = 0; a < this->Size; a++)
        {
            toReturn.data[a] = data[a] + other.data[a];
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator-(const Vec<DATA_TYPE, SIZE> &other) const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (uint32_t a = 0; a < this->Size; a++)
        {
            toReturn.data[a] = data[a] - other.data[a];
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator*(DATA_TYPE scalar) const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (uint32_t a = 0; a < this->Size; a++)
        {
            toReturn.data[a] = data[a] * scalar;
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator/(DATA_TYPE scalar) const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (uint32_t a = 0; a < this->Size; a++)
        {
            toReturn.data[a] = data[a] / scalar;
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator-() const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (int a = 0; a < SIZE; a++)
        {
            toReturn.data[a] = -this->data[a];
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::lengthSquared() const
    {
        DATA_TYPE toReturn = 0.0;
        for (int a = 0; a < SIZE; a++)
        {
            toReturn += this->data[a] * this->data[a];
        }
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::length() const
    {
        return std::sqrt(lengthSquared());
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::normalize()
    {
        DATA_TYPE lenSqr = lengthSquared();

        if (isZero<DATA_TYPE>(lenSqr))
            return;

        DATA_TYPE invLength = DATA_TYPE(1) / std::sqrt(lenSqr);

        for (int a = 0; a < SIZE; a++)
        {
            this->data[a] *= invLength;
        }
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::normalized() const
    {
        auto toReturn = *this;
        toReturn.normalize();
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::dot(const Vec<DATA_TYPE, SIZE>& rhs) const
    {
        DATA_TYPE result = 0;
        for (uint32_t a = 0; a < SIZE; a++)
            result += data[a] * rhs.data[a];
        return result;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::dot(const Vec<DATA_TYPE, SIZE>& lhs, const Vec<DATA_TYPE, SIZE>& rhs)
    {
        return lhs.dot(rhs);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::operator*(const Vec<DATA_TYPE, SIZE>& other) const
    {
        Vec<DATA_TYPE, SIZE> toReturn;
        for (uint32_t a = 0; a < SIZE; a++)
            toReturn.data[a] = data[a] * other.data[a];
        return toReturn;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::distanceToSquared(const Vec<DATA_TYPE, SIZE>& other) const
    {
        DATA_TYPE sum = 0;
        for (uint32_t a = 0; a < SIZE; a++) {
            DATA_TYPE d = data[a] - other.data[a];
            sum += d * d;
        }
        return sum;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    DATA_TYPE Vec<DATA_TYPE, SIZE>::distanceTo(const Vec<DATA_TYPE, SIZE>& other) const
    {
        return std::sqrt(distanceToSquared(other));
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::setZero()
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = 0;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::setFrom(const Vec<DATA_TYPE, SIZE>& other)
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = other.data[a];
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::absolute()
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = std::abs(data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::floor()
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = std::floor(data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::ceil()
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = std::ceil(data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::round()
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = std::round(data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::clamp(DATA_TYPE minVal, DATA_TYPE maxVal)
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = data[a] < minVal ? minVal : (data[a] > maxVal ? maxVal : data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    void Vec<DATA_TYPE, SIZE>::clamp(const Vec<DATA_TYPE, SIZE>& minVec, const Vec<DATA_TYPE, SIZE>& maxVec)
    {
        for (uint32_t a = 0; a < SIZE; a++)
            data[a] = data[a] < minVec.data[a] ? minVec.data[a] : (data[a] > maxVec.data[a] ? maxVec.data[a] : data[a]);
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::min(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b)
    {
        Vec<DATA_TYPE, SIZE> result;
        for (uint32_t i = 0; i < SIZE; i++)
            result.data[i] = a.data[i] < b.data[i] ? a.data[i] : b.data[i];
        return result;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::max(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b)
    {
        Vec<DATA_TYPE, SIZE> result;
        for (uint32_t i = 0; i < SIZE; i++)
            result.data[i] = a.data[i] > b.data[i] ? a.data[i] : b.data[i];
        return result;
    }

    template <class DATA_TYPE, uint32_t SIZE>
    Vec<DATA_TYPE, SIZE> Vec<DATA_TYPE, SIZE>::lerp(const Vec<DATA_TYPE, SIZE>& a, const Vec<DATA_TYPE, SIZE>& b, DATA_TYPE t)
    {
        Vec<DATA_TYPE, SIZE> result;
        for (uint32_t i = 0; i < SIZE; i++)
            result.data[i] = a.data[i] + t * (b.data[i] - a.data[i]);
        return result;
    }

}
