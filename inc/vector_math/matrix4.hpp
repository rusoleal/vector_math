#pragma once

#include <cstring>
#include <vector_math/mat.hpp>
#include <vector_math/vector3.hpp>
#include <vector_math/vector4.hpp>
#include <vector_math/quaternion.hpp>

namespace systems::leal::vector_math
{

    /// 4×4 matrix of type @p DATA_TYPE stored in row-major order.
    ///
    /// Element at row @p r, column @p c is `data[r*4 + c]`.
    /// Uses the column-vector convention: `result = M * v` (vector on the right).
    /// Translation occupies the last column: `data[3]`=tx, `data[7]`=ty, `data[11]`=tz.
    ///
    /// Inherits Mat<DATA_TYPE,4,4> (generic multiply, transpose) and all Vec utilities.
    template <class DATA_TYPE>
    class Matrix4 : public Mat<DATA_TYPE, 4, 4>
    {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Matrix4();

        /// Constructs all elements from a single @p value.
        Matrix4(DATA_TYPE value);

        /// Constructs from a 16-element row-major C array.
        Matrix4(DATA_TYPE buffer[16]);

        // ── Operators ─────────────────────────────────────────────────────────

        /// Matrix–matrix multiplication.
        Matrix4<DATA_TYPE> operator*(const Matrix4<DATA_TYPE> &rhs) const;

        /// Scalar multiplication: scales every element.
        Matrix4<DATA_TYPE> operator*(DATA_TYPE scalar) const;

        /// Matrix–vector multiplication (column vector on right, returns Vector4).
        Vector4<DATA_TYPE> operator*(const Vector4<DATA_TYPE> &vector) const;

        /// Component-wise addition.
        Matrix4<DATA_TYPE> operator+(const Matrix4<DATA_TYPE>& rhs) const;

        /// Component-wise subtraction.
        Matrix4<DATA_TYPE> operator-(const Matrix4<DATA_TYPE>& rhs) const;

        /// Negation: returns `-(*this)` element-wise.
        Matrix4<DATA_TYPE> operator-() const;

        // ── Static factories ──────────────────────────────────────────────────

        /// Returns a 4×4 identity matrix.
        static Matrix4<DATA_TYPE> identity();

        /// Builds a right-handed view matrix looking from @p eye toward @p target with @p up as the up direction.
        static Matrix4<DATA_TYPE> lookAt(const Vector3<DATA_TYPE> &eye, const Vector3<DATA_TYPE> &target, const Vector3<DATA_TYPE> &up);

        /// Builds a perspective projection matrix.
        /// @param fov          Vertical field of view in radians.
        /// @param aspectRatio  Width / height.
        /// @param znear        Distance to the near plane (positive).
        /// @param zfar         Distance to the far plane (positive, > znear).
        static Matrix4<DATA_TYPE> perspective(DATA_TYPE fov, DATA_TYPE aspectRatio, DATA_TYPE znear, DATA_TYPE zfar);

        /// Builds a symmetric orthographic projection matrix centered on the origin.
        /// @param width   Total width of the view volume.
        /// @param height  Total height of the view volume.
        static Matrix4<DATA_TYPE> ortho(DATA_TYPE width, DATA_TYPE height, DATA_TYPE znear, DATA_TYPE zfar);

        /// Builds an off-center orthographic projection matrix.
        static Matrix4<DATA_TYPE> orthoOffCenter(DATA_TYPE left, DATA_TYPE top, DATA_TYPE right, DATA_TYPE bottom, DATA_TYPE znear, DATA_TYPE zfar);

        /// Builds a translation matrix from a Vector3 offset.
        static Matrix4<DATA_TYPE> translate(const Vector3<DATA_TYPE> translation);

        /// Builds a non-uniform scale matrix.
        static Matrix4<DATA_TYPE> scale(const Vector3<DATA_TYPE> scale);

        /// Builds a rotation matrix from a unit quaternion.
        static Matrix4<DATA_TYPE> rotate(const Quaternion<DATA_TYPE> &rotation);

        /// Builds a rotation matrix around the X axis by @p angle radians.
        static Matrix4<DATA_TYPE> rotateX(DATA_TYPE angle);

        /// Builds a rotation matrix around the Y axis by @p angle radians.
        static Matrix4<DATA_TYPE> rotateY(DATA_TYPE angle);

        /// Builds a rotation matrix around the Z axis by @p angle radians.
        static Matrix4<DATA_TYPE> rotateZ(DATA_TYPE angle);

        /// Builds a combined TRS (Translate–Rotate–Scale) matrix.
        /// Equivalent to `translate(t) * rotate(r) * scale(s)`.
        static Matrix4<DATA_TYPE> compose(const Vector3<DATA_TYPE>& translation, const Quaternion<DATA_TYPE>& rotation, const Vector3<DATA_TYPE>& scaleVec);

        // ── Methods ───────────────────────────────────────────────────────────

        /// Returns the sum of the diagonal elements (M[0][0] + M[1][1] + M[2][2] + M[3][3]).
        DATA_TYPE trace() const;

        /// Returns the determinant of this matrix.
        DATA_TYPE determinant() const;

        /// Inverts this matrix in-place.
        /// @return true on success; false if the matrix is singular (no-op in that case).
        bool invert();

        /// Returns an inverted copy. Returns an unchanged copy if the matrix is singular.
        Matrix4<DATA_TYPE> inverted() const;

        /// Returns row @p row (0–3) as a Vector4.
        Vector4<DATA_TYPE> getRow(int row) const;

        /// Writes @p v into row @p row (0–3).
        void setRow(int row, const Vector4<DATA_TYPE>& v);

        /// Returns column @p col (0–3) as a Vector4.
        Vector4<DATA_TYPE> getColumn(int col) const;

        /// Writes @p v into column @p col (0–3).
        void setColumn(int col, const Vector4<DATA_TYPE>& v);

        /// Extracts the translation from the last column (data[3], data[7], data[11]).
        Vector3<DATA_TYPE> getTranslation() const;

        /// Writes @p t into the last column (data[3], data[7], data[11]).
        void setTranslation(const Vector3<DATA_TYPE>& t);

        /// Extracts per-axis scale factors as the lengths of the three basis columns.
        /// Valid only for pure TRS matrices (no shear).
        Vector3<DATA_TYPE> getScale() const;

        /// Transforms a 3D point by this matrix: sets w=1, multiplies, returns xyz.
        /// Applies the full TRS transform including translation.
        Vector3<DATA_TYPE> transform3(const Vector3<DATA_TYPE>& v) const;

        /// Transforms a 3D direction by this matrix: sets w=0, multiplies, returns xyz.
        /// Applies only rotation and scale — translation is ignored.
        Vector3<DATA_TYPE> rotate3(const Vector3<DATA_TYPE>& v) const;

    };

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE>::Matrix4() : Mat<DATA_TYPE, 4, 4>() {}

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE>::Matrix4(DATA_TYPE value) : Mat<DATA_TYPE, 4, 4>(value) {}

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE>::Matrix4(DATA_TYPE buffer[16])
    {
        memcpy(this->data, buffer, sizeof(DATA_TYPE) * 16);
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::operator*(const Matrix4<DATA_TYPE> &rhs) const {
        auto toReturn = ((Mat<DATA_TYPE,4,4> *)this)->operator*(rhs);
        return *(Matrix4<DATA_TYPE> *)&toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::operator*(DATA_TYPE scalar) const {
        auto toReturn = ((Mat<DATA_TYPE,4,4> *)this)->operator*(scalar);
        return *(Matrix4<DATA_TYPE> *)&toReturn;
    }

    template <class DATA_TYPE>
    Vector4<DATA_TYPE> Matrix4<DATA_TYPE>::operator*(const Vector4<DATA_TYPE> &vector) const {
        return Vector4(
            this->data[ 0] * vector.data[0] + this->data[ 1] * vector.data[1] + this->data[ 2] * vector.data[2] + this->data[ 3] * vector.data[3],
            this->data[ 4] * vector.data[0] + this->data[ 5] * vector.data[1] + this->data[ 6] * vector.data[2] + this->data[ 7] * vector.data[3],
            this->data[ 8] * vector.data[0] + this->data[ 9] * vector.data[1] + this->data[10] * vector.data[2] + this->data[11] * vector.data[3],
            this->data[12] * vector.data[0] + this->data[13] * vector.data[1] + this->data[14] * vector.data[2] + this->data[15] * vector.data[3]
        );
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::identity()
    {
        DATA_TYPE data[] = {
            1.0,0.0,0.0,0.0,
            0.0,1.0,0.0,0.0,
            0.0,0.0,1.0,0.0,
            0.0,0.0,0.0,1.0,
        };
        return Matrix4<DATA_TYPE>(data);
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::lookAt(const Vector3<DATA_TYPE> &eye, const Vector3<DATA_TYPE> &target, const Vector3<DATA_TYPE> &up)
    {
        auto result = Matrix4<DATA_TYPE>::identity();

        auto zAxis = target - eye;
        zAxis.normalize();

        auto xAxis = Vector3<DATA_TYPE>::cross(up, zAxis);
        xAxis.normalize();

        auto yAxis = Vector3<DATA_TYPE>::cross(zAxis, xAxis);

        result.data[0] = xAxis.data[0];
        result.data[1] = xAxis.data[1];
        result.data[2] = xAxis.data[2];

        result.data[4] = yAxis.data[0];
        result.data[5] = yAxis.data[1];
        result.data[6] = yAxis.data[2];

        result.data[8] = zAxis.data[0];
        result.data[9] = zAxis.data[1];
        result.data[10] = zAxis.data[2];

        result.data[3] = -Vector3<DATA_TYPE>::dot(xAxis, eye);
        result.data[7] = -Vector3<DATA_TYPE>::dot(yAxis, eye);
        result.data[11] = -Vector3<DATA_TYPE>::dot(zAxis, eye);

        return result;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::perspective(DATA_TYPE fov, DATA_TYPE aspectRatio, DATA_TYPE znear, DATA_TYPE zfar)
    {
        auto toReturn = Matrix4<DATA_TYPE>(DATA_TYPE(0));
        auto yScale = DATA_TYPE(1.0) / tan(fov * DATA_TYPE(0.5));
        auto q = zfar / (zfar - znear);

        toReturn.data[0] = yScale / aspectRatio;
        toReturn.data[5] = yScale;
        toReturn.data[10] = q;
        toReturn.data[14] = DATA_TYPE(1.0);
        toReturn.data[11] = -q * znear;

        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::ortho(DATA_TYPE width, DATA_TYPE height, DATA_TYPE znear, DATA_TYPE zfar)
    {
        DATA_TYPE halfWidth = width * DATA_TYPE(0.5);
        DATA_TYPE halfHeight = height * DATA_TYPE(0.5);

        return Matrix4<DATA_TYPE>::orthoOffCenter(-halfWidth, halfHeight, halfWidth, -halfHeight, znear, zfar);
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::orthoOffCenter(DATA_TYPE left, DATA_TYPE top, DATA_TYPE right, DATA_TYPE bottom, DATA_TYPE znear, DATA_TYPE zfar)
    {
        auto toReturn = Matrix4<DATA_TYPE>::identity();

        DATA_TYPE zRange = DATA_TYPE(1.0) / (zfar - znear);

        toReturn.data[0] = DATA_TYPE(2.0) / (right - left);
        toReturn.data[5] = DATA_TYPE(2.0) / (top - bottom);
        toReturn.data[10] = zRange;
        toReturn.data[3] = (left + right) / (left - right);
        toReturn.data[7] = (top + bottom) / (bottom - top);
        toReturn.data[11] = -znear * zRange;

        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::translate(const Vector3<DATA_TYPE> translation)
    {
        auto toReturn = Matrix4<DATA_TYPE>::identity();
        toReturn.data[3] = translation.data[0];
        toReturn.data[7] = translation.data[1];
        toReturn.data[11] = translation.data[2];
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::scale(const Vector3<DATA_TYPE> scale)
    {
        auto toReturn = Matrix4<DATA_TYPE>(DATA_TYPE(0));
        toReturn.data[0] = scale.data[0];
        toReturn.data[5] = scale.data[1];
        toReturn.data[10] = scale.data[2];
        toReturn.data[15] = 1.0;
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::rotate(const Quaternion<DATA_TYPE> &rotation)
    {
        DATA_TYPE xx = rotation.data[0] * rotation.data[0];
        DATA_TYPE yy = rotation.data[1] * rotation.data[1];
        DATA_TYPE zz = rotation.data[2] * rotation.data[2];
        DATA_TYPE xy = rotation.data[0] * rotation.data[1];
        DATA_TYPE zw = rotation.data[2] * rotation.data[3];
        DATA_TYPE zx = rotation.data[2] * rotation.data[0];
        DATA_TYPE yw = rotation.data[1] * rotation.data[3];
        DATA_TYPE yz = rotation.data[1] * rotation.data[2];
        DATA_TYPE xw = rotation.data[0] * rotation.data[3];

        auto toReturn = Matrix4<DATA_TYPE>::identity();
        toReturn.data[0] = DATA_TYPE(1.0) - (DATA_TYPE(2.0) * (yy + zz));
        toReturn.data[4] = DATA_TYPE(2.0) * (xy + zw);
        toReturn.data[8] = DATA_TYPE(2.0) * (zx - yw);
        toReturn.data[1] = DATA_TYPE(2.0) * (xy - zw);
        toReturn.data[5] = DATA_TYPE(1.0) - (DATA_TYPE(2.0) * (zz + xx));
        toReturn.data[9] = DATA_TYPE(2.0) * (yz + xw);
        toReturn.data[2] = DATA_TYPE(2.0) * (zx + yw);
        toReturn.data[6] = DATA_TYPE(2.0) * (yz - xw);
        toReturn.data[10] = DATA_TYPE(1.0) - (DATA_TYPE(2.0) * (yy + xx));
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::rotateX(DATA_TYPE angle)
    {
        DATA_TYPE cosA = std::cos(angle);
        DATA_TYPE sinA = std::sin(angle);
        auto toReturn = Matrix4<DATA_TYPE>::identity();
        toReturn.data[5] = cosA;
        toReturn.data[6] = sinA;
        toReturn.data[9] = -sinA;
        toReturn.data[10] = cosA;
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::rotateY(DATA_TYPE angle)
    {
        DATA_TYPE cosA = std::cos(angle);
        DATA_TYPE sinA = std::sin(angle);
        auto toReturn = Matrix4<DATA_TYPE>::identity();
        toReturn.data[0] = cosA;
        toReturn.data[2] = -sinA;
        toReturn.data[8] = sinA;
        toReturn.data[10] = cosA;
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::rotateZ(DATA_TYPE angle)
    {
        DATA_TYPE cosA = std::cos(angle);
        DATA_TYPE sinA = std::sin(angle);
        auto toReturn = Matrix4<DATA_TYPE>::identity();
        toReturn.data[0] = cosA;
        toReturn.data[1] = sinA;
        toReturn.data[4] = -sinA;
        toReturn.data[5] = cosA;
        return toReturn;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::compose(const Vector3<DATA_TYPE>& translation, const Quaternion<DATA_TYPE>& rotation, const Vector3<DATA_TYPE>& scaleVec)
    {
        auto result = Matrix4<DATA_TYPE>::rotate(rotation);
        // Scale columns 0, 1, 2 by scaleVec components
        result.data[0] *= scaleVec.data[0]; result.data[1] *= scaleVec.data[1]; result.data[2]  *= scaleVec.data[2];
        result.data[4] *= scaleVec.data[0]; result.data[5] *= scaleVec.data[1]; result.data[6]  *= scaleVec.data[2];
        result.data[8] *= scaleVec.data[0]; result.data[9] *= scaleVec.data[1]; result.data[10] *= scaleVec.data[2];
        result.data[3]  = translation.data[0];
        result.data[7]  = translation.data[1];
        result.data[11] = translation.data[2];
        return result;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::operator+(const Matrix4<DATA_TYPE>& rhs) const
    {
        Matrix4<DATA_TYPE> result;
        for (int i = 0; i < 16; i++)
            result.data[i] = this->data[i] + rhs.data[i];
        return result;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::operator-(const Matrix4<DATA_TYPE>& rhs) const
    {
        Matrix4<DATA_TYPE> result;
        for (int i = 0; i < 16; i++)
            result.data[i] = this->data[i] - rhs.data[i];
        return result;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::operator-() const
    {
        Matrix4<DATA_TYPE> result;
        for (int i = 0; i < 16; i++)
            result.data[i] = -this->data[i];
        return result;
    }

    template <class DATA_TYPE>
    DATA_TYPE Matrix4<DATA_TYPE>::trace() const
    {
        return this->data[0] + this->data[5] + this->data[10] + this->data[15];
    }

    template <class DATA_TYPE>
    DATA_TYPE Matrix4<DATA_TYPE>::determinant() const
    {
        const DATA_TYPE* m = this->data;
        DATA_TYPE b00 = m[10]*m[15] - m[11]*m[14];
        DATA_TYPE b01 = m[9]*m[15]  - m[11]*m[13];
        DATA_TYPE b02 = m[9]*m[14]  - m[10]*m[13];
        DATA_TYPE b03 = m[8]*m[15]  - m[11]*m[12];
        DATA_TYPE b04 = m[8]*m[14]  - m[10]*m[12];
        DATA_TYPE b05 = m[8]*m[13]  - m[9]*m[12];
        return m[0]*(m[5]*b00 - m[6]*b01 + m[7]*b02)
             - m[1]*(m[4]*b00 - m[6]*b03 + m[7]*b04)
             + m[2]*(m[4]*b01 - m[5]*b03 + m[7]*b05)
             - m[3]*(m[4]*b02 - m[5]*b04 + m[6]*b05);
    }

    template <class DATA_TYPE>
    bool Matrix4<DATA_TYPE>::invert()
    {
        const DATA_TYPE* m = this->data;
        DATA_TYPE b00 = m[0]*m[5]   - m[1]*m[4];
        DATA_TYPE b01 = m[0]*m[6]   - m[2]*m[4];
        DATA_TYPE b02 = m[0]*m[7]   - m[3]*m[4];
        DATA_TYPE b03 = m[1]*m[6]   - m[2]*m[5];
        DATA_TYPE b04 = m[1]*m[7]   - m[3]*m[5];
        DATA_TYPE b05 = m[2]*m[7]   - m[3]*m[6];
        DATA_TYPE b06 = m[8]*m[13]  - m[9]*m[12];
        DATA_TYPE b07 = m[8]*m[14]  - m[10]*m[12];
        DATA_TYPE b08 = m[8]*m[15]  - m[11]*m[12];
        DATA_TYPE b09 = m[9]*m[14]  - m[10]*m[13];
        DATA_TYPE b10 = m[9]*m[15]  - m[11]*m[13];
        DATA_TYPE b11 = m[10]*m[15] - m[11]*m[14];

        DATA_TYPE det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
        if (isZero<DATA_TYPE>(det)) return false;
        DATA_TYPE inv = DATA_TYPE(1) / det;

        DATA_TYPE n[16];
        for (int i = 0; i < 16; i++) n[i] = m[i];

        this->data[0]  = ( n[5]*b11  - n[6]*b10  + n[7]*b09)  * inv;
        this->data[1]  = (-n[1]*b11  + n[2]*b10  - n[3]*b09)  * inv;
        this->data[2]  = ( n[13]*b05 - n[14]*b04 + n[15]*b03) * inv;
        this->data[3]  = (-n[9]*b05  + n[10]*b04 - n[11]*b03) * inv;
        this->data[4]  = (-n[4]*b11  + n[6]*b08  - n[7]*b07)  * inv;
        this->data[5]  = ( n[0]*b11  - n[2]*b08  + n[3]*b07)  * inv;
        this->data[6]  = (-n[12]*b05 + n[14]*b02 - n[15]*b01) * inv;
        this->data[7]  = ( n[8]*b05  - n[10]*b02 + n[11]*b01) * inv;
        this->data[8]  = ( n[4]*b10  - n[5]*b08  + n[7]*b06)  * inv;
        this->data[9]  = (-n[0]*b10  + n[1]*b08  - n[3]*b06)  * inv;
        this->data[10] = ( n[12]*b04 - n[13]*b02 + n[15]*b00) * inv;
        this->data[11] = (-n[8]*b04  + n[9]*b02  - n[11]*b00) * inv;
        this->data[12] = (-n[4]*b09  + n[5]*b07  - n[6]*b06)  * inv;
        this->data[13] = ( n[0]*b09  - n[1]*b07  + n[2]*b06)  * inv;
        this->data[14] = (-n[12]*b03 + n[13]*b01 - n[14]*b00) * inv;
        this->data[15] = ( n[8]*b03  - n[9]*b01  + n[10]*b00) * inv;
        return true;
    }

    template <class DATA_TYPE>
    Matrix4<DATA_TYPE> Matrix4<DATA_TYPE>::inverted() const
    {
        auto result = *this;
        result.invert();
        return result;
    }

    template <class DATA_TYPE>
    Vector4<DATA_TYPE> Matrix4<DATA_TYPE>::getRow(int row) const
    {
        return Vector4<DATA_TYPE>(
            this->data[row*4], this->data[row*4+1],
            this->data[row*4+2], this->data[row*4+3]
        );
    }

    template <class DATA_TYPE>
    void Matrix4<DATA_TYPE>::setRow(int row, const Vector4<DATA_TYPE>& v)
    {
        this->data[row*4]   = v.data[0];
        this->data[row*4+1] = v.data[1];
        this->data[row*4+2] = v.data[2];
        this->data[row*4+3] = v.data[3];
    }

    template <class DATA_TYPE>
    Vector4<DATA_TYPE> Matrix4<DATA_TYPE>::getColumn(int col) const
    {
        return Vector4<DATA_TYPE>(
            this->data[col], this->data[4+col],
            this->data[8+col], this->data[12+col]
        );
    }

    template <class DATA_TYPE>
    void Matrix4<DATA_TYPE>::setColumn(int col, const Vector4<DATA_TYPE>& v)
    {
        this->data[col]    = v.data[0];
        this->data[4+col]  = v.data[1];
        this->data[8+col]  = v.data[2];
        this->data[12+col] = v.data[3];
    }

    template <class DATA_TYPE>
    Vector3<DATA_TYPE> Matrix4<DATA_TYPE>::getTranslation() const
    {
        return Vector3<DATA_TYPE>(this->data[3], this->data[7], this->data[11]);
    }

    template <class DATA_TYPE>
    void Matrix4<DATA_TYPE>::setTranslation(const Vector3<DATA_TYPE>& t)
    {
        this->data[3]  = t.data[0];
        this->data[7]  = t.data[1];
        this->data[11] = t.data[2];
    }

    template <class DATA_TYPE>
    Vector3<DATA_TYPE> Matrix4<DATA_TYPE>::getScale() const
    {
        return Vector3<DATA_TYPE>(
            std::sqrt(this->data[0]*this->data[0] + this->data[4]*this->data[4] + this->data[8]*this->data[8]),
            std::sqrt(this->data[1]*this->data[1] + this->data[5]*this->data[5] + this->data[9]*this->data[9]),
            std::sqrt(this->data[2]*this->data[2] + this->data[6]*this->data[6] + this->data[10]*this->data[10])
        );
    }

    template <class DATA_TYPE>
    Vector3<DATA_TYPE> Matrix4<DATA_TYPE>::transform3(const Vector3<DATA_TYPE>& v) const
    {
        auto r = (*this) * Vector4<DATA_TYPE>(v.data[0], v.data[1], v.data[2], DATA_TYPE(1));
        return Vector3<DATA_TYPE>(r.data[0], r.data[1], r.data[2]);
    }

    template <class DATA_TYPE>
    Vector3<DATA_TYPE> Matrix4<DATA_TYPE>::rotate3(const Vector3<DATA_TYPE>& v) const
    {
        auto r = (*this) * Vector4<DATA_TYPE>(v.data[0], v.data[1], v.data[2], DATA_TYPE(0));
        return Vector3<DATA_TYPE>(r.data[0], r.data[1], r.data[2]);
    }

}
