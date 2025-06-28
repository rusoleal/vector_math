#pragma once

#include <cstring>
#include <vector_math/mat.hpp>
#include <vector_math/vector3.hpp>
#include <vector_math/vector4.hpp>
#include <vector_math/quaternion.hpp>

namespace systems::leal::vector_math
{

    template <class DATA_TYPE>
    class Matrix4 : public Mat<DATA_TYPE, 4, 4>
    {
    public:
        Matrix4() : Mat<DATA_TYPE, 4, 4>() {}
        Matrix4(DATA_TYPE value) : Mat<DATA_TYPE, 4, 4>(value) {}
        Matrix4(DATA_TYPE buffer[16])
        {
            memcpy(this->data, buffer, sizeof(DATA_TYPE) * 16);
        }

        Matrix4<DATA_TYPE> operator*(const Matrix4<DATA_TYPE> rhs) {
            auto toReturn = ((Mat<DATA_TYPE,4,4> *)this)->operator*(rhs);
            return *(Matrix4<DATA_TYPE> *)&toReturn;
        }

        Matrix4<DATA_TYPE> operator*(DATA_TYPE scalar) {
            auto toReturn = ((Mat<DATA_TYPE,4,4> *)this)->operator*(scalar);
            return *(Matrix4<DATA_TYPE> *)&toReturn;
        }

        Vector4<DATA_TYPE> operator*(const Vector4<DATA_TYPE> &vector) {
            return Vector4(
                this->data[ 0] * vector.data[0] + this->data[ 1] * vector.data[1] + this->data[ 2] * vector.data[2] + this->data[ 3] * vector.data[3],
                this->data[ 4] * vector.data[0] + this->data[ 5] * vector.data[1] + this->data[ 6] * vector.data[2] + this->data[ 7] * vector.data[3],
                this->data[ 8] * vector.data[0] + this->data[ 9] * vector.data[1] + this->data[10] * vector.data[2] + this->data[11] * vector.data[3],
                this->data[12] * vector.data[0] + this->data[13] * vector.data[1] + this->data[14] * vector.data[2] + this->data[15] * vector.data[3]
            );
        }

        static Matrix4<DATA_TYPE> identity()
        {
            DATA_TYPE data[] = {
                1.0,0.0,0.0,0.0,
                0.0,1.0,0.0,0.0,
                0.0,0.0,1.0,0.0,
                0.0,0.0,0.0,1.0,
            };
            return Matrix4<DATA_TYPE>(data);
        }

        static Matrix4<DATA_TYPE> lookAt(const Vector3<DATA_TYPE> &eye, const Vector3<DATA_TYPE> &target, const Vector3<DATA_TYPE> &up)
        {
            return Matrix4<DATA_TYPE>::identity();
        }

        static Matrix4<DATA_TYPE> perspective(DATA_TYPE fov, DATA_TYPE aspectRatio, DATA_TYPE znear, DATA_TYPE zfar)
        {
            return Matrix4<DATA_TYPE>::identity();
        }

        static Matrix4<DATA_TYPE> ortho(DATA_TYPE width, DATA_TYPE height, DATA_TYPE znear, DATA_TYPE zfar)
        {
            return Matrix4<DATA_TYPE>::identity();
        }

        static Matrix4<DATA_TYPE> orthoOffCenter(DATA_TYPE left, DATA_TYPE top, DATA_TYPE right, DATA_TYPE bottom, DATA_TYPE znear, DATA_TYPE zfar)
        {
            return Matrix4<DATA_TYPE>::identity();
        }

        static Matrix4<DATA_TYPE> translate(const Vector3<DATA_TYPE> translation)
        {
            auto toReturn = Matrix4<DATA_TYPE>::identity();
            toReturn.data[3] = translation.data[0];
            toReturn.data[7] = translation.data[1];
            toReturn.data[11] = translation.data[2];
            return toReturn;
        }

        static Matrix4<DATA_TYPE> scale(const Vector3<DATA_TYPE> scale)
        {
            auto toReturn = Matrix4<DATA_TYPE>(0);
            toReturn.data[0] = scale.data[0];
            toReturn.data[5] = scale.data[1];
            toReturn.data[10] = scale.data[2];
            toReturn.data[15] = 1.0;
            return toReturn;
        }

        static Matrix4<DATA_TYPE> rotate(const Quaternion<DATA_TYPE> &rotation)
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

        static Matrix4<DATA_TYPE> rotateX(DATA_TYPE angle)
        {
            DATA_TYPE cos = cos(angle);
            DATA_TYPE sin = sin(angle);
            auto toReturn = Matrix4<DATA_TYPE>::identity();
            toReturn.data[5] = cos;
            toReturn.data[6] = sin;
            toReturn.data[9] = -sin;
            toReturn.data[10] = cos;
            return toReturn;
        }

        static Matrix4<DATA_TYPE> rotateY(DATA_TYPE angle)
        {
            DATA_TYPE cos = cos(angle);
            DATA_TYPE sin = sin(angle);
            auto toReturn = Matrix4<DATA_TYPE>::identity();
            toReturn.data[0] = cos;
            toReturn.data[2] = -sin;
            toReturn.data[8] = sin;
            toReturn.data[10] = cos;
            return toReturn;
        }

        static Matrix4<DATA_TYPE> rotateZ(DATA_TYPE angle)
        {
            DATA_TYPE cos = cos(angle);
            DATA_TYPE sin = sin(angle);
            auto toReturn = Matrix4<DATA_TYPE>::identity();
            toReturn.data[0] = cos;
            toReturn.data[2] = -sin;
            toReturn.data[8] = sin;
            toReturn.data[10] = cos;
            return toReturn;
        }
    };

}
