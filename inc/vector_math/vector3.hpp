#pragma once

#include<vector_math/vec.hpp>

namespace systems::leal::vector_math {


    template<class DATA_TYPE>
    class Vector3:public Vec<DATA_TYPE,3> {
    public:

        //////////////////
        // constructors //
        //////////////////
        Vector3() = default;
        Vector3(DATA_TYPE value);
        Vector3(DATA_TYPE x, DATA_TYPE y, DATA_TYPE z);

        ///////////////
        // operators //
        ///////////////
        Vector3<DATA_TYPE> operator-(const Vector3<DATA_TYPE> &other) const;
        Vector3<DATA_TYPE> operator-() const;

        /////////////
        // methods //
        /////////////
        static Vector3<DATA_TYPE> cross(const Vector3<DATA_TYPE> &lhs, const Vector3<DATA_TYPE> &rhs);
        DATA_TYPE dot(const Vector3<DATA_TYPE> &rhs) const;
        static DATA_TYPE dot(const Vector3<DATA_TYPE> &lhs, const Vector3<DATA_TYPE> &rhs);

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
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator-(const Vector3<DATA_TYPE> &other) const {
        auto toReturn = ((Vec<DATA_TYPE,3>)*this) - (Vec<DATA_TYPE,3>)other;
        return *(Vector3<DATA_TYPE> *)&toReturn;
    }

    template<class DATA_TYPE>
    Vector3<DATA_TYPE> Vector3<DATA_TYPE>::operator-() const {
        auto toReturn = -*(Vec<DATA_TYPE,3> *)this;
        return *(Vector3<DATA_TYPE> *)&toReturn;
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

}
