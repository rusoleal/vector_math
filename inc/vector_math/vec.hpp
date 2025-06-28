#pragma once

#include <cstdlib>

namespace systems::leal::vector_math {

    template<class DATA_TYPE, uint32_t SIZE>
    class Vec{
    public:
        typedef DATA_TYPE DataType;

        enum Params { Size = SIZE };

        DATA_TYPE data[SIZE];

        Vec() {
            for (uint32_t a=0; a<this->Size; a++) {
                data[a] = 0;
            }
        }

        Vec(DATA_TYPE value) {
            for (uint32_t a=0; a<this->Size; a++) {
                data[a] = value;
            }
        }

        bool operator==(const Vec<DATA_TYPE, SIZE> &other) const {
            for (uint32_t a=0; a<this->Size; a++) {
                if (data[a] != other.data[a]) return false;
            }
            return true;
        }

        bool operator!=(const Vec<DATA_TYPE, SIZE> &other) const {
            return !operator==(other);
        }

        Vec<DATA_TYPE, SIZE> operator+(const Vec<DATA_TYPE, SIZE> &other) const {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<this->Size; a++) {
                toReturn.data[a] = data[a] + other.data[a];
            }
            return toReturn;
        }

        Vec<DATA_TYPE, SIZE> operator-(const Vec<DATA_TYPE, SIZE> &other) const {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<this->Size; a++) {
                toReturn.data[a] = data[a] - other.data[a];
            }
            return toReturn;
        }

        Vec<DATA_TYPE, SIZE> operator*(DATA_TYPE scalar) const {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<this->Size; a++) {
                toReturn.data[a] = data[a] * scalar;
            }
            return toReturn;
        }

        friend Vec<DATA_TYPE, SIZE> operator*(DATA_TYPE scalar, const Vec<DATA_TYPE, SIZE> &lhs) {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<lhs.Size; a++) {
                toReturn.data[a] = lhs.data[a] * scalar;
            }
            return toReturn;
        }

        Vec<DATA_TYPE, SIZE> operator/(DATA_TYPE scalar) const {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<this->Size; a++) {
                toReturn.data[a] = data[a] / scalar;
            }
            return toReturn;
        }

        friend Vec<DATA_TYPE, SIZE> operator/(const Vec<DATA_TYPE, SIZE> &lhs, DATA_TYPE scalar) {
            Vec<DATA_TYPE, SIZE> toReturn;
            for (uint32_t a=0; a<lhs.Size; a++) {
                toReturn.data[a] = lhs.data[a] / scalar;
            }
            return toReturn;
        }
    };


}
