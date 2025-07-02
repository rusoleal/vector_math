#pragma once

#include<vector_math/vec.hpp>

namespace systems::leal::vector_math {

    template<class DATA_TYPE, uint32_t ROWS, uint32_t COLS>
    class Mat:public Vec<DATA_TYPE,ROWS*COLS> {
    public:
        Mat():Vec<DATA_TYPE,ROWS*COLS>(){}
        Mat(DATA_TYPE value):Vec<DATA_TYPE,ROWS*COLS>(value) {}

        ///////////////
        // operators //
        ///////////////

        template<uint32_t N>
        Mat<DATA_TYPE, ROWS, N> operator*(const Mat<DATA_TYPE,COLS,N> &rhs) const {
            Mat<DATA_TYPE, ROWS, N> toReturn;

            int toReturnIndex=0;
            for (int a=0; a<ROWS; a++) {
                for (int b=0; b<N; b++) {
                    DATA_TYPE value=0;
                    for (int c=0; c<COLS; c++) {
                        value += this->data[a*COLS+c] * rhs.data[c*N+b];
                    }
                    toReturn.data[toReturnIndex] = value;
                    toReturnIndex++;
                }
            }
            return toReturn;
        }

        Mat<DATA_TYPE, ROWS, COLS> operator*(DATA_TYPE scalar) const {
            Mat<DATA_TYPE, ROWS, COLS> toReturn;
            for (uint32_t a=0; a<this->Size; a++) {
                toReturn.data[a] = this->data[a] * scalar;
            }
            return toReturn;
        }

        Mat<DATA_TYPE,COLS,ROWS> transpose() const {
            Mat<DATA_TYPE,COLS,ROWS> toReturn;

            int ownIndex=0;
            for (int a=0; a<COLS; a++)
                for (int b=0; b<ROWS; b++) {
                    toReturn.data[COLS*b + a] = this->data[ownIndex];
                    ownIndex++;
                }

            return toReturn;
        }

    };

}