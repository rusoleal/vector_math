#pragma once

#include<vector_math/mat.hpp>

namespace systems::leal::vector_math {

    /// 2×2 matrix of type @p DATA_TYPE.
    ///
    /// Inherits all Mat<DATA_TYPE,2,2> operations (multiplication, transpose, etc.)
    /// and Vec<DATA_TYPE,4> utilities (dot, lerp, clamp, …).
    /// Elements are stored row-major: `data[r*2 + c]` = M[r][c].
    template<class DATA_TYPE>
    class Matrix2:public Mat<DATA_TYPE,2,2> {
    public:
        /// Default-constructs all elements to zero.
        Matrix2() = default;
    };

}
