#pragma once

#include<vector_math/mat.hpp>

namespace systems::leal::vector_math {

    /// 3×3 matrix of type @p DATA_TYPE.
    ///
    /// Inherits all Mat<DATA_TYPE,3,3> operations (multiplication, transpose, etc.)
    /// and Vec<DATA_TYPE,9> utilities (dot, lerp, clamp, …).
    /// Elements are stored row-major: `data[r*3 + c]` = M[r][c].
    template<class DATA_TYPE>
    class Matrix3:public Mat<DATA_TYPE,3,3> {
    public:
        /// Default-constructs all elements to zero.
        Matrix3() = default;

    };

}
