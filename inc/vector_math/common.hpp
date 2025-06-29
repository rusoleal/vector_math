#pragma once

#include <limits>

namespace systems::leal::vector_math {

    template<typename DATA_TYPE>
    bool isZero(DATA_TYPE value) {
        return std::abs(value) < std::numeric_limits<DATA_TYPE>::epsilon();
    }

}