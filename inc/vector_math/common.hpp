#pragma once

#include <limits>
#include <cmath>

#if defined(__x86__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    #define __VECTOR_MATH_ARCH_X86_X64
    #undef __VECTOR_MATH_ARCH_ARM
#endif

#if defined(__arm__) || defined(__arm64__)
    #undef __VECTOR_MATH_ARCH_X86_X64
    #define __VECTOR_MATH_ARCH_ARM
#endif

namespace systems::leal::vector_math {

    template<typename DATA_TYPE>
    bool isZero(DATA_TYPE value) {
        return std::abs(value) < std::numeric_limits<DATA_TYPE>::epsilon();
    }

}