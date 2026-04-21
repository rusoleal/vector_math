#pragma once

#include <limits>
#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

/// @defgroup arch Architecture Detection
/// Compile-time architecture macros set by common.hpp.
/// - `__VECTOR_MATH_ARCH_X86_X64` — defined on x86/x86-64; enables x86 SIMD code paths.
/// - `__VECTOR_MATH_ARCH_ARM`     — defined on ARM/ARM64; enables NEON paths.

#if defined(__x86__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    #define __VECTOR_MATH_ARCH_X86_X64
    #undef __VECTOR_MATH_ARCH_ARM
#endif

#if defined(__arm__) || defined(__arm64__)
    #undef __VECTOR_MATH_ARCH_X86_X64
    #define __VECTOR_MATH_ARCH_ARM
#endif

namespace systems::leal::vector_math {

    /// Returns true if @p value is within machine epsilon of zero.
    template<typename DATA_TYPE>
    bool isZero(DATA_TYPE value) {
        return std::abs(value) < std::numeric_limits<DATA_TYPE>::epsilon();
    }

}
