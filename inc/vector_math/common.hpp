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

// ---------------------------------------------------------------------------
// x86/x64 runtime CPU feature detection
//
// The vector_math headers themselves use compile-time ISA macros (__AVX__,
// __AVX2__, etc.) so that inline operators remain branch-free. These helpers
// are provided for consumers that want to query capabilities at runtime, e.g.
// to decide whether to compile a separate AVX2-dispatched kernel or to choose
// a code path at a higher level.
// ---------------------------------------------------------------------------
#if defined(__VECTOR_MATH_ARCH_X86_X64)
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif

namespace systems::leal::vector_math {

    /// Returns true if the CPU and operating system support AVX.
    /// When the translation unit is compiled with -mavx (or /arch:AVX on MSVC)
    /// this short-circuits to true.
    inline bool cpuSupportsAVX() {
        #if defined(__AVX__)
            return true;
        #elif defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports("avx");
        #elif defined(_MSC_VER)
            int cpuInfo[4];
            __cpuid(cpuInfo, 1);
            return (cpuInfo[2] & (1 << 28)) != 0;
        #else
            return false;
        #endif
    }

    /// Returns true if the CPU and operating system support AVX2.
    /// When the translation unit is compiled with -mavx2 (or /arch:AVX2 on MSVC)
    /// this short-circuits to true.
    inline bool cpuSupportsAVX2() {
        #if defined(__AVX2__)
            return true;
        #elif defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports("avx2");
        #elif defined(_MSC_VER)
            int cpuInfo[4];
            __cpuidex(cpuInfo, 7, 0);
            return (cpuInfo[1] & (1 << 5)) != 0;
        #else
            return false;
        #endif
    }

} // namespace systems::leal::vector_math
#endif // __VECTOR_MATH_ARCH_X86_X64

namespace systems::leal::vector_math {

    /// Returns true if @p value is within machine epsilon of zero.
    template<typename DATA_TYPE>
    bool isZero(DATA_TYPE value) {
        return std::abs(value) < std::numeric_limits<DATA_TYPE>::epsilon();
    }

}
