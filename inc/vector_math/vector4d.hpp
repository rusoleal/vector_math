#pragma once

#include <vector_math/vector4.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math {

    /// 32-byte aligned double 4D vector with SIMD-accelerated operations.
    ///
    /// Inherits the full Vector4<double> API. The arithmetic overloads below
    /// replace the generic implementations with architecture-specific paths:
    /// - **x86/x86-64**: AVX 256-bit intrinsics (`_mm256_load_pd`, etc.)
    /// - **AArch64**: NEON 128-bit intrinsics (`float64x2_t`, `vaddq_f64`, etc.)
    /// - **Other**: scalar fallback.
    ///
    /// @note The vector must be 32-byte aligned for AVX loads/stores.
    class alignas(32) Vector4d : public Vector4<double> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Vector4d() = default;

        /// Constructs from explicit x, y, z, w values.
        Vector4d(double x, double y, double z, double w) {
            this->data[0] = x;
            this->data[1] = y;
            this->data[2] = z;
            this->data[3] = w;
        }

        // ── SIMD arithmetic operators ─────────────────────────────────────────

        /// Component-wise addition.
        inline Vector4d operator+(const Vector4d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4d result;
                _mm256_store_pd(result.data,
                    _mm256_add_pd(_mm256_load_pd(this->data), _mm256_load_pd(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vaddq_f64(vld1q_f64(&this->data[0]), vld1q_f64(&rhs.data[0])));
                vst1q_f64(&result.data[2], vaddq_f64(vld1q_f64(&this->data[2]), vld1q_f64(&rhs.data[2])));
                return result;
            #else
                return Vector4d(this->data[0]+rhs.data[0], this->data[1]+rhs.data[1],
                                this->data[2]+rhs.data[2], this->data[3]+rhs.data[3]);
            #endif
        }

        /// Component-wise subtraction.
        inline Vector4d operator-(const Vector4d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4d result;
                _mm256_store_pd(result.data,
                    _mm256_sub_pd(_mm256_load_pd(this->data), _mm256_load_pd(rhs.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vsubq_f64(vld1q_f64(&this->data[0]), vld1q_f64(&rhs.data[0])));
                vst1q_f64(&result.data[2], vsubq_f64(vld1q_f64(&this->data[2]), vld1q_f64(&rhs.data[2])));
                return result;
            #else
                return Vector4d(this->data[0]-rhs.data[0], this->data[1]-rhs.data[1],
                                this->data[2]-rhs.data[2], this->data[3]-rhs.data[3]);
            #endif
        }

        /// Negation.
        inline Vector4d operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4d result;
                _mm256_store_pd(result.data,
                    _mm256_xor_pd(_mm256_load_pd(this->data), _mm256_set1_pd(-0.0)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vnegq_f64(vld1q_f64(&this->data[0])));
                vst1q_f64(&result.data[2], vnegq_f64(vld1q_f64(&this->data[2])));
                return result;
            #else
                return Vector4d(-this->data[0], -this->data[1], -this->data[2], -this->data[3]);
            #endif
        }

        /// Scalar multiplication: `v * s`.
        inline Vector4d operator*(double scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4d result;
                _mm256_store_pd(result.data,
                    _mm256_mul_pd(_mm256_load_pd(this->data), _mm256_set1_pd(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vmulq_n_f64(vld1q_f64(&this->data[0]), scalar));
                vst1q_f64(&result.data[2], vmulq_n_f64(vld1q_f64(&this->data[2]), scalar));
                return result;
            #else
                return Vector4d(this->data[0]*scalar, this->data[1]*scalar,
                                this->data[2]*scalar, this->data[3]*scalar);
            #endif
        }

        /// Scalar multiplication (commutative): `s * v`.
        friend inline Vector4d operator*(double scalar, const Vector4d& v) { return v * scalar; }

        /// Scalar division.
        inline Vector4d operator/(double scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Vector4d result;
                _mm256_store_pd(result.data,
                    _mm256_div_pd(_mm256_load_pd(this->data), _mm256_set1_pd(scalar)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                double inv = 1.0 / scalar;
                vst1q_f64(&result.data[0], vmulq_n_f64(vld1q_f64(&this->data[0]), inv));
                vst1q_f64(&result.data[2], vmulq_n_f64(vld1q_f64(&this->data[2]), inv));
                return result;
            #else
                return Vector4d(this->data[0]/scalar, this->data[1]/scalar,
                                this->data[2]/scalar, this->data[3]/scalar);
            #endif
        }

        // ── SIMD dot product ──────────────────────────────────────────────────

        /// Returns the dot product `this · rhs`.
        ///
        /// x86: multiply all 4 lanes, hadd pairs, extract 128-bit halves and add.
        /// AArch64: pairwise add after lane-multiply, then vaddvq_f64 for final sum.
        inline double dot(const Vector4d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m256d mul   = _mm256_mul_pd(_mm256_load_pd(this->data), _mm256_load_pd(rhs.data));
                __m256d hadd  = _mm256_hadd_pd(mul, mul);
                __m128d lo    = _mm256_castpd256_pd128(hadd);
                __m128d hi    = _mm256_extractf128_pd(hadd, 1);
                return _mm_cvtsd_f64(_mm_add_pd(lo, hi));
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                float64x2_t prod_lo = vmulq_f64(vld1q_f64(&this->data[0]), vld1q_f64(&rhs.data[0]));
                float64x2_t prod_hi = vmulq_f64(vld1q_f64(&this->data[2]), vld1q_f64(&rhs.data[2]));
                return vaddvq_f64(vpaddq_f64(prod_lo, prod_hi));
            #else
                return this->data[0]*rhs.data[0] + this->data[1]*rhs.data[1]
                     + this->data[2]*rhs.data[2] + this->data[3]*rhs.data[3];
            #endif
        }

        /// Static version: returns `lhs · rhs`.
        static inline double dot(const Vector4d& lhs, const Vector4d& rhs) { return lhs.dot(rhs); }

    };

}
