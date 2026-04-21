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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
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

        /// Returns the squared Euclidean length.
        inline double lengthSquared() const { return dot(*this); }

        /// In-place normalization using the SIMD dot path.
        inline void normalize() {
            double lenSqr = lengthSquared();
            if (isZero<double>(lenSqr)) return;
            *this = *this * (1.0 / std::sqrt(lenSqr));
        }

        /// Returns a normalized copy.
        inline Vector4d normalized() const {
            Vector4d result = *this;
            result.normalize();
            return result;
        }

        /// Returns the squared Euclidean distance to @p other.
        inline double distanceToSquared(const Vector4d& other) const {
            Vector4d delta = *this - other;
            return delta.dot(delta);
        }

        /// Replaces each component with its absolute value.
        inline void absolute() {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                __m256d value = _mm256_load_pd(this->data);
                __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffffLL));
                _mm256_store_pd(this->data, _mm256_and_pd(value, mask));
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                vst1q_f64(&this->data[0], vabsq_f64(vld1q_f64(&this->data[0])));
                vst1q_f64(&this->data[2], vabsq_f64(vld1q_f64(&this->data[2])));
            #else
                this->data[0] = std::abs(this->data[0]);
                this->data[1] = std::abs(this->data[1]);
                this->data[2] = std::abs(this->data[2]);
                this->data[3] = std::abs(this->data[3]);
            #endif
        }

        /// Clamps each component to [minVal, maxVal].
        inline void clamp(double minVal, double maxVal) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                __m256d value = _mm256_load_pd(this->data);
                __m256d minV = _mm256_set1_pd(minVal);
                __m256d maxV = _mm256_set1_pd(maxVal);
                _mm256_store_pd(this->data, _mm256_min_pd(_mm256_max_pd(value, minV), maxV));
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                float64x2_t minLo = vdupq_n_f64(minVal);
                float64x2_t maxLo = vdupq_n_f64(maxVal);
                vst1q_f64(&this->data[0], vminq_f64(vmaxq_f64(vld1q_f64(&this->data[0]), minLo), maxLo));
                vst1q_f64(&this->data[2], vminq_f64(vmaxq_f64(vld1q_f64(&this->data[2]), minLo), maxLo));
            #else
                this->data[0] = this->data[0] < minVal ? minVal : (this->data[0] > maxVal ? maxVal : this->data[0]);
                this->data[1] = this->data[1] < minVal ? minVal : (this->data[1] > maxVal ? maxVal : this->data[1]);
                this->data[2] = this->data[2] < minVal ? minVal : (this->data[2] > maxVal ? maxVal : this->data[2]);
                this->data[3] = this->data[3] < minVal ? minVal : (this->data[3] > maxVal ? maxVal : this->data[3]);
            #endif
        }

        /// Clamps each component independently to [minVec[i], maxVec[i]].
        inline void clamp(const Vector4d& minVec, const Vector4d& maxVec) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                __m256d value = _mm256_load_pd(this->data);
                __m256d minV = _mm256_load_pd(minVec.data);
                __m256d maxV = _mm256_load_pd(maxVec.data);
                _mm256_store_pd(this->data, _mm256_min_pd(_mm256_max_pd(value, minV), maxV));
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                vst1q_f64(&this->data[0], vminq_f64(vmaxq_f64(vld1q_f64(&this->data[0]), vld1q_f64(&minVec.data[0])), vld1q_f64(&maxVec.data[0])));
                vst1q_f64(&this->data[2], vminq_f64(vmaxq_f64(vld1q_f64(&this->data[2]), vld1q_f64(&minVec.data[2])), vld1q_f64(&maxVec.data[2])));
            #else
                this->data[0] = this->data[0] < minVec.data[0] ? minVec.data[0] : (this->data[0] > maxVec.data[0] ? maxVec.data[0] : this->data[0]);
                this->data[1] = this->data[1] < minVec.data[1] ? minVec.data[1] : (this->data[1] > maxVec.data[1] ? maxVec.data[1] : this->data[1]);
                this->data[2] = this->data[2] < minVec.data[2] ? minVec.data[2] : (this->data[2] > maxVec.data[2] ? maxVec.data[2] : this->data[2]);
                this->data[3] = this->data[3] < minVec.data[3] ? minVec.data[3] : (this->data[3] > maxVec.data[3] ? maxVec.data[3] : this->data[3]);
            #endif
        }

        /// Returns the component-wise minimum.
        static inline Vector4d min(const Vector4d& a, const Vector4d& b) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector4d result;
                _mm256_store_pd(result.data, _mm256_min_pd(_mm256_load_pd(a.data), _mm256_load_pd(b.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vminq_f64(vld1q_f64(&a.data[0]), vld1q_f64(&b.data[0])));
                vst1q_f64(&result.data[2], vminq_f64(vld1q_f64(&a.data[2]), vld1q_f64(&b.data[2])));
                return result;
            #else
                return Vector4d(
                    a.data[0] < b.data[0] ? a.data[0] : b.data[0],
                    a.data[1] < b.data[1] ? a.data[1] : b.data[1],
                    a.data[2] < b.data[2] ? a.data[2] : b.data[2],
                    a.data[3] < b.data[3] ? a.data[3] : b.data[3]
                );
            #endif
        }

        /// Returns the component-wise maximum.
        static inline Vector4d max(const Vector4d& a, const Vector4d& b) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector4d result;
                _mm256_store_pd(result.data, _mm256_max_pd(_mm256_load_pd(a.data), _mm256_load_pd(b.data)));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                vst1q_f64(&result.data[0], vmaxq_f64(vld1q_f64(&a.data[0]), vld1q_f64(&b.data[0])));
                vst1q_f64(&result.data[2], vmaxq_f64(vld1q_f64(&a.data[2]), vld1q_f64(&b.data[2])));
                return result;
            #else
                return Vector4d(
                    a.data[0] > b.data[0] ? a.data[0] : b.data[0],
                    a.data[1] > b.data[1] ? a.data[1] : b.data[1],
                    a.data[2] > b.data[2] ? a.data[2] : b.data[2],
                    a.data[3] > b.data[3] ? a.data[3] : b.data[3]
                );
            #endif
        }

        /// Linear interpolation: returns a + t * (b - a).
        static inline Vector4d lerp(const Vector4d& a, const Vector4d& b, double t) {
            #if defined(__VECTOR_MATH_ARCH_X86_X64) && defined(__AVX__)
                Vector4d result;
                __m256d av = _mm256_load_pd(a.data);
                __m256d bv = _mm256_load_pd(b.data);
                __m256d tv = _mm256_set1_pd(t);
                _mm256_store_pd(result.data, _mm256_add_pd(av, _mm256_mul_pd(tv, _mm256_sub_pd(bv, av))));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Vector4d result;
                float64x2_t tv = vdupq_n_f64(t);
                vst1q_f64(&result.data[0], vaddq_f64(vld1q_f64(&a.data[0]), vmulq_f64(tv, vsubq_f64(vld1q_f64(&b.data[0]), vld1q_f64(&a.data[0])))));
                vst1q_f64(&result.data[2], vaddq_f64(vld1q_f64(&a.data[2]), vmulq_f64(tv, vsubq_f64(vld1q_f64(&b.data[2]), vld1q_f64(&a.data[2])))));
                return result;
            #else
                return Vector4d(
                    a.data[0] + t * (b.data[0] - a.data[0]),
                    a.data[1] + t * (b.data[1] - a.data[1]),
                    a.data[2] + t * (b.data[2] - a.data[2]),
                    a.data[3] + t * (b.data[3] - a.data[3])
                );
            #endif
        }

    };

}
