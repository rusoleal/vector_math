#pragma once

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4d.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math
{
    /// 32-byte aligned double 4×4 matrix with SIMD-accelerated operations.
    ///
    /// Inherits the full Matrix4<double> API. The `operator*` overloads below
    /// replace the generic implementations with architecture-specific paths:
    /// - **x86/x86-64**: AVX 256-bit intrinsics (`_mm256_load_pd`, `_mm256_mul_pd`, etc.)
    ///   FMA3 intrinsics (`_mm256_fmadd_pd`) are used when compiled with `-mfma`.
    /// - **AArch64**: NEON 128-bit intrinsics (`float64x2_t`, `vfmaq_f64`, `vpaddq_f64`)
    /// - **ARM 32-bit**: scalar fallback (no double-precision NEON on ARMv7)
    /// - **Other**: scalar fallback.
    ///
    /// @note Both the matrix and the Vector4d operand must be 32-byte aligned.
    class alignas(32) Matrix4d : public Matrix4<double> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Matrix4d() : Matrix4<double>() {}

        /// Constructs all elements from a single @p value.
        Matrix4d(double value) : Matrix4<double>(value) {}

        /// Constructs from a 16-element row-major C array.
        Matrix4d(double buffer[16]) : Matrix4<double>(buffer) {}

        /// Constructs from a base Matrix4<double> (e.g. returned by static factories).
        Matrix4d(const Matrix4<double>& m) : Matrix4<double>(m) {}

        // ── SIMD-accelerated operators ────────────────────────────────────────

        /// Matrix–matrix multiplication using SIMD intrinsics.
        /// x86: each output row = broadcast(A[i,*]) · rows of B, accumulated via FMA when available.
        inline Matrix4d operator*(const Matrix4d &rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4d toReturn;
                __m256d row1 = _mm256_load_pd(&rhs.data[0]);
                __m256d row2 = _mm256_load_pd(&rhs.data[4]);
                __m256d row3 = _mm256_load_pd(&rhs.data[8]);
                __m256d row4 = _mm256_load_pd(&rhs.data[12]);
                for(int i=0; i<4; i++) {
                    __m256d brod1 = _mm256_set1_pd(this->data[4*i + 0]);
                    __m256d brod2 = _mm256_set1_pd(this->data[4*i + 1]);
                    __m256d brod3 = _mm256_set1_pd(this->data[4*i + 2]);
                    __m256d brod4 = _mm256_set1_pd(this->data[4*i + 3]);
                    #ifdef __FMA__
                        __m256d row = _mm256_fmadd_pd(brod4, row4,
                                       _mm256_fmadd_pd(brod3, row3,
                                        _mm256_fmadd_pd(brod2, row2,
                                         _mm256_mul_pd(brod1, row1))));
                    #else
                        __m256d row = _mm256_add_pd(
                                    _mm256_add_pd(
                                        _mm256_mul_pd(brod1, row1),
                                        _mm256_mul_pd(brod2, row2)),
                                    _mm256_add_pd(
                                        _mm256_mul_pd(brod3, row3),
                                        _mm256_mul_pd(brod4, row4)));
                    #endif
                    _mm256_store_pd(&toReturn.data[4*i], row);
                }
                return toReturn;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                #if defined(__arm64__) || defined(__aarch64__)
                    // AArch64 NEON: each row of B is split into two float64x2_t halves.
                    // For each row i of A, broadcast each scalar element and accumulate
                    // via FMA across the four row-halves of B.
                    Matrix4d toReturn;
                    float64x2_t B0_lo = vld1q_f64(&rhs.data[0]);
                    float64x2_t B0_hi = vld1q_f64(&rhs.data[2]);
                    float64x2_t B1_lo = vld1q_f64(&rhs.data[4]);
                    float64x2_t B1_hi = vld1q_f64(&rhs.data[6]);
                    float64x2_t B2_lo = vld1q_f64(&rhs.data[8]);
                    float64x2_t B2_hi = vld1q_f64(&rhs.data[10]);
                    float64x2_t B3_lo = vld1q_f64(&rhs.data[12]);
                    float64x2_t B3_hi = vld1q_f64(&rhs.data[14]);
                    for (int i = 0; i < 4; i++) {
                        float64x2_t a0 = vdupq_n_f64(this->data[i*4 + 0]);
                        float64x2_t a1 = vdupq_n_f64(this->data[i*4 + 1]);
                        float64x2_t a2 = vdupq_n_f64(this->data[i*4 + 2]);
                        float64x2_t a3 = vdupq_n_f64(this->data[i*4 + 3]);
                        float64x2_t r_lo = vmulq_f64(a0, B0_lo);
                        float64x2_t r_hi = vmulq_f64(a0, B0_hi);
                        r_lo = vfmaq_f64(r_lo, a1, B1_lo);
                        r_hi = vfmaq_f64(r_hi, a1, B1_hi);
                        r_lo = vfmaq_f64(r_lo, a2, B2_lo);
                        r_hi = vfmaq_f64(r_hi, a2, B2_hi);
                        r_lo = vfmaq_f64(r_lo, a3, B3_lo);
                        r_hi = vfmaq_f64(r_hi, a3, B3_hi);
                        vst1q_f64(&toReturn.data[i*4 + 0], r_lo);
                        vst1q_f64(&toReturn.data[i*4 + 2], r_hi);
                    }
                    return toReturn;
                #else
                    auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
                    return *(Matrix4d *)&toReturn;
                #endif
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
                return *(Matrix4d *)&toReturn;
            #endif
        }

        /// Matrix–Vector4d multiplication using SIMD intrinsics.
        /// Uses a row-dot-product pattern: each result element is the dot product of
        /// a matrix row with the input vector, computed via pairwise FMA + vpaddq_f64.
        inline Vector4d operator*(const Vector4d &rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m256d row0 = _mm256_load_pd(&this->data[0]);
                __m256d row1 = _mm256_load_pd(&this->data[4]);
                __m256d row2 = _mm256_load_pd(&this->data[8]);
                __m256d row3 = _mm256_load_pd(&this->data[12]);
                // Transpose: produce the four matrix columns
                __m256d t0   = _mm256_unpacklo_pd(row0, row1); // [m00,m10,m02,m12]
                __m256d t1   = _mm256_unpackhi_pd(row0, row1); // [m01,m11,m03,m13]
                __m256d t2   = _mm256_unpacklo_pd(row2, row3); // [m20,m30,m22,m32]
                __m256d t3   = _mm256_unpackhi_pd(row2, row3); // [m21,m31,m23,m33]
                __m256d col0 = _mm256_permute2f128_pd(t0, t2, 0x20); // [m00,m10,m20,m30]
                __m256d col1 = _mm256_permute2f128_pd(t1, t3, 0x20); // [m01,m11,m21,m31]
                __m256d col2 = _mm256_permute2f128_pd(t0, t2, 0x31); // [m02,m12,m22,m32]
                __m256d col3 = _mm256_permute2f128_pd(t1, t3, 0x31); // [m03,m13,m23,m33]
                // Broadcast each vector component and accumulate
                __m256d vx = _mm256_set1_pd(rhs.data[0]);
                __m256d vy = _mm256_set1_pd(rhs.data[1]);
                __m256d vz = _mm256_set1_pd(rhs.data[2]);
                __m256d vw = _mm256_set1_pd(rhs.data[3]);
                #ifdef __FMA__
                    __m256d result = _mm256_fmadd_pd(vw, col3,
                                      _mm256_fmadd_pd(vz, col2,
                                       _mm256_fmadd_pd(vy, col1,
                                        _mm256_mul_pd(vx, col0))));
                #else
                    __m256d result = _mm256_add_pd(
                        _mm256_add_pd(_mm256_mul_pd(vx, col0), _mm256_mul_pd(vy, col1)),
                        _mm256_add_pd(_mm256_mul_pd(vz, col2), _mm256_mul_pd(vw, col3))
                    );
                #endif
                Vector4d toReturn;
                _mm256_store_pd(toReturn.data, result);
                return toReturn;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                #if defined(__arm64__) || defined(__aarch64__)
                    // AArch64 NEON: load vector as two float64x2_t halves.
                    // Each result element = dot(row_i, v). Compute two dot products at
                    // once: multiply row halves element-wise, then vpaddq_f64 for the
                    // horizontal sum → [dot(row_i, v), dot(row_{i+1}, v)].
                    float64x2_t v_lo = vld1q_f64(&rhs.data[0]);
                    float64x2_t v_hi = vld1q_f64(&rhs.data[2]);
                    float64x2_t r0 = vfmaq_f64(vmulq_f64(vld1q_f64(&this->data[0]),  v_lo), vld1q_f64(&this->data[2]),  v_hi);
                    float64x2_t r1 = vfmaq_f64(vmulq_f64(vld1q_f64(&this->data[4]),  v_lo), vld1q_f64(&this->data[6]),  v_hi);
                    float64x2_t r2 = vfmaq_f64(vmulq_f64(vld1q_f64(&this->data[8]),  v_lo), vld1q_f64(&this->data[10]), v_hi);
                    float64x2_t r3 = vfmaq_f64(vmulq_f64(vld1q_f64(&this->data[12]), v_lo), vld1q_f64(&this->data[14]), v_hi);
                    Vector4d toReturn;
                    vst1q_f64(&toReturn.data[0], vpaddq_f64(r0, r1));
                    vst1q_f64(&toReturn.data[2], vpaddq_f64(r2, r3));
                    return toReturn;
                #else
                    auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
                    return *(Vector4d *)&toReturn;
                #endif
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator*(rhs);
                return *(Vector4d *)&toReturn;
            #endif
        }

        /// Component-wise addition using SIMD intrinsics.
        /// x86: 4 AVX 256-bit loads/adds/stores (one per row). AArch64: 8 NEON 128-bit ops.
        inline Matrix4d operator+(const Matrix4d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4d result;
                _mm256_store_pd(&result.data[0],  _mm256_add_pd(_mm256_load_pd(&this->data[0]),  _mm256_load_pd(&rhs.data[0])));
                _mm256_store_pd(&result.data[4],  _mm256_add_pd(_mm256_load_pd(&this->data[4]),  _mm256_load_pd(&rhs.data[4])));
                _mm256_store_pd(&result.data[8],  _mm256_add_pd(_mm256_load_pd(&this->data[8]),  _mm256_load_pd(&rhs.data[8])));
                _mm256_store_pd(&result.data[12], _mm256_add_pd(_mm256_load_pd(&this->data[12]), _mm256_load_pd(&rhs.data[12])));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Matrix4d result;
                for (int i = 0; i < 16; i += 2) {
                    vst1q_f64(&result.data[i],
                        vaddq_f64(vld1q_f64(&this->data[i]), vld1q_f64(&rhs.data[i])));
                }
                return result;
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator+(rhs);
                return *(Matrix4d *)&toReturn;
            #endif
        }

        /// Component-wise subtraction using SIMD intrinsics.
        inline Matrix4d operator-(const Matrix4d& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4d result;
                _mm256_store_pd(&result.data[0],  _mm256_sub_pd(_mm256_load_pd(&this->data[0]),  _mm256_load_pd(&rhs.data[0])));
                _mm256_store_pd(&result.data[4],  _mm256_sub_pd(_mm256_load_pd(&this->data[4]),  _mm256_load_pd(&rhs.data[4])));
                _mm256_store_pd(&result.data[8],  _mm256_sub_pd(_mm256_load_pd(&this->data[8]),  _mm256_load_pd(&rhs.data[8])));
                _mm256_store_pd(&result.data[12], _mm256_sub_pd(_mm256_load_pd(&this->data[12]), _mm256_load_pd(&rhs.data[12])));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Matrix4d result;
                for (int i = 0; i < 16; i += 2) {
                    vst1q_f64(&result.data[i],
                        vsubq_f64(vld1q_f64(&this->data[i]), vld1q_f64(&rhs.data[i])));
                }
                return result;
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator-(rhs);
                return *(Matrix4d *)&toReturn;
            #endif
        }

        /// Negation using SIMD intrinsics (flips sign bit via XOR with -0.0).
        inline Matrix4d operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4d result;
                __m256d sign_mask = _mm256_set1_pd(-0.0);
                _mm256_store_pd(&result.data[0],  _mm256_xor_pd(_mm256_load_pd(&this->data[0]),  sign_mask));
                _mm256_store_pd(&result.data[4],  _mm256_xor_pd(_mm256_load_pd(&this->data[4]),  sign_mask));
                _mm256_store_pd(&result.data[8],  _mm256_xor_pd(_mm256_load_pd(&this->data[8]),  sign_mask));
                _mm256_store_pd(&result.data[12], _mm256_xor_pd(_mm256_load_pd(&this->data[12]), sign_mask));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Matrix4d result;
                for (int i = 0; i < 16; i += 2)
                    vst1q_f64(&result.data[i], vnegq_f64(vld1q_f64(&this->data[i])));
                return result;
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator-();
                return *(Matrix4d *)&toReturn;
            #endif
        }

        /// Scalar multiplication using SIMD intrinsics.
        inline Matrix4d operator*(double scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4d result;
                __m256d s = _mm256_set1_pd(scalar);
                _mm256_store_pd(&result.data[0],  _mm256_mul_pd(_mm256_load_pd(&this->data[0]),  s));
                _mm256_store_pd(&result.data[4],  _mm256_mul_pd(_mm256_load_pd(&this->data[4]),  s));
                _mm256_store_pd(&result.data[8],  _mm256_mul_pd(_mm256_load_pd(&this->data[8]),  s));
                _mm256_store_pd(&result.data[12], _mm256_mul_pd(_mm256_load_pd(&this->data[12]), s));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Matrix4d result;
                for (int i = 0; i < 16; i += 2)
                    vst1q_f64(&result.data[i], vmulq_n_f64(vld1q_f64(&this->data[i]), scalar));
                return result;
            #else
                auto toReturn = ((Matrix4<double> *)this)->operator*(scalar);
                return *(Matrix4d *)&toReturn;
            #endif
        }

        /// Builds a right-handed view matrix — optimized override of Matrix4<double>::lookAt.
        ///
        /// Inlines all 3D vector math to avoid generic Vec<> loop overhead, intermediate
        /// Vector3 temporaries, isZero checks, and the identity() initialization cost.
        static inline Matrix4d lookAt(const Vector3<double> &eye, const Vector3<double> &target, const Vector3<double> &up)
        {
            // zAxis = normalize(target - eye)
            double zx = target.data[0] - eye.data[0];
            double zy = target.data[1] - eye.data[1];
            double zz = target.data[2] - eye.data[2];
            double zInvLen = 1.0 / std::sqrt(zx*zx + zy*zy + zz*zz);
            zx *= zInvLen; zy *= zInvLen; zz *= zInvLen;

            // xAxis = normalize(cross(up, zAxis))
            double xx = up.data[1]*zz - up.data[2]*zy;
            double xy = up.data[2]*zx - up.data[0]*zz;
            double xz = up.data[0]*zy - up.data[1]*zx;
            double xInvLen = 1.0 / std::sqrt(xx*xx + xy*xy + xz*xz);
            xx *= xInvLen; xy *= xInvLen; xz *= xInvLen;

            // yAxis = cross(zAxis, xAxis) — unit length by construction, no normalize needed
            double yx = zy*xz - zz*xy;
            double yy = zz*xx - zx*xz;
            double yz = zx*xy - zy*xx;

            double buf[16] = {
                xx,  xy,  xz,  -(xx*eye.data[0] + xy*eye.data[1] + xz*eye.data[2]),
                yx,  yy,  yz,  -(yx*eye.data[0] + yy*eye.data[1] + yz*eye.data[2]),
                zx,  zy,  zz,  -(zx*eye.data[0] + zy*eye.data[1] + zz*eye.data[2]),
                0.0, 0.0, 0.0, 1.0
            };
            return Matrix4d(buf);
        }

    };
}
