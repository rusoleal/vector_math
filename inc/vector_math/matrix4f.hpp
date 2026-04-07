#pragma once

#include <vector_math/matrix4.hpp>
#include <vector_math/vector4f.hpp>
#include <vector_math/common.hpp>

#ifdef __VECTOR_MATH_ARCH_X86_X64
    #include <immintrin.h>
#elif defined(__VECTOR_MATH_ARCH_ARM)
    #include <arm_neon.h>
#endif

namespace systems::leal::vector_math
{
    /// 16-byte aligned float 4×4 matrix with SIMD-accelerated operations.
    ///
    /// Inherits the full Matrix4<float> API. The `operator*` overloads below
    /// replace the generic implementations with architecture-specific paths:
    /// - **x86/x86-64**: SSE 128-bit intrinsics (`_mm_load_ps`, `_mm_mul_ps`, etc.)
    /// - **ARM**: NEON intrinsics (`vld1q_f32`, `vfmaq_laneq_f32`, etc.)
    /// - **Other**: falls back to the generic scalar implementation.
    ///
    /// @note Both the matrix and the Vector4f operand must be 16-byte aligned.
    class alignas(16) Matrix4f : public Matrix4<float> {
    public:

        // ── Constructors ──────────────────────────────────────────────────────

        /// Default-constructs all elements to zero.
        Matrix4f() : Matrix4<float>() {}

        /// Constructs all elements from a single @p value.
        Matrix4f(float value) : Matrix4<float>(value) {}

        /// Constructs from a 16-element row-major C array.
        Matrix4f(float buffer[16]) : Matrix4<float>(buffer) {}

        /// Constructs from a base Matrix4<float> (e.g. returned by static factories).
        Matrix4f(const Matrix4<float>& m) : Matrix4<float>(m) {}

        // ── SIMD-accelerated operators ────────────────────────────────────────

        /// Matrix–matrix multiplication using SIMD intrinsics.
        inline Matrix4f operator*(const Matrix4f &rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4f toReturn;
                __m128 row1 = _mm_load_ps(&rhs.data[0]);
                __m128 row2 = _mm_load_ps(&rhs.data[4]);
                __m128 row3 = _mm_load_ps(&rhs.data[8]);
                __m128 row4 = _mm_load_ps(&rhs.data[12]);
                for(int i=0; i<4; i++) {
                    __m128 brod1 = _mm_set1_ps(this->data[4*i + 0]);
                    __m128 brod2 = _mm_set1_ps(this->data[4*i + 1]);
                    __m128 brod3 = _mm_set1_ps(this->data[4*i + 2]);
                    __m128 brod4 = _mm_set1_ps(this->data[4*i + 3]);
                    __m128 row = _mm_add_ps(
                                _mm_add_ps(
                                    _mm_mul_ps(brod1, row1),
                                    _mm_mul_ps(brod2, row2)),
                                _mm_add_ps(
                                    _mm_mul_ps(brod3, row3),
                                    _mm_mul_ps(brod4, row4)));
                    _mm_store_ps(&toReturn.data[4*i], row);
                }
                return toReturn;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                Matrix4f toReturn;
                // Load rows of rhs (these are the vectors we accumulate into each result row)
                float32x4_t B0 = vld1q_f32(rhs.data);
                float32x4_t B1 = vld1q_f32(rhs.data+4);
                float32x4_t B2 = vld1q_f32(rhs.data+8);
                float32x4_t B3 = vld1q_f32(rhs.data+12);

                // result_row_i = this[i][0]*rhs_row0 + this[i][1]*rhs_row1 + ...
                float32x4_t A0 = vld1q_f32(this->data);
                float32x4_t C0 = vmovq_n_f32(0);
                C0 = vfmaq_laneq_f32(C0, B0, A0, 0);
                C0 = vfmaq_laneq_f32(C0, B1, A0, 1);
                C0 = vfmaq_laneq_f32(C0, B2, A0, 2);
                C0 = vfmaq_laneq_f32(C0, B3, A0, 3);
                vst1q_f32(toReturn.data, C0);

                float32x4_t A1 = vld1q_f32(this->data+4);
                float32x4_t C1 = vmovq_n_f32(0);
                C1 = vfmaq_laneq_f32(C1, B0, A1, 0);
                C1 = vfmaq_laneq_f32(C1, B1, A1, 1);
                C1 = vfmaq_laneq_f32(C1, B2, A1, 2);
                C1 = vfmaq_laneq_f32(C1, B3, A1, 3);
                vst1q_f32(toReturn.data+4, C1);

                float32x4_t A2 = vld1q_f32(this->data+8);
                float32x4_t C2 = vmovq_n_f32(0);
                C2 = vfmaq_laneq_f32(C2, B0, A2, 0);
                C2 = vfmaq_laneq_f32(C2, B1, A2, 1);
                C2 = vfmaq_laneq_f32(C2, B2, A2, 2);
                C2 = vfmaq_laneq_f32(C2, B3, A2, 3);
                vst1q_f32(toReturn.data+8, C2);

                float32x4_t A3 = vld1q_f32(this->data+12);
                float32x4_t C3 = vmovq_n_f32(0);
                C3 = vfmaq_laneq_f32(C3, B0, A3, 0);
                C3 = vfmaq_laneq_f32(C3, B1, A3, 1);
                C3 = vfmaq_laneq_f32(C3, B2, A3, 2);
                C3 = vfmaq_laneq_f32(C3, B3, A3, 3);
                vst1q_f32(toReturn.data+12, C3);

                return toReturn;
            #else
                return Matrix4f(((Matrix4<float> *)this)->operator*(rhs));
            #endif
        }

        /// Matrix–Vector4f multiplication using SIMD intrinsics.
        /// x86: transpose+broadcast pattern to avoid slow horizontal adds.
        /// ARM: pairwise-add pattern (two vpaddq_f32 passes → [dot(r0,v), dot(r1,v), dot(r2,v), dot(r3,v)]).
        inline Vector4f operator*(const Vector4f &rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 col0 = _mm_load_ps(&this->data[0]);
                __m128 col1 = _mm_load_ps(&this->data[4]);
                __m128 col2 = _mm_load_ps(&this->data[8]);
                __m128 col3 = _mm_load_ps(&this->data[12]);
                __m128 v    = _mm_load_ps(rhs.data);
                // Transpose so col0..col3 become the matrix columns
                _MM_TRANSPOSE4_PS(col0, col1, col2, col3);
                // Broadcast each vector component and accumulate
                __m128 vx = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0,0,0,0));
                __m128 vy = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1,1,1,1));
                __m128 vz = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,2,2,2));
                __m128 vw = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3,3,3,3));
                __m128 result = _mm_add_ps(
                    _mm_add_ps(_mm_mul_ps(vx, col0), _mm_mul_ps(vy, col1)),
                    _mm_add_ps(_mm_mul_ps(vz, col2), _mm_mul_ps(vw, col3))
                );
                Vector4f toReturn;
                _mm_store_ps(toReturn.data, result);
                return toReturn;
            #elif defined(__VECTOR_MATH_ARCH_ARM) && (defined(__arm64__) || defined(__aarch64__))
                // Load each row and the vector, then compute 4 dot products simultaneously:
                // pairwise-add of element-wise products reduces [r0*v, r1*v, r2*v, r3*v]
                // to [dot(r0,v), dot(r1,v), dot(r2,v), dot(r3,v)] in two vpaddq passes.
                float32x4_t v  = vld1q_f32(rhs.data);
                float32x4_t p0 = vmulq_f32(vld1q_f32(&this->data[0]),  v);
                float32x4_t p1 = vmulq_f32(vld1q_f32(&this->data[4]),  v);
                float32x4_t p2 = vmulq_f32(vld1q_f32(&this->data[8]),  v);
                float32x4_t p3 = vmulq_f32(vld1q_f32(&this->data[12]), v);
                // First pass: [p0[0]+p0[1], p0[2]+p0[3], p1[0]+p1[1], p1[2]+p1[3]]
                //             [p2[0]+p2[1], p2[2]+p2[3], p3[0]+p3[1], p3[2]+p3[3]]
                float32x4_t sum01 = vpaddq_f32(p0, p1);
                float32x4_t sum23 = vpaddq_f32(p2, p3);
                // Second pass: [dot(r0,v), dot(r1,v), dot(r2,v), dot(r3,v)]
                float32x4_t result = vpaddq_f32(sum01, sum23);
                Vector4f toReturn;
                vst1q_f32(toReturn.data, result);
                return toReturn;
            #else
                auto toReturn = ((Matrix4<float> *)this)->operator*(rhs);
                return *(Vector4f *)&toReturn;
            #endif
        }

    };
}
