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

        /// Matrix–Vector4f multiplication — forwards to the scalar base class.
        /// The compiler auto-vectorizes the generic 4-dot product better than hand-written SSE hadd.
        inline Vector4f operator*(const Vector4f &rhs) const {
            auto toReturn = ((Matrix4<float> *)this)->operator*(rhs);
            return *(Vector4f *)&toReturn;
        }
        /// Component-wise addition using SIMD intrinsics.
        inline Matrix4f operator+(const Matrix4f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4f result;
                _mm_store_ps(&result.data[0],  _mm_add_ps(_mm_load_ps(&this->data[0]),  _mm_load_ps(&rhs.data[0])));
                _mm_store_ps(&result.data[4],  _mm_add_ps(_mm_load_ps(&this->data[4]),  _mm_load_ps(&rhs.data[4])));
                _mm_store_ps(&result.data[8],  _mm_add_ps(_mm_load_ps(&this->data[8]),  _mm_load_ps(&rhs.data[8])));
                _mm_store_ps(&result.data[12], _mm_add_ps(_mm_load_ps(&this->data[12]), _mm_load_ps(&rhs.data[12])));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Matrix4f result;
                vst1q_f32(&result.data[0],  vaddq_f32(vld1q_f32(&this->data[0]),  vld1q_f32(&rhs.data[0])));
                vst1q_f32(&result.data[4],  vaddq_f32(vld1q_f32(&this->data[4]),  vld1q_f32(&rhs.data[4])));
                vst1q_f32(&result.data[8],  vaddq_f32(vld1q_f32(&this->data[8]),  vld1q_f32(&rhs.data[8])));
                vst1q_f32(&result.data[12], vaddq_f32(vld1q_f32(&this->data[12]), vld1q_f32(&rhs.data[12])));
                return result;
            #else
                auto toReturn = ((Matrix4<float> *)this)->operator+(rhs);
                return *(Matrix4f *)&toReturn;
            #endif
        }

        /// Component-wise subtraction using SIMD intrinsics.
        inline Matrix4f operator-(const Matrix4f& rhs) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4f result;
                _mm_store_ps(&result.data[0],  _mm_sub_ps(_mm_load_ps(&this->data[0]),  _mm_load_ps(&rhs.data[0])));
                _mm_store_ps(&result.data[4],  _mm_sub_ps(_mm_load_ps(&this->data[4]),  _mm_load_ps(&rhs.data[4])));
                _mm_store_ps(&result.data[8],  _mm_sub_ps(_mm_load_ps(&this->data[8]),  _mm_load_ps(&rhs.data[8])));
                _mm_store_ps(&result.data[12], _mm_sub_ps(_mm_load_ps(&this->data[12]), _mm_load_ps(&rhs.data[12])));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Matrix4f result;
                vst1q_f32(&result.data[0],  vsubq_f32(vld1q_f32(&this->data[0]),  vld1q_f32(&rhs.data[0])));
                vst1q_f32(&result.data[4],  vsubq_f32(vld1q_f32(&this->data[4]),  vld1q_f32(&rhs.data[4])));
                vst1q_f32(&result.data[8],  vsubq_f32(vld1q_f32(&this->data[8]),  vld1q_f32(&rhs.data[8])));
                vst1q_f32(&result.data[12], vsubq_f32(vld1q_f32(&this->data[12]), vld1q_f32(&rhs.data[12])));
                return result;
            #else
                auto toReturn = ((Matrix4<float> *)this)->operator-(rhs);
                return *(Matrix4f *)&toReturn;
            #endif
        }

        /// Negation using SIMD intrinsics.
        inline Matrix4f operator-() const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4f result;
                __m128 sign_mask = _mm_set1_ps(-0.0f);
                _mm_store_ps(&result.data[0],  _mm_xor_ps(_mm_load_ps(&this->data[0]),  sign_mask));
                _mm_store_ps(&result.data[4],  _mm_xor_ps(_mm_load_ps(&this->data[4]),  sign_mask));
                _mm_store_ps(&result.data[8],  _mm_xor_ps(_mm_load_ps(&this->data[8]),  sign_mask));
                _mm_store_ps(&result.data[12], _mm_xor_ps(_mm_load_ps(&this->data[12]), sign_mask));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Matrix4f result;
                vst1q_f32(&result.data[0],  vnegq_f32(vld1q_f32(&this->data[0])));
                vst1q_f32(&result.data[4],  vnegq_f32(vld1q_f32(&this->data[4])));
                vst1q_f32(&result.data[8],  vnegq_f32(vld1q_f32(&this->data[8])));
                vst1q_f32(&result.data[12], vnegq_f32(vld1q_f32(&this->data[12])));
                return result;
            #else
                auto toReturn = ((Matrix4<float> *)this)->operator-();
                return *(Matrix4f *)&toReturn;
            #endif
        }

        /// Scalar multiplication using SIMD intrinsics.
        inline Matrix4f operator*(float scalar) const {
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                Matrix4f result;
                __m128 s = _mm_set1_ps(scalar);
                _mm_store_ps(&result.data[0],  _mm_mul_ps(_mm_load_ps(&this->data[0]),  s));
                _mm_store_ps(&result.data[4],  _mm_mul_ps(_mm_load_ps(&this->data[4]),  s));
                _mm_store_ps(&result.data[8],  _mm_mul_ps(_mm_load_ps(&this->data[8]),  s));
                _mm_store_ps(&result.data[12], _mm_mul_ps(_mm_load_ps(&this->data[12]), s));
                return result;
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                Matrix4f result;
                vst1q_f32(&result.data[0],  vmulq_n_f32(vld1q_f32(&this->data[0]), scalar));
                vst1q_f32(&result.data[4],  vmulq_n_f32(vld1q_f32(&this->data[4]), scalar));
                vst1q_f32(&result.data[8],  vmulq_n_f32(vld1q_f32(&this->data[8]), scalar));
                vst1q_f32(&result.data[12], vmulq_n_f32(vld1q_f32(&this->data[12]), scalar));
                return result;
            #else
                auto toReturn = ((Matrix4<float> *)this)->operator*(scalar);
                return *(Matrix4f *)&toReturn;
            #endif
        }

        /// Builds a right-handed view matrix — optimized override of Matrix4<float>::lookAt.
        static inline Matrix4f lookAt(const Vector3<float> &eye, const Vector3<float> &target, const Vector3<float> &up)
        {
            float zx = target.data[0] - eye.data[0];
            float zy = target.data[1] - eye.data[1];
            float zz = target.data[2] - eye.data[2];
            float zInvLen = 1.0f / std::sqrt(zx*zx + zy*zy + zz*zz);
            zx *= zInvLen; zy *= zInvLen; zz *= zInvLen;

            float xx = up.data[1]*zz - up.data[2]*zy;
            float xy = up.data[2]*zx - up.data[0]*zz;
            float xz = up.data[0]*zy - up.data[1]*zx;
            float xInvLen = 1.0f / std::sqrt(xx*xx + xy*xy + xz*xz);
            xx *= xInvLen; xy *= xInvLen; xz *= xInvLen;

            float yx = zy*xz - zz*xy;
            float yy = zz*xx - zx*xz;
            float yz = zx*xy - zy*xx;

            float buf[16] = {
                xx,   xy,   xz,   -(xx*eye.data[0] + xy*eye.data[1] + xz*eye.data[2]),
                yx,   yy,   yz,   -(yx*eye.data[0] + yy*eye.data[1] + yz*eye.data[2]),
                zx,   zy,   zz,   -(zx*eye.data[0] + zy*eye.data[1] + zz*eye.data[2]),
                0.0f, 0.0f, 0.0f, 1.0f
            };
            return Matrix4f(buf);
        }

        /// Builds a rotation matrix from a unit quaternion directly as Matrix4f.
        static inline Matrix4f rotate(const Quaternion<float>& rotation)
        {
            Matrix4f result;
            float xx = rotation.data[0] * rotation.data[0];
            float yy = rotation.data[1] * rotation.data[1];
            float zz = rotation.data[2] * rotation.data[2];
            float xy = rotation.data[0] * rotation.data[1];
            float zw = rotation.data[2] * rotation.data[3];
            float zx = rotation.data[2] * rotation.data[0];
            float yw = rotation.data[1] * rotation.data[3];
            float yz = rotation.data[1] * rotation.data[2];
            float xw = rotation.data[0] * rotation.data[3];

            result.data[0]  = 1.0f - (2.0f * (yy + zz));
            result.data[4]  = 2.0f * (xy + zw);
            result.data[8]  = 2.0f * (zx - yw);
            result.data[1]  = 2.0f * (xy - zw);
            result.data[5]  = 1.0f - (2.0f * (zz + xx));
            result.data[9]  = 2.0f * (yz + xw);
            result.data[2]  = 2.0f * (zx + yw);
            result.data[6]  = 2.0f * (yz - xw);
            result.data[10] = 1.0f - (2.0f * (yy + xx));
            result.data[3]  = result.data[7] = result.data[11] = 0.0f;
            result.data[15] = 1.0f;
            return result;
        }

        /// Builds a TRS matrix with SIMD column scaling for the rotation basis.
        static inline Matrix4f compose(const Vector3<float>& translation, const Quaternion<float>& rotation, const Vector3<float>& scaleVec)
        {
            Matrix4f result = Matrix4f::rotate(rotation);
            #ifdef __VECTOR_MATH_ARCH_X86_X64
                __m128 scale = _mm_set_ps(1.0f, scaleVec.data[2], scaleVec.data[1], scaleVec.data[0]);
                _mm_store_ps(&result.data[0], _mm_mul_ps(_mm_load_ps(&result.data[0]), scale));
                _mm_store_ps(&result.data[4], _mm_mul_ps(_mm_load_ps(&result.data[4]), scale));
                _mm_store_ps(&result.data[8], _mm_mul_ps(_mm_load_ps(&result.data[8]), scale));
            #elif defined(__VECTOR_MATH_ARCH_ARM)
                float32x4_t scale = { scaleVec.data[0], scaleVec.data[1], scaleVec.data[2], 1.0f };
                vst1q_f32(&result.data[0], vmulq_f32(vld1q_f32(&result.data[0]), scale));
                vst1q_f32(&result.data[4], vmulq_f32(vld1q_f32(&result.data[4]), scale));
                vst1q_f32(&result.data[8], vmulq_f32(vld1q_f32(&result.data[8]), scale));
            #else
                result.data[0] *= scaleVec.data[0]; result.data[1] *= scaleVec.data[1]; result.data[2]  *= scaleVec.data[2];
                result.data[4] *= scaleVec.data[0]; result.data[5] *= scaleVec.data[1]; result.data[6]  *= scaleVec.data[2];
                result.data[8] *= scaleVec.data[0]; result.data[9] *= scaleVec.data[1]; result.data[10] *= scaleVec.data[2];
            #endif
            result.data[3]  = translation.data[0];
            result.data[7]  = translation.data[1];
            result.data[11] = translation.data[2];
            return result;
        }

    };
}
