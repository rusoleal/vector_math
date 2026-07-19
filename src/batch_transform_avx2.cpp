// Compiled with -mavx2 -mfma (GCC/Clang) or /arch:AVX2 (MSVC) — see
// CMakeLists.txt. This TU is only built on x86/x64, and transformVectors()
// only calls these functions after confirming AVX2 support at runtime.
//
// The kernels below use an AoS -> SoA -> AoS transpose pattern: 4 Vector4d
// (or 8 Vector4f) are transposed so that one AVX register holds one coordinate
// from 4/8 different vectors. This lets the matrix columns be broadcast once
// and reused across the whole batch, and exposes enough FMA parallelism to
// beat a scalar auto-vectorized loop.
#include <vector_math/batch_transform.hpp>

#include <immintrin.h>

namespace systems::leal::vector_math
{

    namespace detail
    {

        void transformVectorsAvx2(const Matrix4f &m, const Vector4f *in, Vector4f *out, std::size_t count)
        {
            const float *md = m.data;

            // Broadcast matrix columns once.
            __m256 m00 = _mm256_set1_ps(md[0]);
            __m256 m01 = _mm256_set1_ps(md[1]);
            __m256 m02 = _mm256_set1_ps(md[2]);
            __m256 m03 = _mm256_set1_ps(md[3]);
            __m256 m10 = _mm256_set1_ps(md[4]);
            __m256 m11 = _mm256_set1_ps(md[5]);
            __m256 m12 = _mm256_set1_ps(md[6]);
            __m256 m13 = _mm256_set1_ps(md[7]);
            __m256 m20 = _mm256_set1_ps(md[8]);
            __m256 m21 = _mm256_set1_ps(md[9]);
            __m256 m22 = _mm256_set1_ps(md[10]);
            __m256 m23 = _mm256_set1_ps(md[11]);
            __m256 m30 = _mm256_set1_ps(md[12]);
            __m256 m31 = _mm256_set1_ps(md[13]);
            __m256 m32 = _mm256_set1_ps(md[14]);
            __m256 m33 = _mm256_set1_ps(md[15]);

            std::size_t i = 0;
            for (; i + 8 <= count; i += 8)
            {
                // Load 8 Vector4f as 4 AVX registers (2 vectors each).
                __m256 a = _mm256_loadu_ps(reinterpret_cast<const float *>(&in[i + 0]));
                __m256 b = _mm256_loadu_ps(reinterpret_cast<const float *>(&in[i + 2]));
                __m256 c = _mm256_loadu_ps(reinterpret_cast<const float *>(&in[i + 4]));
                __m256 d = _mm256_loadu_ps(reinterpret_cast<const float *>(&in[i + 6]));

                // Transpose AoS -> SoA.
                __m256 t0 = _mm256_unpacklo_ps(a, c); // [x0,x4,y0,y4, x1,x5,y1,y5]
                __m256 t1 = _mm256_unpackhi_ps(a, c); // [z0,z4,w0,w4, z1,z5,w1,w5]
                __m256 t2 = _mm256_unpacklo_ps(b, d); // [x2,x6,y2,y6, x3,x7,y3,y7]
                __m256 t3 = _mm256_unpackhi_ps(b, d); // [z2,z6,w2,w6, z3,z7,w3,w7]

                __m256 xTmp = _mm256_unpacklo_ps(t0, t2); // [x0,x2,x4,x6, x1,x3,x5,x7]
                __m256 yTmp = _mm256_unpackhi_ps(t0, t2); // [y0,y2,y4,y6, y1,y3,y5,y7]
                __m256 zTmp = _mm256_unpacklo_ps(t1, t3); // [z0,z2,z4,z6, z1,z3,z5,z7]
                __m256 wTmp = _mm256_unpackhi_ps(t1, t3); // [w0,w2,w4,w6, w1,w3,w5,w7]

                const __m256i idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
                __m256 xxxx = _mm256_permutevar8x32_ps(xTmp, idx); // [x0,x1,x2,x3,x4,x5,x6,x7]
                __m256 yyyy = _mm256_permutevar8x32_ps(yTmp, idx); // [y0,y1,y2,y3,y4,y5,y6,y7]
                __m256 zzzz = _mm256_permutevar8x32_ps(zTmp, idx); // [z0,z1,z2,z3,z4,z5,z6,z7]
                __m256 wwww = _mm256_permutevar8x32_ps(wTmp, idx); // [w0,w1,w2,w3,w4,w5,w6,w7]

                // Compute 8 outputs in parallel.
                __m256 outX, outY, outZ, outW;
#ifdef __FMA__
                outX = _mm256_fmadd_ps(m00, xxxx, _mm256_fmadd_ps(m01, yyyy, _mm256_fmadd_ps(m02, zzzz, _mm256_mul_ps(m03, wwww))));
                outY = _mm256_fmadd_ps(m10, xxxx, _mm256_fmadd_ps(m11, yyyy, _mm256_fmadd_ps(m12, zzzz, _mm256_mul_ps(m13, wwww))));
                outZ = _mm256_fmadd_ps(m20, xxxx, _mm256_fmadd_ps(m21, yyyy, _mm256_fmadd_ps(m22, zzzz, _mm256_mul_ps(m23, wwww))));
                outW = _mm256_fmadd_ps(m30, xxxx, _mm256_fmadd_ps(m31, yyyy, _mm256_fmadd_ps(m32, zzzz, _mm256_mul_ps(m33, wwww))));
#else
                outX = _mm256_add_ps(_mm256_mul_ps(m00, xxxx), _mm256_add_ps(_mm256_mul_ps(m01, yyyy), _mm256_add_ps(_mm256_mul_ps(m02, zzzz), _mm256_mul_ps(m03, wwww))));
                outY = _mm256_add_ps(_mm256_mul_ps(m10, xxxx), _mm256_add_ps(_mm256_mul_ps(m11, yyyy), _mm256_add_ps(_mm256_mul_ps(m12, zzzz), _mm256_mul_ps(m13, wwww))));
                outZ = _mm256_add_ps(_mm256_mul_ps(m20, xxxx), _mm256_add_ps(_mm256_mul_ps(m21, yyyy), _mm256_add_ps(_mm256_mul_ps(m22, zzzz), _mm256_mul_ps(m23, wwww))));
                outW = _mm256_add_ps(_mm256_mul_ps(m30, xxxx), _mm256_add_ps(_mm256_mul_ps(m31, yyyy), _mm256_add_ps(_mm256_mul_ps(m32, zzzz), _mm256_mul_ps(m33, wwww))));
#endif

                // Transpose SoA -> AoS.
                __m256 r0 = _mm256_unpacklo_ps(outX, outY); // [x0,y0,x1,y1, x4,y4,x5,y5]
                __m256 r1 = _mm256_unpackhi_ps(outX, outY); // [x2,y2,x3,y3, x6,y6,x7,y7]
                __m256 r2 = _mm256_unpacklo_ps(outZ, outW); // [z0,w0,z1,w1, z4,w4,z5,w5]
                __m256 r3 = _mm256_unpackhi_ps(outZ, outW); // [z2,w2,z3,w3, z6,w6,z7,w7]

                __m256 p0 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(1, 0, 1, 0)); // [v0, v4]
                __m256 p1 = _mm256_shuffle_ps(r0, r2, _MM_SHUFFLE(3, 2, 3, 2)); // [v1, v5]
                __m256 p2 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(1, 0, 1, 0)); // [v2, v6]
                __m256 p3 = _mm256_shuffle_ps(r1, r3, _MM_SHUFFLE(3, 2, 3, 2)); // [v3, v7]

                __m256 o0 = _mm256_permute2f128_ps(p0, p1, 0x20); // [v0, v1]
                __m256 o1 = _mm256_permute2f128_ps(p2, p3, 0x20); // [v2, v3]
                __m256 o2 = _mm256_permute2f128_ps(p0, p1, 0x31); // [v4, v5]
                __m256 o3 = _mm256_permute2f128_ps(p2, p3, 0x31); // [v6, v7]

                _mm256_storeu_ps(reinterpret_cast<float *>(&out[i + 0]), o0);
                _mm256_storeu_ps(reinterpret_cast<float *>(&out[i + 2]), o1);
                _mm256_storeu_ps(reinterpret_cast<float *>(&out[i + 4]), o2);
                _mm256_storeu_ps(reinterpret_cast<float *>(&out[i + 6]), o3);
            }

            // Remainder: one vector at a time using the existing SIMD operators.
            for (; i < count; ++i)
                out[i] = m * in[i];
        }

        void transformVectorsAvx2(const Matrix4d &m, const Vector4d *in, Vector4d *out, std::size_t count)
        {
            const double *md = m.data;

            __m256d m00 = _mm256_set1_pd(md[0]);
            __m256d m01 = _mm256_set1_pd(md[1]);
            __m256d m02 = _mm256_set1_pd(md[2]);
            __m256d m03 = _mm256_set1_pd(md[3]);
            __m256d m10 = _mm256_set1_pd(md[4]);
            __m256d m11 = _mm256_set1_pd(md[5]);
            __m256d m12 = _mm256_set1_pd(md[6]);
            __m256d m13 = _mm256_set1_pd(md[7]);
            __m256d m20 = _mm256_set1_pd(md[8]);
            __m256d m21 = _mm256_set1_pd(md[9]);
            __m256d m22 = _mm256_set1_pd(md[10]);
            __m256d m23 = _mm256_set1_pd(md[11]);
            __m256d m30 = _mm256_set1_pd(md[12]);
            __m256d m31 = _mm256_set1_pd(md[13]);
            __m256d m32 = _mm256_set1_pd(md[14]);
            __m256d m33 = _mm256_set1_pd(md[15]);

            std::size_t i = 0;
            for (; i + 4 <= count; i += 4)
            {
                // Load 4 Vector4d.
                __m256d a = _mm256_loadu_pd(reinterpret_cast<const double *>(&in[i + 0]));
                __m256d b = _mm256_loadu_pd(reinterpret_cast<const double *>(&in[i + 1]));
                __m256d c = _mm256_loadu_pd(reinterpret_cast<const double *>(&in[i + 2]));
                __m256d d = _mm256_loadu_pd(reinterpret_cast<const double *>(&in[i + 3]));

                // Transpose AoS -> SoA.
                __m256d t0 = _mm256_unpacklo_pd(a, b); // [x0,x1,z0,z1]
                __m256d t1 = _mm256_unpackhi_pd(a, b); // [y0,y1,w0,w1]
                __m256d t2 = _mm256_unpacklo_pd(c, d); // [x2,x3,z2,z3]
                __m256d t3 = _mm256_unpackhi_pd(c, d); // [y2,y3,w2,w3]

                __m256d xxxx = _mm256_permute2f128_pd(t0, t2, 0x20); // [x0,x1,x2,x3]
                __m256d yyyy = _mm256_permute2f128_pd(t1, t3, 0x20); // [y0,y1,y2,y3]
                __m256d zzzz = _mm256_permute2f128_pd(t0, t2, 0x31); // [z0,z1,z2,z3]
                __m256d wwww = _mm256_permute2f128_pd(t1, t3, 0x31); // [w0,w1,w2,w3]

                // Compute 4 outputs in parallel.
                __m256d outX, outY, outZ, outW;
#ifdef __FMA__
                outX = _mm256_fmadd_pd(m00, xxxx, _mm256_fmadd_pd(m01, yyyy, _mm256_fmadd_pd(m02, zzzz, _mm256_mul_pd(m03, wwww))));
                outY = _mm256_fmadd_pd(m10, xxxx, _mm256_fmadd_pd(m11, yyyy, _mm256_fmadd_pd(m12, zzzz, _mm256_mul_pd(m13, wwww))));
                outZ = _mm256_fmadd_pd(m20, xxxx, _mm256_fmadd_pd(m21, yyyy, _mm256_fmadd_pd(m22, zzzz, _mm256_mul_pd(m23, wwww))));
                outW = _mm256_fmadd_pd(m30, xxxx, _mm256_fmadd_pd(m31, yyyy, _mm256_fmadd_pd(m32, zzzz, _mm256_mul_pd(m33, wwww))));
#else
                outX = _mm256_add_pd(_mm256_mul_pd(m00, xxxx), _mm256_add_pd(_mm256_mul_pd(m01, yyyy), _mm256_add_pd(_mm256_mul_pd(m02, zzzz), _mm256_mul_pd(m03, wwww))));
                outY = _mm256_add_pd(_mm256_mul_pd(m10, xxxx), _mm256_add_pd(_mm256_mul_pd(m11, yyyy), _mm256_add_pd(_mm256_mul_pd(m12, zzzz), _mm256_mul_pd(m13, wwww))));
                outZ = _mm256_add_pd(_mm256_mul_pd(m20, xxxx), _mm256_add_pd(_mm256_mul_pd(m21, yyyy), _mm256_add_pd(_mm256_mul_pd(m22, zzzz), _mm256_mul_pd(m23, wwww))));
                outW = _mm256_add_pd(_mm256_mul_pd(m30, xxxx), _mm256_add_pd(_mm256_mul_pd(m31, yyyy), _mm256_add_pd(_mm256_mul_pd(m32, zzzz), _mm256_mul_pd(m33, wwww))));
#endif

                // Transpose SoA -> AoS.
                __m256d r0 = _mm256_unpacklo_pd(outX, outY); // [x0,y0,x2,y2]
                __m256d r1 = _mm256_unpackhi_pd(outX, outY); // [x1,y1,x3,y3]
                __m256d r2 = _mm256_unpacklo_pd(outZ, outW); // [z0,w0,z2,w2]
                __m256d r3 = _mm256_unpackhi_pd(outZ, outW); // [z1,w1,z3,w3]

                __m256d o0 = _mm256_permute2f128_pd(r0, r2, 0x20); // [v0]
                __m256d o1 = _mm256_permute2f128_pd(r1, r3, 0x20); // [v1]
                __m256d o2 = _mm256_permute2f128_pd(r0, r2, 0x31); // [v2]
                __m256d o3 = _mm256_permute2f128_pd(r1, r3, 0x31); // [v3]

                _mm256_storeu_pd(reinterpret_cast<double *>(&out[i + 0]), o0);
                _mm256_storeu_pd(reinterpret_cast<double *>(&out[i + 1]), o1);
                _mm256_storeu_pd(reinterpret_cast<double *>(&out[i + 2]), o2);
                _mm256_storeu_pd(reinterpret_cast<double *>(&out[i + 3]), o3);
            }

            for (; i < count; ++i)
                out[i] = m * in[i];
        }

    } // namespace detail

} // namespace systems::leal::vector_math
