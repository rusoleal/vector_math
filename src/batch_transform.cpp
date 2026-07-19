#include <vector_math/batch_transform.hpp>
#include <vector_math/common.hpp>

namespace systems::leal::vector_math
{

    namespace detail
    {

        void transformVectorsScalar(const Matrix4f &m, const Vector4f *in, Vector4f *out, std::size_t count)
        {
            for (std::size_t i = 0; i < count; ++i)
                out[i] = m * in[i];
        }

        void transformVectorsScalar(const Matrix4d &m, const Vector4d *in, Vector4d *out, std::size_t count)
        {
            for (std::size_t i = 0; i < count; ++i)
                out[i] = m * in[i];
        }

#if defined(__VECTOR_MATH_ARCH_X86_X64)
        // Defined in batch_transform_avx2.cpp, compiled with -mavx2/-mfma.
        void transformVectorsAvx2(const Matrix4f &m, const Vector4f *in, Vector4f *out, std::size_t count);
        void transformVectorsAvx2(const Matrix4d &m, const Vector4d *in, Vector4d *out, std::size_t count);
#endif

    } // namespace detail

    void transformVectors(const Matrix4f &m, const Vector4f *in, Vector4f *out, std::size_t count)
    {
#if defined(__VECTOR_MATH_ARCH_X86_X64)
        static const bool hasAvx2 = cpuSupportsAVX2();
        if (hasAvx2)
        {
            detail::transformVectorsAvx2(m, in, out, count);
            return;
        }
#endif
        detail::transformVectorsScalar(m, in, out, count);
    }

    void transformVectors(const Matrix4d &m, const Vector4d *in, Vector4d *out, std::size_t count)
    {
#if defined(__VECTOR_MATH_ARCH_X86_X64)
        static const bool hasAvx2 = cpuSupportsAVX2();
        if (hasAvx2)
        {
            detail::transformVectorsAvx2(m, in, out, count);
            return;
        }
#endif
        detail::transformVectorsScalar(m, in, out, count);
    }

} // namespace systems::leal::vector_math
