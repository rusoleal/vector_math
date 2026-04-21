#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <gtest/gtest.h>
#include <vector_math/vector_math.hpp>
#include <vector_math/vector2.hpp>

using namespace systems::leal::vector_math;

static constexpr float  FLOAT_EPS  = 1e-5f;
static constexpr double DOUBLE_EPS = 1e-10;

// Helper: element-wise EXPECT_NEAR for Vec-based types
template<typename T, uint32_t N>
void expectNear(const Vec<T, N> &a, const Vec<T, N> &b, double eps) {
    for (uint32_t i = 0; i < N; i++)
        EXPECT_NEAR(a.data[i], b.data[i], eps) << "  at index " << i;
}

// ============================================================
// Vec<T, N>  –  generic template
// ============================================================
TEST(Vec, DefaultConstructorZeroInitializes) {
    Vec<float, 4> v;
    for (int i = 0; i < 4; i++) EXPECT_EQ(v.data[i], 0.f);
}

TEST(Vec, ValueConstructorFillsAllElements) {
    Vec<float, 3> v(7.f);
    for (int i = 0; i < 3; i++) EXPECT_EQ(v.data[i], 7.f);
}

TEST(Vec, EqualityAndInequality) {
    Vec<float, 3> a(1.f), b(1.f), c(2.f);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(Vec, Addition) {
    Vec<float, 3> a(1.f), b(2.f);
    auto r = a + b;
    for (int i = 0; i < 3; i++) EXPECT_EQ(r.data[i], 3.f);
}

TEST(Vec, Subtraction) {
    Vec<float, 3> a(5.f), b(2.f);
    auto r = a - b;
    for (int i = 0; i < 3; i++) EXPECT_EQ(r.data[i], 3.f);
}

TEST(Vec, ScalarMultiply) {
    Vec<float, 3> v(3.f);
    auto r1 = v * 2.f;
    auto r2 = 2.f * v;
    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(r1.data[i], 6.f);
        EXPECT_EQ(r2.data[i], 6.f);
    }
}

TEST(Vec, ScalarDivide) {
    Vec<float, 3> v(6.f);
    auto r = v / 2.f;
    for (int i = 0; i < 3; i++) EXPECT_EQ(r.data[i], 3.f);
}

TEST(Vec, Negation) {
    Vec<float, 3> v(4.f);
    auto r = -v;
    for (int i = 0; i < 3; i++) EXPECT_EQ(r.data[i], -4.f);
}

TEST(Vec, LengthSquared) {
    Vec<float, 3> v;
    v.data[0] = 3.f; v.data[1] = 4.f; v.data[2] = 0.f;
    EXPECT_NEAR(v.lengthSquared(), 25.f, FLOAT_EPS);
}

TEST(Vec, Length) {
    Vec<float, 3> v;
    v.data[0] = 3.f; v.data[1] = 4.f; v.data[2] = 0.f;
    EXPECT_NEAR(v.length(), 5.f, FLOAT_EPS);
}

TEST(Vec, NormalizeUnitAxes) {
    Vec<float, 3> v;
    v.data[0] = 3.f; v.data[1] = 4.f; v.data[2] = 0.f;
    v.normalize();
    EXPECT_NEAR(v.length(), 1.f, FLOAT_EPS);
    EXPECT_NEAR(v.data[0], 0.6f, FLOAT_EPS);
    EXPECT_NEAR(v.data[1], 0.8f, FLOAT_EPS);
}

TEST(Vec, NormalizeZeroVectorIsNoop) {
    Vec<float, 3> v; // all zeros
    v.normalize();
    for (int i = 0; i < 3; i++) EXPECT_EQ(v.data[i], 0.f);
}

TEST(Vec, Normalized) {
    Vec<float, 3> v;
    v.data[0] = 0.f; v.data[1] = 0.f; v.data[2] = 5.f;
    auto n = v.normalized();
    EXPECT_NEAR(n.length(), 1.f, FLOAT_EPS);
    EXPECT_NEAR(n.data[2], 1.f, FLOAT_EPS);
    // original unchanged
    EXPECT_NEAR(v.data[2], 5.f, FLOAT_EPS);
}

// ============================================================
// Vector2
// ============================================================
TEST(Vector2, ConstructorAndSize) {
    EXPECT_EQ(sizeof(Vector2<float>), 8u);
    Vector2<float> v(3.f, 4.f);
    EXPECT_EQ(v.data[0], 3.f);
    EXPECT_EQ(v.data[1], 4.f);
}

TEST(Vector2, DefaultIsZero) {
    Vector2<float> v;
    EXPECT_EQ(v.data[0], 0.f);
    EXPECT_EQ(v.data[1], 0.f);
}

TEST(Vector2, Length) {
    Vector2<float> v(3.f, 4.f);
    EXPECT_NEAR(v.length(), 5.f, FLOAT_EPS);
}

TEST(Vector2fSIMD, ConstructorAlignmentAndArithmetic) {
    EXPECT_EQ(alignof(Vector2f), 16u);
    Vector2<float> ga(1.5f, -2.0f), gb(-0.5f, 5.0f);
    Vector2f a(ga), b(gb);

    expectNear(Vector2<float>(a + b), ga + gb, FLOAT_EPS);
    expectNear(Vector2<float>(a - b), ga - gb, FLOAT_EPS);
    expectNear(Vector2<float>(-a), -ga, FLOAT_EPS);
    expectNear(Vector2<float>(a * 2.5f), ga * 2.5f, FLOAT_EPS);
    expectNear(Vector2<float>(a / 2.0f), ga / 2.0f, FLOAT_EPS);
}

TEST(Vector2fSIMD, DotCrossNormalizeReflectMatchGeneric) {
    Vector2<float> gv(3.0f, 4.0f), gu(0.0f, 1.0f);
    Vector2f v(gv), u(gu);

    EXPECT_NEAR(v.dot(u), gv.dot(gu), FLOAT_EPS);
    EXPECT_NEAR(v.cross(u), gv.cross(gu), FLOAT_EPS);
    expectNear(Vector2<float>(v.normalized()), gv.normalized(), FLOAT_EPS);
    expectNear(Vector2<float>(v.reflected(u)), gv.reflected(gu), FLOAT_EPS);
}

TEST(Vector2dSIMD, ConstructorAlignmentAndArithmetic) {
    EXPECT_EQ(alignof(Vector2d), 16u);
    Vector2<double> ga(1.5, -2.0), gb(-0.5, 5.0);
    Vector2d a(ga), b(gb);

    expectNear(Vector2<double>(a + b), ga + gb, DOUBLE_EPS);
    expectNear(Vector2<double>(a - b), ga - gb, DOUBLE_EPS);
    expectNear(Vector2<double>(-a), -ga, DOUBLE_EPS);
    expectNear(Vector2<double>(a * 2.5), ga * 2.5, DOUBLE_EPS);
    expectNear(Vector2<double>(a / 2.0), ga / 2.0, DOUBLE_EPS);
}

TEST(Vector2dSIMD, DotCrossNormalizeReflectMatchGeneric) {
    Vector2<double> gv(3.0, 4.0), gu(0.0, 1.0);
    Vector2d v(gv), u(gu);

    EXPECT_NEAR(v.dot(u), gv.dot(gu), DOUBLE_EPS);
    EXPECT_NEAR(v.cross(u), gv.cross(gu), DOUBLE_EPS);
    expectNear(Vector2<double>(v.normalized()), gv.normalized(), DOUBLE_EPS);
    expectNear(Vector2<double>(v.reflected(u)), gv.reflected(gu), DOUBLE_EPS);
}

// ============================================================
// Vector3
// ============================================================
TEST(Vector3Float, Constructors) {
    EXPECT_EQ(sizeof(Vector3<float>), 12u);
    EXPECT_EQ(Vector3<float>(), Vector3<float>(0));
    Vector3<float> v(1.f, 2.f, 3.f);
    EXPECT_EQ(v.data[0], 1.f);
    EXPECT_EQ(v.data[1], 2.f);
    EXPECT_EQ(v.data[2], 3.f);
}

TEST(Vector3Float, Arithmetics) {
    EXPECT_NEAR(Vector3<float>(1,0,0).length(), 1.f, FLOAT_EPS);
    EXPECT_NEAR(Vector3<float>(0,1,0).length(), 1.f, FLOAT_EPS);
    EXPECT_NEAR(Vector3<float>(0,0,1).length(), 1.f, FLOAT_EPS);
    EXPECT_EQ(-Vector3<float>(1,2,3), Vector3<float>(-1,-2,-3));
}

TEST(Vector3Float, DotProduct) {
    EXPECT_NEAR(Vector3<float>::dot(Vector3<float>(1,2,3), Vector3<float>(4,5,6)), 32.f, FLOAT_EPS);
    // orthogonal vectors
    EXPECT_NEAR(Vector3<float>::dot(Vector3<float>(1,0,0), Vector3<float>(0,1,0)), 0.f, FLOAT_EPS);
    // instance form
    EXPECT_NEAR(Vector3<float>(1,2,3).dot(Vector3<float>(4,5,6)), 32.f, FLOAT_EPS);
}

TEST(Vector3Float, CrossProduct) {
    // basis vectors
    auto r = Vector3<float>::cross(Vector3<float>(1,0,0), Vector3<float>(0,1,0));
    EXPECT_NEAR(r.data[0], 0.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[1], 0.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[2], 1.f, FLOAT_EPS);
    // anti-commutativity
    auto r2 = Vector3<float>::cross(Vector3<float>(0,1,0), Vector3<float>(1,0,0));
    EXPECT_NEAR(r2.data[2], -1.f, FLOAT_EPS);
    // parallel vectors → zero
    auto r3 = Vector3<float>::cross(Vector3<float>(2,0,0), Vector3<float>(5,0,0));
    EXPECT_NEAR(r3.length(), 0.f, FLOAT_EPS);
}

TEST(Vector3Float, Normalize) {
    Vector3<float> v(0,0,7);
    v.normalize();
    EXPECT_NEAR(v.data[0], 0.f, FLOAT_EPS);
    EXPECT_NEAR(v.data[1], 0.f, FLOAT_EPS);
    EXPECT_NEAR(v.data[2], 1.f, FLOAT_EPS);
}

TEST(Vector3Double, Constructors) {
    EXPECT_EQ(sizeof(Vector3<double>), 24u);
    EXPECT_EQ(Vector3<double>(), Vector3<double>(0));
}

TEST(Vector3Double, Arithmetics) {
    EXPECT_NEAR(Vector3<double>(1,0,0).length(), 1.0, DOUBLE_EPS);
    EXPECT_NEAR(Vector3<double>(0,1,0).length(), 1.0, DOUBLE_EPS);
    EXPECT_NEAR(Vector3<double>(0,0,1).length(), 1.0, DOUBLE_EPS);
    EXPECT_EQ(-Vector3<double>(1,2,3), Vector3<double>(-1,-2,-3));
}

TEST(Vector3Double, DotProduct) {
    EXPECT_NEAR(Vector3<double>::dot(Vector3<double>(1,2,3), Vector3<double>(4,5,6)), 32.0, DOUBLE_EPS);
}

TEST(Vector3Double, CrossProduct) {
    auto r = Vector3<double>::cross(Vector3<double>(1,0,0), Vector3<double>(0,1,0));
    EXPECT_NEAR(r.data[2], 1.0, DOUBLE_EPS);
}

TEST(Vector3fSIMD, ConstructorAlignmentAndPadding) {
    EXPECT_EQ(alignof(Vector3f), 16u);
    EXPECT_EQ(sizeof(Vector3f), 16u);
    Vector3f v(1.0f, 2.0f, 3.0f);
    EXPECT_EQ(v.data[0], 1.0f);
    EXPECT_EQ(v.data[1], 2.0f);
    EXPECT_EQ(v.data[2], 3.0f);
    EXPECT_EQ(v.data[3], 0.0f);
}

TEST(Vector3fSIMD, ArithmeticMatchesGeneric) {
    Vector3<float> ga(1.5f, -2.0f, 3.25f), gb(-0.5f, 5.0f, 1.75f);
    Vector3f a(ga), b(gb);

    expectNear(Vector3<float>(a + b), ga + gb, FLOAT_EPS);
    expectNear(Vector3<float>(a - b), ga - gb, FLOAT_EPS);
    expectNear(Vector3<float>(-a), -ga, FLOAT_EPS);
    expectNear(Vector3<float>(a * 2.5f), ga * 2.5f, FLOAT_EPS);
    expectNear(Vector3<float>(2.5f * a), 2.5f * ga, FLOAT_EPS);
    expectNear(Vector3<float>(a / 2.0f), ga / 2.0f, FLOAT_EPS);
}

TEST(Vector3fSIMD, DotCrossAndLengthMatchGeneric) {
    Vector3<float> ga(1.0f, 2.0f, 3.0f), gb(4.0f, -5.0f, 6.0f);
    Vector3f a(ga), b(gb);

    EXPECT_NEAR(a.dot(b), ga.dot(gb), FLOAT_EPS);
    EXPECT_NEAR(Vector3f::dot(a, b), Vector3<float>::dot(ga, gb), FLOAT_EPS);
    expectNear(Vector3<float>(Vector3f::cross(a, b)), Vector3<float>::cross(ga, gb), FLOAT_EPS);
    EXPECT_NEAR(a.lengthSquared(), ga.lengthSquared(), FLOAT_EPS);
    EXPECT_NEAR(a.length(), ga.length(), FLOAT_EPS);
}

TEST(Vector3fSIMD, NormalizeAndReflectMatchGeneric) {
    Vector3<float> gv(0.0f, 3.0f, 4.0f);
    Vector3f v(gv);
    auto genericNormalized = gv.normalized();
    auto simdNormalized = v.normalized();
    expectNear(Vector3<float>(simdNormalized), genericNormalized, FLOAT_EPS);

    Vector3<float> gn(0.0f, 1.0f, 0.0f);
    Vector3f n(gn);
    auto genericReflected = gv.reflected(gn);
    auto simdReflected = v.reflected(n);
    expectNear(Vector3<float>(simdReflected), genericReflected, FLOAT_EPS);
}

TEST(Vector3fSIMD, ZeroNormalizeIsNoop) {
    Vector3f zero;
    zero.normalize();
    EXPECT_EQ(zero.data[0], 0.0f);
    EXPECT_EQ(zero.data[1], 0.0f);
    EXPECT_EQ(zero.data[2], 0.0f);
    EXPECT_EQ(zero.data[3], 0.0f);
}

TEST(Vector3dSIMD, ConstructorAlignmentAndPadding) {
    EXPECT_EQ(alignof(Vector3d), 32u);
    EXPECT_EQ(sizeof(Vector3d), 32u);
    Vector3d v(1.0, 2.0, 3.0);
    EXPECT_EQ(v.data[0], 1.0);
    EXPECT_EQ(v.data[1], 2.0);
    EXPECT_EQ(v.data[2], 3.0);
    EXPECT_EQ(v.data[3], 0.0);
}

TEST(Vector3dSIMD, ArithmeticMatchesGeneric) {
    Vector3<double> ga(1.5, -2.0, 3.25), gb(-0.5, 5.0, 1.75);
    Vector3d a(ga), b(gb);

    expectNear(Vector3<double>(a + b), ga + gb, DOUBLE_EPS);
    expectNear(Vector3<double>(a - b), ga - gb, DOUBLE_EPS);
    expectNear(Vector3<double>(-a), -ga, DOUBLE_EPS);
    expectNear(Vector3<double>(a * 2.5), ga * 2.5, DOUBLE_EPS);
    expectNear(Vector3<double>(2.5 * a), 2.5 * ga, DOUBLE_EPS);
    expectNear(Vector3<double>(a / 2.0), ga / 2.0, DOUBLE_EPS);
}

TEST(Vector3dSIMD, DotCrossAndLengthMatchGeneric) {
    Vector3<double> ga(1.0, 2.0, 3.0), gb(4.0, -5.0, 6.0);
    Vector3d a(ga), b(gb);

    EXPECT_NEAR(a.dot(b), ga.dot(gb), DOUBLE_EPS);
    EXPECT_NEAR(Vector3d::dot(a, b), Vector3<double>::dot(ga, gb), DOUBLE_EPS);
    expectNear(Vector3<double>(Vector3d::cross(a, b)), Vector3<double>::cross(ga, gb), DOUBLE_EPS);
    EXPECT_NEAR(a.lengthSquared(), ga.lengthSquared(), DOUBLE_EPS);
    EXPECT_NEAR(a.length(), ga.length(), DOUBLE_EPS);
}

TEST(Vector3dSIMD, NormalizeAndReflectMatchGeneric) {
    Vector3<double> gv(0.0, 3.0, 4.0);
    Vector3d v(gv);
    auto genericNormalized = gv.normalized();
    auto simdNormalized = v.normalized();
    expectNear(Vector3<double>(simdNormalized), genericNormalized, DOUBLE_EPS);

    Vector3<double> gn(0.0, 1.0, 0.0);
    Vector3d n(gn);
    auto genericReflected = gv.reflected(gn);
    auto simdReflected = v.reflected(n);
    expectNear(Vector3<double>(simdReflected), genericReflected, DOUBLE_EPS);
}

TEST(Vector3dSIMD, ZeroNormalizeIsNoop) {
    Vector3d zero;
    zero.normalize();
    EXPECT_EQ(zero.data[0], 0.0);
    EXPECT_EQ(zero.data[1], 0.0);
    EXPECT_EQ(zero.data[2], 0.0);
    EXPECT_EQ(zero.data[3], 0.0);
}

// ============================================================
// Vector4 / Vector4f / Vector4d
// ============================================================
TEST(Vector4, Constructor) {
    Vector4<float> v(1,2,3,4);
    EXPECT_EQ(v.data[0], 1.f);
    EXPECT_EQ(v.data[1], 2.f);
    EXPECT_EQ(v.data[2], 3.f);
    EXPECT_EQ(v.data[3], 4.f);
}

TEST(Vector4f, ConstructorAndAlignment) {
    EXPECT_EQ(alignof(Vector4f), 16u);
    Vector4f v(1,2,3,4);
    EXPECT_EQ(v.data[0], 1.f);
    EXPECT_EQ(v.data[1], 2.f);
    EXPECT_EQ(v.data[2], 3.f);
    EXPECT_EQ(v.data[3], 4.f);
}

TEST(Vector4d, ConstructorAndAlignment) {
    EXPECT_EQ(alignof(Vector4d), 32u);
    Vector4d v(1,2,3,4);
    EXPECT_EQ(v.data[0], 1.0);
    EXPECT_EQ(v.data[1], 2.0);
    EXPECT_EQ(v.data[2], 3.0);
    EXPECT_EQ(v.data[3], 4.0);
}

TEST(Vector4fSIMD, ArithmeticMatchesGeneric) {
    Vector4<float> ga(1.5f, -2.0f, 3.25f, 4.0f);
    Vector4<float> gb(-0.5f, 5.0f, 1.75f, -2.0f);
    Vector4f a(1.5f, -2.0f, 3.25f, 4.0f);
    Vector4f b(-0.5f, 5.0f, 1.75f, -2.0f);

    expectNear(a + b, ga + gb, FLOAT_EPS);
    expectNear(a - b, ga - gb, FLOAT_EPS);
    expectNear(-a, -ga, FLOAT_EPS);
    expectNear(a * 2.5f, ga * 2.5f, FLOAT_EPS);
    expectNear(2.5f * a, 2.5f * ga, FLOAT_EPS);
    expectNear(a / 2.0f, ga / 2.0f, FLOAT_EPS);
}

TEST(Vector4fSIMD, DotMatchesGeneric) {
    Vector4<float> ga(1.0f, 2.0f, 3.0f, 4.0f);
    Vector4<float> gb(4.0f, 3.0f, 2.0f, 1.0f);
    Vector4f a(1.0f, 2.0f, 3.0f, 4.0f);
    Vector4f b(4.0f, 3.0f, 2.0f, 1.0f);

    EXPECT_NEAR(a.dot(b), ga.dot(gb), FLOAT_EPS);
    EXPECT_NEAR(Vector4f::dot(a, b), Vector4<float>::dot(ga, gb), FLOAT_EPS);
}

TEST(Vector4fSIMD, LengthNormalizeAndDistanceMatchGeneric) {
    Vector4<float> gv(0.0f, 3.0f, 4.0f, 12.0f);
    Vector4<float> gu(1.0f, -1.0f, 2.0f, 3.0f);
    Vector4f v(0.0f, 3.0f, 4.0f, 12.0f);
    Vector4f u(1.0f, -1.0f, 2.0f, 3.0f);

    EXPECT_NEAR(v.lengthSquared(), gv.lengthSquared(), FLOAT_EPS);
    EXPECT_NEAR(v.distanceToSquared(u), gv.distanceToSquared(gu), FLOAT_EPS);
    expectNear(v.normalized(), gv.normalized(), FLOAT_EPS);
}

TEST(Vector4fSIMD, HelperOpsMatchGeneric) {
    Vector4<float> ga(-1.5f, 2.5f, -3.5f, 4.5f);
    Vector4<float> gb(0.5f, 3.0f, -4.0f, 2.0f);
    Vector4f a(-1.5f, 2.5f, -3.5f, 4.5f);
    Vector4f b(0.5f, 3.0f, -4.0f, 2.0f);

    auto minGeneric = Vector4<float>::min(ga, gb);
    auto maxGeneric = Vector4<float>::max(ga, gb);
    expectNear(Vector4f::min(a, b), minGeneric, FLOAT_EPS);
    expectNear(Vector4f::max(a, b), maxGeneric, FLOAT_EPS);
    expectNear(Vector4f::lerp(a, b, 0.25f), Vector4<float>::lerp(ga, gb, 0.25f), FLOAT_EPS);

    auto absGeneric = ga;
    absGeneric.absolute();
    a.absolute();
    expectNear(a, absGeneric, FLOAT_EPS);

    auto clampScalarGeneric = gb;
    clampScalarGeneric.clamp(-1.0f, 2.5f);
    b.clamp(-1.0f, 2.5f);
    expectNear(b, clampScalarGeneric, FLOAT_EPS);

    Vector4<float> gv(-2.0f, 0.0f, -5.0f, 1.0f);
    Vector4<float> gw(1.0f, 2.0f, -1.0f, 3.0f);
    Vector4<float> clampVecGeneric(-1.5f, 2.5f, -3.5f, 4.5f);
    clampVecGeneric.clamp(gv, gw);
    Vector4f clampVec(-1.5f, 2.5f, -3.5f, 4.5f);
    clampVec.clamp(Vector4f(gv.data[0], gv.data[1], gv.data[2], gv.data[3]),
                   Vector4f(gw.data[0], gw.data[1], gw.data[2], gw.data[3]));
    expectNear(clampVec, clampVecGeneric, FLOAT_EPS);
}

// ============================================================
// Matrix4<float>  –  generic
// ============================================================
TEST(Matrix4Float, Constructors) {
    EXPECT_EQ(sizeof(Matrix4<float>), 64u);
    EXPECT_EQ(Matrix4<float>::identity(), Matrix4<float>::identity());
    EXPECT_NE(Matrix4<float>(), Matrix4<float>(1));
}

TEST(Matrix4Float, IdentityIsNeutralElement) {
    float buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix4<float> M(buf);
    EXPECT_EQ(M * Matrix4<float>::identity(), M);
    EXPECT_EQ(Matrix4<float>::identity() * M, M);
}

TEST(Matrix4Float, Sum) {
    EXPECT_EQ(Matrix4<float>()+Matrix4<float>(1)+Matrix4<float>(1), Matrix4<float>(2));
}

TEST(Matrix4Float, ScalarMultiply) {
    float buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix4<float> M(buf);
    auto R = M * 2.f;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(R.data[i], buf[i] * 2.f, FLOAT_EPS);
}

TEST(Matrix4Float, MatrixMultiply) {
    float m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    float m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};
    EXPECT_EQ(Matrix4<float>(m1)*Matrix4<float>(m2), Matrix4<float>(m3));
}

TEST(Matrix4Float, MatrixVectorMultiply) {
    float m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    EXPECT_EQ(Matrix4<float>(m1)*Vector4<float>(2,6,10,14), Vector4<float>(100,68,228,388));
}

TEST(Matrix4Float, MultiplyAssociativity) {
    float a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float b[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    float c[16] = {1,1,0,2,0,2,1,1,2,0,1,0,1,2,0,1};
    auto AB_C = (Matrix4<float>(a) * Matrix4<float>(b)) * Matrix4<float>(c);
    auto A_BC = Matrix4<float>(a) * (Matrix4<float>(b) * Matrix4<float>(c));
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(AB_C.data[i], A_BC.data[i], FLOAT_EPS);
}

TEST(Matrix4Float, Transpose) {
    EXPECT_EQ(Matrix4<float>::identity().transpose(), Matrix4<float>::identity());
    float buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    auto T = Matrix4<float>(buf).transpose();
    // row-major: element [r][c] at index r*4+c → after transpose, [c][r]
    EXPECT_EQ(T.data[1],  5.f);  // was [0][1]=2, now [1][0]
    EXPECT_EQ(T.data[4],  2.f);
    EXPECT_EQ(T.data[2],  9.f);
    EXPECT_EQ(T.data[8],  3.f);
    // double transpose = original
    EXPECT_EQ(Matrix4<float>(buf).transpose().transpose(), Matrix4<float>(buf));
}

TEST(Matrix4Float, Translate) {
    auto T = Matrix4<float>::translate(Vector3<float>(3,7,11));
    // identity * homogeneous point (1,2,3,1) → (1+3, 2+7, 3+11, 1) = (4,9,14,1)
    auto r = T * Vector4<float>(1,2,3,1);
    EXPECT_NEAR(r.data[0], 4.f,  FLOAT_EPS);
    EXPECT_NEAR(r.data[1], 9.f,  FLOAT_EPS);
    EXPECT_NEAR(r.data[2], 14.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[3], 1.f,  FLOAT_EPS);
}

TEST(Matrix4Float, TranslateByZeroIsIdentity) {
    EXPECT_EQ(Matrix4<float>::translate(Vector3<float>(0,0,0)), Matrix4<float>::identity());
}

TEST(Matrix4Float, Scale) {
    auto S = Matrix4<float>::scale(Vector3<float>(2,3,4));
    auto r = S * Vector4<float>(1,1,1,1);
    EXPECT_NEAR(r.data[0], 2.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[1], 3.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[2], 4.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[3], 1.f, FLOAT_EPS);
}

TEST(Matrix4Float, ScaleByOneIsIdentity) {
    EXPECT_EQ(Matrix4<float>::scale(Vector3<float>(1,1,1)), Matrix4<float>::identity());
}

TEST(Matrix4Float, PerspectiveStructure) {
    const float fov = (float)M_PI / 2.f;
    const float aspect = 1.f, znear = 0.1f, zfar = 100.f;
    auto P = Matrix4<float>::perspective(fov, aspect, znear, zfar);
    float yScale = 1.f / std::tan(fov * 0.5f);
    float q = zfar / (zfar - znear);
    EXPECT_NEAR(P.data[0],  yScale / aspect, FLOAT_EPS);
    EXPECT_NEAR(P.data[5],  yScale,          FLOAT_EPS);
    EXPECT_NEAR(P.data[10], q,               FLOAT_EPS);
    EXPECT_NEAR(P.data[11], -q * znear,      FLOAT_EPS);
    EXPECT_NEAR(P.data[14], 1.f,             FLOAT_EPS);
    EXPECT_NEAR(P.data[15], 0.f,             FLOAT_EPS);
}

TEST(Matrix4Float, Ortho) {
    // ortho(w, h, znear, zfar) delegates to orthoOffCenter(-w/2, w/2, -h/2, h/2, znear, zfar)
    auto O1 = Matrix4<float>::ortho(20.f, 20.f, 0.1f, 100.f);
    auto O2 = Matrix4<float>::orthoOffCenter(-10.f, 10.f, 10.f, -10.f, 0.1f, 100.f);
    EXPECT_EQ(O1, O2);
}

TEST(Matrix4Float, LookAtAlongZAxis) {
    // Camera at origin looking toward +Z
    auto V = Matrix4<float>::lookAt(
        Vector3<float>(0,0,0),
        Vector3<float>(0,0,1),
        Vector3<float>(0,1,0)
    );
    // +Z forward: x-col should be (1,0,0), y-col (0,1,0), z-col (0,0,1)
    EXPECT_NEAR(V.data[0], 1.f, FLOAT_EPS); // xAxis.x
    EXPECT_NEAR(V.data[5], 1.f, FLOAT_EPS); // yAxis.y
    EXPECT_NEAR(V.data[10], 1.f, FLOAT_EPS); // zAxis.z
    // No translation
    EXPECT_NEAR(V.data[3],  0.f, FLOAT_EPS);
    EXPECT_NEAR(V.data[7],  0.f, FLOAT_EPS);
    EXPECT_NEAR(V.data[11], 0.f, FLOAT_EPS);
}

TEST(Matrix4Float, LookAtTranslation) {
    // Camera at (0,0,-5) looking at origin
    auto V = Matrix4<float>::lookAt(
        Vector3<float>(0,0,-5),
        Vector3<float>(0,0,0),
        Vector3<float>(0,1,0)
    );
    EXPECT_NEAR(V.data[11], 5.f, FLOAT_EPS); // -dot(zAxis, eye) = -dot((0,0,1),(0,0,-5)) = 5
}

TEST(Matrix4Float, OrthoOffCenterStructure) {
    auto O = Matrix4<float>::orthoOffCenter(-10.f, 10.f, 10.f, -10.f, 0.1f, 100.f);
    EXPECT_NEAR(O.data[0],  2.f / 20.f,  FLOAT_EPS);   // 2/(right-left)
    EXPECT_NEAR(O.data[5],  2.f / 20.f,  FLOAT_EPS);   // 2/(top-bottom)
    EXPECT_NEAR(O.data[3],  0.f,          FLOAT_EPS);   // (left+right)/(left-right) = 0
    EXPECT_NEAR(O.data[7],  0.f,          FLOAT_EPS);   // (top+bottom)/(bottom-top) = 0
}

TEST(Matrix4Float, RotateQuaternionIdentity) {
    // Identity quaternion (0,0,0,1) should produce identity rotation matrix
    auto R = Matrix4<float>::rotate(Quaternion<float>(0,0,0,1));
    EXPECT_EQ(R, Matrix4<float>::identity());
}

TEST(Matrix4Float, RotateQuaternion180Z) {
    // 180° around Z: quaternion (0,0,1,0)
    auto R = Matrix4<float>::rotate(Quaternion<float>(0,0,1,0));
    // Rotates (1,0,0,1) → (-1,0,0,1)
    auto r = R * Vector4<float>(1,0,0,1);
    EXPECT_NEAR(r.data[0], -1.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[1],  0.f, FLOAT_EPS);
    EXPECT_NEAR(r.data[2],  0.f, FLOAT_EPS);
}

// ============================================================
// Matrix4f  –  SIMD float
// ============================================================
TEST(Matrix4FloatSIMD, Constructors) {
    EXPECT_EQ(alignof(Matrix4f), 16u);
    EXPECT_EQ(Matrix4f::identity(), Matrix4f::identity());
}

TEST(Matrix4FloatSIMD, MatrixMultiply) {
    float m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    float m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};
    EXPECT_EQ(Matrix4f(m1)*Matrix4f(m2), Matrix4f(m3));
}

TEST(Matrix4FloatSIMD, MatrixVectorMultiply) {
    float m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    EXPECT_EQ(Matrix4f(m1)*Vector4f(2,6,10,14), Vector4f(100,68,228,388));
}

TEST(Matrix4FloatSIMD, MatchesGenericImplementation) {
    float a[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    float b[16] = {1,1,0,2,0,2,1,1,2,0,1,0,1,2,0,1};
    auto generic = Matrix4<float>(a) * Matrix4<float>(b);
    auto simd    = Matrix4f(a)       * Matrix4f(b);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], FLOAT_EPS);
}

TEST(Matrix4FloatSIMD, VectorMatchesGeneric) {
    float m[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    Vector4<float> v(1,2,3,4);
    auto generic = Matrix4<float>(m) * v;
    auto simd    = Matrix4f(m)       * Vector4f(v.data[0],v.data[1],v.data[2],v.data[3]);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], FLOAT_EPS);
}

TEST(Matrix4FloatSIMD, IdentityIsNeutralElement) {
    float buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix4f M(buf);
    EXPECT_EQ(M * Matrix4f::identity(), M);
    EXPECT_EQ(Matrix4f::identity() * M, M);
}

// ============================================================
// Matrix4<double>  –  generic
// ============================================================
TEST(Matrix4Double, Constructors) {
    EXPECT_EQ(sizeof(Matrix4<double>), 128u);
    EXPECT_EQ(Matrix4<double>::identity(), Matrix4<double>::identity());
    EXPECT_NE(Matrix4<double>(), Matrix4<double>(1));
}

TEST(Matrix4Double, IdentityIsNeutralElement) {
    double buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix4<double> M(buf);
    EXPECT_EQ(M * Matrix4<double>::identity(), M);
    EXPECT_EQ(Matrix4<double>::identity() * M, M);
}

TEST(Matrix4Double, Sum) {
    EXPECT_EQ(Matrix4<double>()+Matrix4<double>(1)+Matrix4<double>(1), Matrix4<double>(2));
}

TEST(Matrix4Double, MatrixMultiply) {
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    double m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};
    EXPECT_EQ(Matrix4<double>(m1)*Matrix4<double>(m2), Matrix4<double>(m3));
}

TEST(Matrix4Double, MatrixVectorMultiply) {
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    EXPECT_EQ(Matrix4<double>(m1)*Vector4<double>(2,6,10,14), Vector4<double>(100,68,228,388));
}

TEST(Matrix4Double, Transpose) {
    EXPECT_EQ(Matrix4<double>::identity().transpose(), Matrix4<double>::identity());
    double buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    EXPECT_EQ(Matrix4<double>(buf).transpose().transpose(), Matrix4<double>(buf));
}

TEST(Matrix4Double, Translate) {
    auto T = Matrix4<double>::translate(Vector3<double>(3,7,11));
    auto r = T * Vector4<double>(1,2,3,1);
    EXPECT_NEAR(r.data[0], 4.0,  DOUBLE_EPS);
    EXPECT_NEAR(r.data[1], 9.0,  DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], 14.0, DOUBLE_EPS);
}

TEST(Matrix4Double, Scale) {
    auto S = Matrix4<double>::scale(Vector3<double>(2,3,4));
    auto r = S * Vector4<double>(1,1,1,1);
    EXPECT_NEAR(r.data[0], 2.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[1], 3.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], 4.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[3], 1.0, DOUBLE_EPS);
}

// ============================================================
// Matrix4d  –  SIMD double
// ============================================================
TEST(Matrix4DoubleSIMD, Constructors) {
    EXPECT_EQ(alignof(Matrix4d), 32u);
    EXPECT_EQ(Matrix4d::identity(), Matrix4d::identity());
}

TEST(Matrix4DoubleSIMD, MatrixMultiply) {
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    double m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};
    EXPECT_EQ(Matrix4d(m1)*Matrix4d(m2), Matrix4d(m3));
}

TEST(Matrix4DoubleSIMD, MatrixVectorMultiply) {
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    EXPECT_EQ(Matrix4d(m1)*Vector4d(2,6,10,14), Vector4d(100,68,228,388));
}

TEST(Matrix4DoubleSIMD, MatchesGenericImplementation) {
    double a[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    double b[16] = {1,1,0,2,0,2,1,1,2,0,1,0,1,2,0,1};
    auto generic = Matrix4<double>(a) * Matrix4<double>(b);
    auto simd    = Matrix4d(a)        * Matrix4d(b);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS);
}

TEST(Matrix4DoubleSIMD, VectorMatchesGeneric) {
    double m[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    Vector4<double> v(1,2,3,4);
    auto generic = Matrix4<double>(m) * v;
    auto simd    = Matrix4d(m)        * Vector4d(v.data[0],v.data[1],v.data[2],v.data[3]);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS);
}

TEST(Matrix4DoubleSIMD, IdentityIsNeutralElement) {
    double buf[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    Matrix4d M(buf);
    EXPECT_EQ(M * Matrix4d::identity(), M);
    EXPECT_EQ(Matrix4d::identity() * M, M);
}

// ============================================================
// Matrix4<double>  –  determinant / invert
// ============================================================
TEST(Matrix4Double, DeterminantIdentity) {
    EXPECT_NEAR(Matrix4<double>::identity().determinant(), 1.0, DOUBLE_EPS);
}

TEST(Matrix4Double, DeterminantKnownMatrix) {
    // Diagonal matrix with known det = 1*2*3*4 = 24
    double buf[16] = {1,0,0,0, 0,2,0,0, 0,0,3,0, 0,0,0,4};
    EXPECT_NEAR(Matrix4<double>(buf).determinant(), 24.0, DOUBLE_EPS);
}

TEST(Matrix4Double, DeterminantSingular) {
    // Row of zeros → det = 0
    EXPECT_NEAR(Matrix4<double>(0.0).determinant(), 0.0, DOUBLE_EPS);
}

TEST(Matrix4Double, InvertIdentityIsIdentity) {
    auto I = Matrix4<double>::identity();
    EXPECT_TRUE(I.invert());
    EXPECT_EQ(I, Matrix4<double>::identity());
}

TEST(Matrix4Double, InvertedRoundtrip) {
    // M * M^-1 == identity
    double buf[16] = {1,2,0,0, 0,1,0,0, 0,0,1,3, 0,0,0,1};
    Matrix4<double> M(buf);
    auto Minv = M.inverted();
    auto result = M * Minv;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(result.data[i*4+j], (i == j) ? 1.0 : 0.0, 1e-9);
}

TEST(Matrix4Double, InvertSingularReturnsFalse) {
    Matrix4<double> M(0.0); // all zeros — singular
    EXPECT_FALSE(M.invert());
    // matrix must remain unchanged
    EXPECT_EQ(M, Matrix4<double>(0.0));
}

// ============================================================
// Matrix4<float>  –  determinant / invert
// ============================================================
TEST(Matrix4Float, DeterminantIdentity) {
    EXPECT_NEAR(Matrix4<float>::identity().determinant(), 1.f, FLOAT_EPS);
}

TEST(Matrix4Float, InvertedRoundtrip) {
    float buf[16] = {1,2,0,0, 0,1,0,0, 0,0,1,3, 0,0,0,1};
    Matrix4<float> M(buf);
    auto result = M * M.inverted();
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(result.data[i*4+j], (i == j) ? 1.f : 0.f, FLOAT_EPS);
}

TEST(Matrix4Float, InvertSingularReturnsFalse) {
    Matrix4<float> M(0.f);
    EXPECT_FALSE(M.invert());
    EXPECT_EQ(M, Matrix4<float>(0.f));
}

// ============================================================
// Matrix4d::lookAt  –  custom optimized override
// ============================================================
TEST(Matrix4DoubleSIMD, LookAtMatchesGeneric) {
    Vector3<double> eye(1,2,3), target(4,5,6), up(0,1,0);
    auto generic = Matrix4<double>::lookAt(eye, target, up);
    auto simd    = Matrix4d::lookAt(eye, target, up);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], 1e-10) << "at index " << i;
}

TEST(Matrix4DoubleSIMD, LookAtAlongZAxis) {
    auto V = Matrix4d::lookAt(
        Vector3<double>(0,0,0),
        Vector3<double>(0,0,1),
        Vector3<double>(0,1,0)
    );
    EXPECT_NEAR(V.data[0],  1.0, DOUBLE_EPS);
    EXPECT_NEAR(V.data[5],  1.0, DOUBLE_EPS);
    EXPECT_NEAR(V.data[10], 1.0, DOUBLE_EPS);
    EXPECT_NEAR(V.data[3],  0.0, DOUBLE_EPS);
    EXPECT_NEAR(V.data[7],  0.0, DOUBLE_EPS);
    EXPECT_NEAR(V.data[11], 0.0, DOUBLE_EPS);
}

TEST(Matrix4DoubleSIMD, LookAtTranslation) {
    auto V = Matrix4d::lookAt(
        Vector3<double>(0,0,-5),
        Vector3<double>(0,0,0),
        Vector3<double>(0,1,0)
    );
    EXPECT_NEAR(V.data[11], 5.0, DOUBLE_EPS);
}

TEST(Matrix4DoubleSIMD, LookAtMultiplyAssociativity) {
    // (M1*M2)*v == M1*(M2*v)
    Vector3<double> eye(3,2,1), target(0,0,0), up(0,1,0);
    Matrix4d V = Matrix4d::lookAt(eye, target, up);
    double a[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    Matrix4d A(a);
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    Matrix4d B(m1);
    auto AB_V  = (A * B) * V;
    auto A_BV  = A * (B * V);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(AB_V.data[i], A_BV.data[i], 1e-9) << "at index " << i;
}

// ============================================================
// Vector4d  –  SIMD double arithmetic
// ============================================================
TEST(Vector4dSIMD, Addition) {
    Vector4d a(1,2,3,4), b(5,6,7,8);
    auto r = a + b;
    EXPECT_NEAR(r.data[0], 6.0,  DOUBLE_EPS);
    EXPECT_NEAR(r.data[1], 8.0,  DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], 10.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[3], 12.0, DOUBLE_EPS);
}

TEST(Vector4dSIMD, Subtraction) {
    Vector4d a(5,6,7,8), b(1,2,3,4);
    auto r = a - b;
    EXPECT_NEAR(r.data[0], 4.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[1], 4.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], 4.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[3], 4.0, DOUBLE_EPS);
}

TEST(Vector4dSIMD, Negation) {
    Vector4d v(1,-2,3,-4);
    auto r = -v;
    EXPECT_NEAR(r.data[0], -1.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[1],  2.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], -3.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[3],  4.0, DOUBLE_EPS);
}

TEST(Vector4dSIMD, ScalarMultiply) {
    Vector4d v(1,2,3,4);
    auto r1 = v * 2.0;
    auto r2 = 2.0 * v;
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(r1.data[i], (i+1)*2.0, DOUBLE_EPS);
        EXPECT_NEAR(r2.data[i], (i+1)*2.0, DOUBLE_EPS);
    }
}

TEST(Vector4dSIMD, ScalarDivide) {
    Vector4d v(2,4,6,8);
    auto r = v / 2.0;
    EXPECT_NEAR(r.data[0], 1.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[1], 2.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[2], 3.0, DOUBLE_EPS);
    EXPECT_NEAR(r.data[3], 4.0, DOUBLE_EPS);
}

TEST(Vector4dSIMD, Dot) {
    Vector4d a(1,2,3,4), b(4,3,2,1);
    EXPECT_NEAR(a.dot(b), 20.0, DOUBLE_EPS);
    EXPECT_NEAR(Vector4d::dot(a, b), 20.0, DOUBLE_EPS);
}

TEST(Vector4dSIMD, MatchesGenericAdd) {
    Vector4<double> ga(1.5, 2.5, 3.5, 4.5), gb(0.1, 0.2, 0.3, 0.4);
    auto generic = ga + gb;
    auto simd    = Vector4d(1.5,2.5,3.5,4.5) + Vector4d(0.1,0.2,0.3,0.4);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS);
}

TEST(Vector4dSIMD, LengthNormalizeAndDistanceMatchGeneric) {
    Vector4<double> gv(0.0, 3.0, 4.0, 12.0);
    Vector4<double> gu(1.0, -1.0, 2.0, 3.0);
    Vector4d v(0.0, 3.0, 4.0, 12.0);
    Vector4d u(1.0, -1.0, 2.0, 3.0);

    EXPECT_NEAR(v.lengthSquared(), gv.lengthSquared(), DOUBLE_EPS);
    EXPECT_NEAR(v.distanceToSquared(u), gv.distanceToSquared(gu), DOUBLE_EPS);
    expectNear(v.normalized(), gv.normalized(), DOUBLE_EPS);
}

TEST(Vector4dSIMD, HelperOpsMatchGeneric) {
    Vector4<double> ga(-1.5, 2.5, -3.5, 4.5);
    Vector4<double> gb(0.5, 3.0, -4.0, 2.0);
    Vector4d a(-1.5, 2.5, -3.5, 4.5);
    Vector4d b(0.5, 3.0, -4.0, 2.0);

    auto minGeneric = Vector4<double>::min(ga, gb);
    auto maxGeneric = Vector4<double>::max(ga, gb);
    expectNear(Vector4d::min(a, b), minGeneric, DOUBLE_EPS);
    expectNear(Vector4d::max(a, b), maxGeneric, DOUBLE_EPS);
    expectNear(Vector4d::lerp(a, b, 0.25), Vector4<double>::lerp(ga, gb, 0.25), DOUBLE_EPS);

    auto absGeneric = ga;
    absGeneric.absolute();
    a.absolute();
    expectNear(a, absGeneric, DOUBLE_EPS);

    auto clampScalarGeneric = gb;
    clampScalarGeneric.clamp(-1.0, 2.5);
    b.clamp(-1.0, 2.5);
    expectNear(b, clampScalarGeneric, DOUBLE_EPS);

    Vector4<double> gv(-2.0, 0.0, -5.0, 1.0);
    Vector4<double> gw(1.0, 2.0, -1.0, 3.0);
    Vector4<double> clampVecGeneric(-1.5, 2.5, -3.5, 4.5);
    clampVecGeneric.clamp(gv, gw);
    Vector4d clampVec(-1.5, 2.5, -3.5, 4.5);
    clampVec.clamp(Vector4d(gv.data[0], gv.data[1], gv.data[2], gv.data[3]),
                   Vector4d(gw.data[0], gw.data[1], gw.data[2], gw.data[3]));
    expectNear(clampVec, clampVecGeneric, DOUBLE_EPS);
}

// ============================================================
// Matrix4d  –  SIMD add / subtract / negate / scalar-multiply
// ============================================================
TEST(Matrix4DoubleSIMD, AddMatchesGeneric) {
    double a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    double b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    auto generic = Matrix4<double>(a) + Matrix4<double>(b);
    auto simd    = Matrix4d(a) + Matrix4d(b);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS) << "at index " << i;
}

TEST(Matrix4DoubleSIMD, SubtractMatchesGeneric) {
    double a[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};
    double b[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    auto generic = Matrix4<double>(a) - Matrix4<double>(b);
    auto simd    = Matrix4d(a) - Matrix4d(b);
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS) << "at index " << i;
}

TEST(Matrix4DoubleSIMD, NegateMatchesGeneric) {
    double a[16] = {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16};
    auto generic = -(Matrix4<double>(a));
    auto simd    = -(Matrix4d(a));
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS) << "at index " << i;
}

TEST(Matrix4DoubleSIMD, ScalarMultiplyMatchesGeneric) {
    double a[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    auto generic = Matrix4<double>(a) * 3.0;
    auto simd    = Matrix4d(a) * 3.0;
    for (int i = 0; i < 16; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], DOUBLE_EPS) << "at index " << i;
}

TEST(Matrix4DoubleSIMD, ComposeMatchesGeneric) {
    Vector3<double> t(3.0, 7.0, 11.0);
    Vector3<double> axis(0.0, 1.0, 0.0);
    Quaternion<double> r = Quaternion<double>::axisAngle(axis, M_PI * 0.5);
    Vector3<double> s(2.0, 3.0, 4.0);

    expectNear(Matrix4d::compose(t, r, s), Matrix4<double>::compose(t, r, s), DOUBLE_EPS);
}

// ============================================================
// Matrix4f  –  SIMD ARM vector multiply (validates on all platforms)
// ============================================================
TEST(Matrix4FloatSIMD, VectorMultiplyMatchesGeneric) {
    float m[16] = {2,0,1,3,1,2,3,0,0,3,2,1,3,1,0,2};
    Vector4<float> v(1,2,3,4);
    auto generic = Matrix4<float>(m) * v;
    auto simd    = Matrix4f(m) * Vector4f(v.data[0],v.data[1],v.data[2],v.data[3]);
    for (int i = 0; i < 4; i++)
        EXPECT_NEAR(simd.data[i], generic.data[i], FLOAT_EPS) << "at index " << i;
}

TEST(Matrix4FloatSIMD, AddSubtractNegateAndScaleMatchGeneric) {
    float a[16] = {1,-2,3,-4,5,-6,7,-8,9,-10,11,-12,13,-14,15,-16};
    float b[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};

    expectNear(Matrix4f(a) + Matrix4f(b), Matrix4<float>(a) + Matrix4<float>(b), FLOAT_EPS);
    expectNear(Matrix4f(a) - Matrix4f(b), Matrix4<float>(a) - Matrix4<float>(b), FLOAT_EPS);
    expectNear(-Matrix4f(a), -(Matrix4<float>(a)), FLOAT_EPS);
    expectNear(Matrix4f(a) * 3.0f, Matrix4<float>(a) * 3.0f, FLOAT_EPS);
}

TEST(Matrix4FloatSIMD, LookAtMatchesGeneric) {
    Vector3<float> eye(1.0f, 2.0f, 3.0f), target(4.0f, 5.0f, 6.0f), up(0.0f, 1.0f, 0.0f);
    expectNear(Matrix4f::lookAt(eye, target, up), Matrix4<float>::lookAt(eye, target, up), FLOAT_EPS);
}

TEST(Matrix4FloatSIMD, LookAtTranslationAlongForwardAxis) {
    auto view = Matrix4f::lookAt(
        Vector3<float>(0.0f, 0.0f, -5.0f),
        Vector3<float>(0.0f, 0.0f, 0.0f),
        Vector3<float>(0.0f, 1.0f, 0.0f)
    );
    EXPECT_NEAR(view.data[11], 5.0f, FLOAT_EPS);
    EXPECT_NEAR(view.data[15], 1.0f, FLOAT_EPS);
}

TEST(Matrix4FloatSIMD, ComposeMatchesGeneric) {
    Vector3<float> t(3.0f, 7.0f, 11.0f);
    Vector3<float> axis(0.0f, 1.0f, 0.0f);
    Quaternion<float> r = Quaternion<float>::axisAngle(axis, float(M_PI) * 0.5f);
    Vector3<float> s(2.0f, 3.0f, 4.0f);

    expectNear(Matrix4f::compose(t, r, s), Matrix4<float>::compose(t, r, s), FLOAT_EPS);
}

// ============================================================
// Quaternion
// ============================================================
TEST(Quaternion, Constructor) {
    Quaternion<float> q(1,2,3,4);
    EXPECT_EQ(q.data[0], 1.f);
    EXPECT_EQ(q.data[1], 2.f);
    EXPECT_EQ(q.data[2], 3.f);
    EXPECT_EQ(q.data[3], 4.f);
}

TEST(Quaternion, IdentityProducesIdentityRotation) {
    EXPECT_EQ(Matrix4<float>::rotate(Quaternion<float>(0,0,0,1)), Matrix4<float>::identity());
}

TEST(Quaternion, RotationPreservesLength) {
    // Any unit quaternion rotation should preserve vector length
    // 90° around Y: quaternion (0, sin(45°), 0, cos(45°))
    float s = std::sqrt(2.f) / 2.f;
    auto R = Matrix4<float>::rotate(Quaternion<float>(0, s, 0, s));
    auto v = Vector4<float>(1, 0, 0, 0);
    auto r = R * v;
    float len = std::sqrt(r.data[0]*r.data[0] + r.data[1]*r.data[1] + r.data[2]*r.data[2]);
    EXPECT_NEAR(len, 1.f, FLOAT_EPS);
}

TEST(QuaternionfSIMD, ConstructorAndAlignment) {
    EXPECT_EQ(alignof(Quaternionf), 16u);
    Quaternionf q(1.0f, 2.0f, 3.0f, 4.0f);
    EXPECT_EQ(q.data[0], 1.0f);
    EXPECT_EQ(q.data[1], 2.0f);
    EXPECT_EQ(q.data[2], 3.0f);
    EXPECT_EQ(q.data[3], 4.0f);
}

TEST(QuaternionfSIMD, ArithmeticMatchesGeneric) {
    Quaternion<float> ga(1.5f, -2.0f, 3.25f, 4.0f), gb(-0.5f, 5.0f, 1.75f, -2.0f);
    Quaternionf a(ga), b(gb);

    expectNear(a + b, ga + gb, FLOAT_EPS);
    expectNear(a - b, ga - gb, FLOAT_EPS);
    expectNear(-a, -ga, FLOAT_EPS);
    expectNear(a * 2.5f, ga * 2.5f, FLOAT_EPS);
    expectNear(2.5f * a, 2.5f * ga, FLOAT_EPS);
}

TEST(QuaternionfSIMD, HamiltonProductMatchesGeneric) {
    Quaternion<float> ga(0.2f, 0.3f, 0.4f, 0.5f), gb(-0.1f, 0.6f, -0.2f, 0.7f);
    Quaternionf a(ga), b(gb);
    expectNear(a * b, ga * gb, FLOAT_EPS);
}

TEST(QuaternionfSIMD, DotConjugateAndInverseMatchGeneric) {
    Quaternion<float> ga(0.2f, 0.3f, 0.4f, 0.5f), gb(-0.1f, 0.6f, -0.2f, 0.7f);
    Quaternionf a(ga), b(gb);

    EXPECT_NEAR(a.dot(b), ga.dot(gb), FLOAT_EPS);
    EXPECT_NEAR(Quaternionf::dot(a, b), Quaternion<float>::dot(ga, gb), FLOAT_EPS);
    expectNear(a.conjugated(), ga.conjugated(), FLOAT_EPS);
    expectNear(a.inverted(), ga.inverted(), FLOAT_EPS);
}

TEST(QuaternionfSIMD, RotatedVectorMatchesGeneric) {
    float s = std::sqrt(2.0f) / 2.0f;
    Quaternion<float> gq(0.0f, s, 0.0f, s);
    Quaternionf q(gq);
    Vector3<float> gv(1.0f, 0.0f, 0.0f);
    Vector3f v(gv);

    expectNear(Vector3<float>(q.rotated(v)), gq.rotated(gv), FLOAT_EPS);
}

TEST(QuaternionfSIMD, IdentityAndAxisAngle) {
    auto qIdentity = Quaternionf::identity();
    EXPECT_NEAR(qIdentity.data[3], 1.0f, FLOAT_EPS);

    Vector3f axis(0.0f, 1.0f, 0.0f);
    auto q = Quaternionf::axisAngle(axis, float(M_PI) * 0.5f);
    Vector3f v(1.0f, 0.0f, 0.0f);
    auto rotated = q.rotated(v);
    EXPECT_NEAR(rotated.data[0], 0.0f, 1e-4f);
    EXPECT_NEAR(rotated.data[2], -1.0f, 1e-4f);
}

TEST(QuaterniondSIMD, ConstructorAndAlignment) {
    EXPECT_EQ(alignof(Quaterniond), 32u);
    Quaterniond q(1.0, 2.0, 3.0, 4.0);
    EXPECT_EQ(q.data[0], 1.0);
    EXPECT_EQ(q.data[1], 2.0);
    EXPECT_EQ(q.data[2], 3.0);
    EXPECT_EQ(q.data[3], 4.0);
}

TEST(QuaterniondSIMD, ArithmeticMatchesGeneric) {
    Quaternion<double> ga(1.5, -2.0, 3.25, 4.0), gb(-0.5, 5.0, 1.75, -2.0);
    Quaterniond a(ga), b(gb);

    expectNear(a + b, ga + gb, DOUBLE_EPS);
    expectNear(a - b, ga - gb, DOUBLE_EPS);
    expectNear(-a, -ga, DOUBLE_EPS);
    expectNear(a * 2.5, ga * 2.5, DOUBLE_EPS);
    expectNear(2.5 * a, 2.5 * ga, DOUBLE_EPS);
}

TEST(QuaterniondSIMD, HamiltonProductMatchesGeneric) {
    Quaternion<double> ga(0.2, 0.3, 0.4, 0.5), gb(-0.1, 0.6, -0.2, 0.7);
    Quaterniond a(ga), b(gb);
    expectNear(a * b, ga * gb, DOUBLE_EPS);
}

TEST(QuaterniondSIMD, DotConjugateAndInverseMatchGeneric) {
    Quaternion<double> ga(0.2, 0.3, 0.4, 0.5), gb(-0.1, 0.6, -0.2, 0.7);
    Quaterniond a(ga), b(gb);

    EXPECT_NEAR(a.dot(b), ga.dot(gb), DOUBLE_EPS);
    EXPECT_NEAR(Quaterniond::dot(a, b), Quaternion<double>::dot(ga, gb), DOUBLE_EPS);
    expectNear(a.conjugated(), ga.conjugated(), DOUBLE_EPS);
    expectNear(a.inverted(), ga.inverted(), DOUBLE_EPS);
}

TEST(QuaterniondSIMD, RotatedVectorMatchesGeneric) {
    double s = std::sqrt(2.0) / 2.0;
    Quaternion<double> gq(0.0, s, 0.0, s);
    Quaterniond q(gq);
    Vector3<double> gv(1.0, 0.0, 0.0);
    Vector3d v(gv);

    expectNear(Vector3<double>(q.rotated(v)), gq.rotated(gv), DOUBLE_EPS);
}

TEST(QuaterniondSIMD, IdentityAndAxisAngle) {
    auto qIdentity = Quaterniond::identity();
    EXPECT_NEAR(qIdentity.data[3], 1.0, DOUBLE_EPS);

    Vector3d axis(0.0, 1.0, 0.0);
    auto q = Quaterniond::axisAngle(axis, M_PI * 0.5);
    Vector3d v(1.0, 0.0, 0.0);
    auto rotated = q.rotated(v);
    EXPECT_NEAR(rotated.data[0], 0.0, 1e-10);
    EXPECT_NEAR(rotated.data[2], -1.0, 1e-10);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
