#include <benchmark/benchmark.h>
#include <vector_math/matrix4.hpp>
#include <vector_math/matrix4f.hpp>
#include <vector_math/matrix4d.hpp>
#include <vector_math/quaternion.hpp>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/scalar_constants.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/geometric.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

using namespace systems::leal::vector_math;

////////////////////
// Creation       //
////////////////////
static void BM_Matrix4Creation(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>());
    }
}
BENCHMARK(BM_Matrix4Creation);

static void BM_Matrix4fCreation(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4f());
    }
}
BENCHMARK(BM_Matrix4fCreation);

static void BM_Matrix4dCreation(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4d());
    }
}
BENCHMARK(BM_Matrix4dCreation);

////////////////////
// Identity       //
////////////////////
static void BM_Matrix4Identity(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::identity());
    }
}
BENCHMARK(BM_Matrix4Identity);

static void BM_Matrix4fIdentity(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4f::identity());
    }
}
BENCHMARK(BM_Matrix4fIdentity);

static void BM_Matrix4dIdentity(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4d::identity());
    }
}
BENCHMARK(BM_Matrix4dIdentity);

////////////////////
// Multiply       //
////////////////////
static void BM_Matrix4Multiply(benchmark::State& state) {
    auto m1 = Matrix4<float>::identity();
    auto m2 = Matrix4<float>(1);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * m2);
    }
}
BENCHMARK(BM_Matrix4Multiply);

static void BM_Matrix4fMultiply(benchmark::State& state) {
    Matrix4f m1 = Matrix4f::identity();
    Matrix4f m2 = Matrix4f(1);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * m2);
    }
}
BENCHMARK(BM_Matrix4fMultiply);

static void BM_Matrix4dMultiply(benchmark::State& state) {
    Matrix4d m1 = Matrix4d::identity();
    Matrix4d m2 = Matrix4d(1);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * m2);
    }
}
BENCHMARK(BM_Matrix4dMultiply);

static void BM_GLM_MatrixMultiply(benchmark::State& state) {
    glm::mat4 m1 = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
    glm::mat4 m2 = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0));
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * m2);
    }
}
BENCHMARK(BM_GLM_MatrixMultiply);

////////////////////////
// Matrix x Vector    //
////////////////////////
static void BM_Matrix4ByVector(benchmark::State& state) {
    auto m1 = Matrix4<float>::identity();
    auto v1 = Vector4<float>(1,2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * v1);
    }
}
BENCHMARK(BM_Matrix4ByVector);

static void BM_Matrix4fByVector(benchmark::State& state) {
    Matrix4f m1 = Matrix4f::identity();
    Vector4f v1 = Vector4f(1,2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * v1);
    }
}
BENCHMARK(BM_Matrix4fByVector);

static void BM_Matrix4dByVector(benchmark::State& state) {
    Matrix4d m1 = Matrix4d::identity();
    Vector4d v1 = Vector4d(1,2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * v1);
    }
}
BENCHMARK(BM_Matrix4dByVector);

static void BM_GLM_MatrixByVector(benchmark::State& state) {
    glm::mat4 m1 = glm::perspective(glm::pi<float>() * 0.25f, 4.0f / 3.0f, 0.1f, 100.f);
    glm::vec4 v1 = glm::vec4(1,2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(m1 * v1);
    }
}
BENCHMARK(BM_GLM_MatrixByVector);

////////////////////
// LookAt         //
////////////////////
static void BM_Matrix4LookAt(benchmark::State& state) {
    auto eye    = Vector3<float>(0,0,0);
    auto target = Vector3<float>(0,0,1000);
    auto up     = Vector3<float>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::lookAt(eye, target, up));
    }
}
BENCHMARK(BM_Matrix4LookAt);

static void BM_Matrix4fLookAt(benchmark::State& state) {
    auto eye    = Vector3<float>(0,0,0);
    auto target = Vector3<float>(0,0,1000);
    auto up     = Vector3<float>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4f::lookAt(eye, target, up));
    }
}
BENCHMARK(BM_Matrix4fLookAt);

static void BM_Matrix4dLookAt(benchmark::State& state) {
    auto eye    = Vector3<double>(0,0,0);
    auto target = Vector3<double>(0,0,1000);
    auto up     = Vector3<double>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4d::lookAt(eye, target, up));
    }
}
BENCHMARK(BM_Matrix4dLookAt);

static void BM_GLM_LookAt(benchmark::State& state) {
    glm::vec3 eye(0,0,0), target(0,0,1000), up(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::lookAt(eye, target, up));
    }
}
BENCHMARK(BM_GLM_LookAt);

////////////////////
// Perspective    //
////////////////////
static void BM_Matrix4Perspective(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f));
    }
}
BENCHMARK(BM_Matrix4Perspective);

static void BM_GLM_Perspective(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f));
    }
}
BENCHMARK(BM_GLM_Perspective);

////////////////////
// Ortho          //
////////////////////
static void BM_Matrix4Ortho(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::ortho(800.0f, 600.0f, 0.1f, 100.0f));
    }
}
BENCHMARK(BM_Matrix4Ortho);

static void BM_GLM_Ortho(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::ortho(-400.0f, 400.0f, -300.0f, 300.0f, 0.1f, 100.0f));
    }
}
BENCHMARK(BM_GLM_Ortho);

////////////////////
// Translate      //
////////////////////
static void BM_Matrix4Translate(benchmark::State& state) {
    auto t = Vector3<float>(1,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::translate(t));
    }
}
BENCHMARK(BM_Matrix4Translate);

static void BM_GLM_Translate(benchmark::State& state) {
    glm::vec3 t(1,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::translate(glm::mat4(1.0f), t));
    }
}
BENCHMARK(BM_GLM_Translate);

////////////////////
// Scale          //
////////////////////
static void BM_Matrix4Scale(benchmark::State& state) {
    auto s = Vector3<float>(2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::scale(s));
    }
}
BENCHMARK(BM_Matrix4Scale);

static void BM_GLM_Scale(benchmark::State& state) {
    glm::vec3 s(2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::scale(glm::mat4(1.0f), s));
    }
}
BENCHMARK(BM_GLM_Scale);

////////////////////
// RotateX        //
////////////////////
static void BM_Matrix4RotateX(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::rotateX(glm::pi<float>() * 0.25f));
    }
}
BENCHMARK(BM_Matrix4RotateX);

static void BM_GLM_RotateX(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.25f, glm::vec3(1,0,0)));
    }
}
BENCHMARK(BM_GLM_RotateX);

////////////////////
// RotateY        //
////////////////////
static void BM_Matrix4RotateY(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::rotateY(glm::pi<float>() * 0.25f));
    }
}
BENCHMARK(BM_Matrix4RotateY);

static void BM_GLM_RotateY(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.25f, glm::vec3(0,1,0)));
    }
}
BENCHMARK(BM_GLM_RotateY);

////////////////////
// RotateZ        //
////////////////////
static void BM_Matrix4RotateZ(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::rotateZ(glm::pi<float>() * 0.25f));
    }
}
BENCHMARK(BM_Matrix4RotateZ);

static void BM_GLM_RotateZ(benchmark::State& state) {
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::rotate(glm::mat4(1.0f), glm::pi<float>() * 0.25f, glm::vec3(0,0,1)));
    }
}
BENCHMARK(BM_GLM_RotateZ);

////////////////////
// RotateQuat     //
////////////////////
static void BM_Matrix4RotateQuat(benchmark::State& state) {
    auto q = Quaternion<float>::axisAngle(Vector3<float>(0,1,0), glm::pi<float>() * 0.25f);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::rotate(q));
    }
}
BENCHMARK(BM_Matrix4RotateQuat);

static void BM_GLM_RotateQuat(benchmark::State& state) {
    glm::quat q = glm::angleAxis(glm::pi<float>() * 0.25f, glm::vec3(0,1,0));
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::toMat4(q));
    }
}
BENCHMARK(BM_GLM_RotateQuat);

////////////////////
// Compose        //
////////////////////
static void BM_Matrix4Compose(benchmark::State& state) {
    auto t = Vector3<float>(1,2,3);
    auto q = Quaternion<float>::axisAngle(Vector3<float>(0,1,0), glm::pi<float>() * 0.25f);
    auto s = Vector3<float>(2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(Matrix4<float>::compose(t, q, s));
    }
}
BENCHMARK(BM_Matrix4Compose);

static void BM_GLM_Compose(benchmark::State& state) {
    glm::vec3 t(1,2,3);
    glm::quat q = glm::angleAxis(glm::pi<float>() * 0.25f, glm::vec3(0,1,0));
    glm::vec3 s(2,3,4);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::translate(glm::mat4(1.0f), t) * glm::toMat4(q) * glm::scale(glm::mat4(1.0f), s));
    }
}
BENCHMARK(BM_GLM_Compose);

////////////////////
// Determinant    //
////////////////////
static void BM_Matrix4Determinant(benchmark::State& state) {
    auto m = Matrix4<float>::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f);
    for (auto _ : state){
        benchmark::DoNotOptimize(m.determinant());
    }
}
BENCHMARK(BM_Matrix4Determinant);

static void BM_GLM_Determinant(benchmark::State& state) {
    glm::mat4 m = glm::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::determinant(m));
    }
}
BENCHMARK(BM_GLM_Determinant);

////////////////////
// Inverse        //
////////////////////
static void BM_Matrix4Inverted(benchmark::State& state) {
    auto m = Matrix4<float>::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f);
    for (auto _ : state){
        benchmark::DoNotOptimize(m.inverted());
    }
}
BENCHMARK(BM_Matrix4Inverted);

static void BM_GLM_Inverse(benchmark::State& state) {
    glm::mat4 m = glm::perspective(glm::pi<float>() * 0.25f, 4.0f/3.0f, 0.1f, 100.0f);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::inverse(m));
    }
}
BENCHMARK(BM_GLM_Inverse);

////////////////////
// Transform3     //
////////////////////
static void BM_Matrix4Transform3(benchmark::State& state) {
    auto m = Matrix4<float>::identity();
    auto v = Vector3<float>(1,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(m.transform3(v));
    }
}
BENCHMARK(BM_Matrix4Transform3);

////////////////////
// Rotate3        //
////////////////////
static void BM_Matrix4Rotate3(benchmark::State& state) {
    auto m = Matrix4<float>::identity();
    auto v = Vector3<float>(1,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(m.rotate3(v));
    }
}
BENCHMARK(BM_Matrix4Rotate3);

////////////////////////////
// Quaternion AxisAngle   //
////////////////////////////
static void BM_QuaternionAxisAngle(benchmark::State& state) {
    auto axis = Vector3<float>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(Quaternion<float>::axisAngle(axis, glm::pi<float>() * 0.25f));
    }
}
BENCHMARK(BM_QuaternionAxisAngle);

static void BM_GLM_QuaternionAxisAngle(benchmark::State& state) {
    glm::vec3 axis(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::angleAxis(glm::pi<float>() * 0.25f, axis));
    }
}
BENCHMARK(BM_GLM_QuaternionAxisAngle);

////////////////////////////
// Quaternion Multiply    //
////////////////////////////
static void BM_QuaternionMultiply(benchmark::State& state) {
    auto q1 = Quaternion<float>::axisAngle(Vector3<float>(0,1,0), glm::pi<float>() * 0.25f);
    auto q2 = Quaternion<float>::axisAngle(Vector3<float>(1,0,0), glm::pi<float>() * 0.5f);
    for (auto _ : state){
        benchmark::DoNotOptimize(q1 * q2);
    }
}
BENCHMARK(BM_QuaternionMultiply);

static void BM_GLM_QuaternionMultiply(benchmark::State& state) {
    glm::quat q1 = glm::angleAxis(glm::pi<float>() * 0.25f, glm::vec3(0,1,0));
    glm::quat q2 = glm::angleAxis(glm::pi<float>() * 0.5f,  glm::vec3(1,0,0));
    for (auto _ : state){
        benchmark::DoNotOptimize(q1 * q2);
    }
}
BENCHMARK(BM_GLM_QuaternionMultiply);

////////////////////////////
// Quaternion Slerp       //
////////////////////////////
static void BM_QuaternionSlerp(benchmark::State& state) {
    auto q1 = Quaternion<float>::axisAngle(Vector3<float>(0,1,0), glm::pi<float>() * 0.25f);
    auto q2 = Quaternion<float>::axisAngle(Vector3<float>(1,0,0), glm::pi<float>() * 0.5f);
    for (auto _ : state){
        benchmark::DoNotOptimize(Quaternion<float>::slerp(q1, q2, 0.5f));
    }
}
BENCHMARK(BM_QuaternionSlerp);

static void BM_GLM_QuaternionSlerp(benchmark::State& state) {
    glm::quat q1 = glm::angleAxis(glm::pi<float>() * 0.25f, glm::vec3(0,1,0));
    glm::quat q2 = glm::angleAxis(glm::pi<float>() * 0.5f,  glm::vec3(1,0,0));
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::slerp(q1, q2, 0.5f));
    }
}
BENCHMARK(BM_GLM_QuaternionSlerp);

////////////////////////////
// Quaternion Rotated     //
////////////////////////////
static void BM_QuaternionRotated(benchmark::State& state) {
    auto q = Quaternion<float>::axisAngle(Vector3<float>(0,1,0), glm::pi<float>() * 0.25f);
    auto v = Vector3<float>(1,0,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(q.rotated(v));
    }
}
BENCHMARK(BM_QuaternionRotated);

static void BM_GLM_QuaternionRotated(benchmark::State& state) {
    glm::quat q = glm::angleAxis(glm::pi<float>() * 0.25f, glm::vec3(0,1,0));
    glm::vec3 v(1,0,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::rotate(q, v));
    }
}
BENCHMARK(BM_GLM_QuaternionRotated);

////////////////////
// Cross          //
////////////////////
static void BM_CrossVector(benchmark::State& state) {
    auto v1 = Vector3<float>(1,2,3);
    auto v2 = Vector3<float>(0,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(Vector3<float>::cross(v1,v2));
    }
}
BENCHMARK(BM_CrossVector);

static void BM_GLM_CrossVector(benchmark::State& state) {
    glm::vec3 v1(1,2,3), v2(0,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::cross(v1, v2));
    }
}
BENCHMARK(BM_GLM_CrossVector);

////////////////////
// Dot            //
////////////////////
static void BM_DotVector(benchmark::State& state) {
    auto v1 = Vector3<float>(1,2,3);
    auto v2 = Vector3<float>(0,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(Vector3<float>::dot(v1,v2));
    }
}
BENCHMARK(BM_DotVector);

static void BM_GLM_DotVector(benchmark::State& state) {
    glm::vec3 v1(1,2,3), v2(0,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::dot(v1, v2));
    }
}
BENCHMARK(BM_GLM_DotVector);

////////////////////////////
// Vector4d arithmetic    //
////////////////////////////
static void BM_Vector4dAdd(benchmark::State& state) {
    Vector4<double> a(1,2,3,4), b(5,6,7,8);
    for (auto _ : state)
        benchmark::DoNotOptimize(a + b);
}
BENCHMARK(BM_Vector4dAdd);

static void BM_Vector4dSIMDAdd(benchmark::State& state) {
    Vector4d a(1,2,3,4), b(5,6,7,8);
    for (auto _ : state)
        benchmark::DoNotOptimize(a + b);
}
BENCHMARK(BM_Vector4dSIMDAdd);

static void BM_Vector4dScalarMultiply(benchmark::State& state) {
    Vector4<double> v(1,2,3,4);
    for (auto _ : state)
        benchmark::DoNotOptimize(v * 2.5);
}
BENCHMARK(BM_Vector4dScalarMultiply);

static void BM_Vector4dSIMDScalarMultiply(benchmark::State& state) {
    Vector4d v(1,2,3,4);
    for (auto _ : state)
        benchmark::DoNotOptimize(v * 2.5);
}
BENCHMARK(BM_Vector4dSIMDScalarMultiply);

static void BM_Vector4dDot(benchmark::State& state) {
    Vector4<double> a(1,2,3,4), b(4,3,2,1);
    for (auto _ : state)
        benchmark::DoNotOptimize(Vector4<double>::dot(a, b));
}
BENCHMARK(BM_Vector4dDot);

static void BM_Vector4dSIMDDot(benchmark::State& state) {
    Vector4d a(1,2,3,4), b(4,3,2,1);
    for (auto _ : state)
        benchmark::DoNotOptimize(Vector4d::dot(a, b));
}
BENCHMARK(BM_Vector4dSIMDDot);

static void BM_GLM_Vec4dAdd(benchmark::State& state) {
    glm::dvec4 a(1,2,3,4), b(5,6,7,8);
    for (auto _ : state)
        benchmark::DoNotOptimize(a + b);
}
BENCHMARK(BM_GLM_Vec4dAdd);

static void BM_GLM_Vec4dDot(benchmark::State& state) {
    glm::dvec4 a(1,2,3,4), b(4,3,2,1);
    for (auto _ : state)
        benchmark::DoNotOptimize(glm::dot(a, b));
}
BENCHMARK(BM_GLM_Vec4dDot);

/////////////////////////////////
// Matrix4d add / subtract / scale
/////////////////////////////////
static void BM_Matrix4dAdd(benchmark::State& state) {
    auto m1 = Matrix4<double>(1);
    auto m2 = Matrix4<double>(2);
    for (auto _ : state)
        benchmark::DoNotOptimize(m1 + m2);
}
BENCHMARK(BM_Matrix4dAdd);

static void BM_Matrix4dSIMDAdd(benchmark::State& state) {
    Matrix4d m1 = Matrix4d(1);
    Matrix4d m2 = Matrix4d(2);
    for (auto _ : state)
        benchmark::DoNotOptimize(m1 + m2);
}
BENCHMARK(BM_Matrix4dSIMDAdd);

static void BM_Matrix4dScalarMultiply(benchmark::State& state) {
    auto m = Matrix4<double>(1);
    for (auto _ : state)
        benchmark::DoNotOptimize(m * 2.5);
}
BENCHMARK(BM_Matrix4dScalarMultiply);

static void BM_Matrix4dSIMDScalarMultiply(benchmark::State& state) {
    Matrix4d m = Matrix4d(1);
    for (auto _ : state)
        benchmark::DoNotOptimize(m * 2.5);
}
BENCHMARK(BM_Matrix4dSIMDScalarMultiply);

static void BM_GLM_Mat4dAdd(benchmark::State& state) {
    glm::dmat4 m1(1), m2(2);
    for (auto _ : state)
        benchmark::DoNotOptimize(m1 + m2);
}
BENCHMARK(BM_GLM_Mat4dAdd);

static void BM_GLM_Mat4dScalarMultiply(benchmark::State& state) {
    glm::dmat4 m(1);
    for (auto _ : state)
        benchmark::DoNotOptimize(m * 2.5);
}
BENCHMARK(BM_GLM_Mat4dScalarMultiply);

///////////////////////////////
// Matrix4f x Vector (ARM)   //
///////////////////////////////
static void BM_Matrix4fByVectorGeneric(benchmark::State& state) {
    auto m = Matrix4<float>::identity();
    auto v = Vector4<float>(1,2,3,4);
    for (auto _ : state)
        benchmark::DoNotOptimize(m * v);
}
BENCHMARK(BM_Matrix4fByVectorGeneric);

////////////////////
// Normalize      //
////////////////////
static void BM_NormalizeVector(benchmark::State& state) {
    auto v = Vector3<float>(1,2,3);
    for (auto _ : state){
        auto copy = v;
        copy.normalize();
        benchmark::DoNotOptimize(copy);
    }
}
BENCHMARK(BM_NormalizeVector);

static void BM_GLM_NormalizeVector(benchmark::State& state) {
    glm::vec3 v(1,2,3);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::normalize(v));
    }
}
BENCHMARK(BM_GLM_NormalizeVector);

////////////////////
// AngleTo        //
////////////////////
static void BM_AngleToVector(benchmark::State& state) {
    auto v1 = Vector3<float>(1,0,0);
    auto v2 = Vector3<float>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(v1.angleTo(v2));
    }
}
BENCHMARK(BM_AngleToVector);

static void BM_GLM_AngleToVector(benchmark::State& state) {
    glm::vec3 v1(1,0,0), v2(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::angle(v1, v2));
    }
}
BENCHMARK(BM_GLM_AngleToVector);

////////////////////
// Reflect        //
////////////////////
static void BM_ReflectVector(benchmark::State& state) {
    auto v = Vector3<float>(1,2,3);
    auto n = Vector3<float>(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(v.reflected(n));
    }
}
BENCHMARK(BM_ReflectVector);

static void BM_GLM_ReflectVector(benchmark::State& state) {
    glm::vec3 v(1,2,3), n(0,1,0);
    for (auto _ : state){
        benchmark::DoNotOptimize(glm::reflect(v, n));
    }
}
BENCHMARK(BM_GLM_ReflectVector);

BENCHMARK_MAIN();
