#include <benchmark/benchmark.h>
#include <vector_math/matrix4.hpp>

// dummy
static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
BENCHMARK(BM_StringCreation);

///////////////////////
// Matrix4 benchmark //
///////////////////////
static void BM_MatrixCreation(benchmark::State& state) {
    for (auto _ : state){
        systems::leal::vector_math::Matrix4<float> emptyMatrix;
    }
}
BENCHMARK(BM_MatrixCreation);

static void BM_MatrixIdentity(benchmark::State& state) {
    for (auto _ : state){
        auto emptyMatrix = systems::leal::vector_math::Matrix4<float>::identity();
    }
}
BENCHMARK(BM_MatrixIdentity);

static void BM_MatrixMultiply(benchmark::State& state) {
    auto m1 = systems::leal::vector_math::Matrix4<float>::identity();
    auto m2 = systems::leal::vector_math::Matrix4<float>(1);
    for (auto _ : state){
        m1 * m2;
    }
}
BENCHMARK(BM_MatrixMultiply);

static void BM_MatrixByVector(benchmark::State& state) {
    auto m1 = systems::leal::vector_math::Matrix4<float>::identity();
    auto v1 = systems::leal::vector_math::Vector4<float>(1,2,3,4);
    for (auto _ : state){
        m1 * v1;
    }
}
BENCHMARK(BM_MatrixByVector);

static void BM_LookAt(benchmark::State& state) {
    auto eye = systems::leal::vector_math::Vector3<float>(0,0,0);
    auto target = systems::leal::vector_math::Vector3<float>(0,0,1000);
    auto up = systems::leal::vector_math::Vector3<float>(0,1,0);
    for (auto _ : state){
        systems::leal::vector_math::Matrix4<float>::lookAt(eye, target, up);
    }
}
BENCHMARK(BM_LookAt);

///////////////////////
// Vector3 benchmark //
///////////////////////
static void BM_CrossVector(benchmark::State& state) {
    auto v1 = systems::leal::vector_math::Vector3<float>(1,2,3);
    auto v2 = systems::leal::vector_math::Vector3<float>(0,2,3);
    for (auto _ : state){
        systems::leal::vector_math::Vector3<float>::cross(v1,v2);
    }
}
BENCHMARK(BM_CrossVector);

static void BM_DotVector(benchmark::State& state) {
    auto v1 = systems::leal::vector_math::Vector3<float>(1,2,3);
    auto v2 = systems::leal::vector_math::Vector3<float>(0,2,3);
    for (auto _ : state){
        systems::leal::vector_math::Vector3<float>::dot(v1,v2);
    }
}
BENCHMARK(BM_DotVector);

BENCHMARK_MAIN();
