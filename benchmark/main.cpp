#include <benchmark/benchmark.h>
#include <vector_math/matrix4.hpp>

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

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

BENCHMARK_MAIN();
