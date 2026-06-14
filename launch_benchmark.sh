cmake -B build/benchmark -S . -DVECTOR_MATH_BUILD_BENCHMARK=ON -DVECTOR_MATH_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release
make -C build/benchmark
build/benchmark/vector_math_benchmark