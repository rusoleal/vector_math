cmake -B build -S . -DVECTOR_MATH_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
make -C build
build/vector_math_benchmark