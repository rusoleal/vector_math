cmake -B build/windows/benchmark -S . -DVECTOR_MATH_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
msbuild build/windows/benchmark/vector_math.sln /p:Configuration=Release
.\build\windows\benchmark\Release\vector_math_benchmark.exe
