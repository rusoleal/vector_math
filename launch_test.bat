cmake -B build/windows/test -S . -DVECTOR_MATH_BUILD_TEST=ON
msbuild build/windows/test/vector_math.sln /p:Configuration=Release
.\build\windows\test\Release\vector_math_test.exe