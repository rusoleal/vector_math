#include <cassert>
#include <vector_math/vector_math.hpp>

using namespace systems::leal::vector_math;

int main(int argc, char **argv) {

    
    printf("sizeof(Matrix4<float>): %lu\n", sizeof(Matrix4<float>));
    printf("sizeof(Matrix4<double>): %lu\n", sizeof(Matrix4<double>));

    assert(Matrix4<float>::identity() == Matrix4<float>::identity());

    assert(Matrix4<float>(1) == Matrix4<float>(1));

    assert(Matrix4<float>() != Matrix4<float>(1));

    assert(Matrix4<float>()+Matrix4<float>(1)+Matrix4<float>(1) == Matrix4<float>(2));

    assert(Matrix4<float>(1)*2 == Matrix4<float>(2));

    assert(2*Matrix4<float>(1) == Matrix4<float>(2));

    assert(Matrix4<float>::identity()*Matrix4<float>(3) == Matrix4<float>(3));

    //assert(Matrix4<float>::identity()*Matrix4<float>::rotate(Quaternion<float>()) == Matrix4<float>(3));

    return 0;
}