#include <gtest/gtest.h>
#include <vector_math/vector_math.hpp>

using namespace systems::leal::vector_math;

TEST(Vector3Float, constructor) {
    EXPECT_EQ(sizeof(Vector3<float>), 12);
    EXPECT_EQ(Vector3<float>(), Vector3<float>(0));
}

TEST(Vector3Float, arithmetics) {
    EXPECT_EQ(Vector3<float>(1,0,0).length(), 1);
    EXPECT_EQ(Vector3<float>(0,1,0).length(), 1);
    EXPECT_EQ(Vector3<float>(0,0,1).length(), 1);
    EXPECT_EQ(-Vector3<float>(1,2,3), Vector3<float>(-1,-2,-3));
}

TEST(Vector3Double, constructor) {
    EXPECT_EQ(sizeof(Vector3<double>), 24);
    EXPECT_EQ(Vector3<double>(), Vector3<double>(0));
}

TEST(Vector3Double, arithmetics) {
    EXPECT_EQ(Vector3<double>(1,0,0).length(), 1);
    EXPECT_EQ(Vector3<double>(0,1,0).length(), 1);
    EXPECT_EQ(Vector3<double>(0,0,1).length(), 1);
    EXPECT_EQ(-Vector3<double>(1,2,3), Vector3<double>(-1,-2,-3));
}

TEST(Matrix4Float, constructors) {
    EXPECT_EQ(sizeof(Matrix4<float>), 64);
    EXPECT_EQ(Matrix4<float>::identity(), Matrix4<float>::identity());
    EXPECT_EQ(Matrix4<float>(1), Matrix4<float>(1));
    EXPECT_NE(Matrix4<float>(), Matrix4<float>(1));
    EXPECT_EQ(Matrix4<float>(1), Matrix4<float>(1));
}

TEST(Matrix4Float, sum) {
    EXPECT_EQ(Matrix4<float>()+Matrix4<float>(1)+Matrix4<float>(1), Matrix4<float>(2));
}

TEST(Matrix4Float, mul) {
    EXPECT_EQ(Matrix4<float>::identity()*Matrix4<float>(1), Matrix4<float>(1));
}

TEST(Matrix4Double, constructors) {
    EXPECT_EQ(sizeof(Matrix4<double>), 128);
    EXPECT_EQ(Matrix4<double>::identity(), Matrix4<double>::identity());
    EXPECT_EQ(Matrix4<double>(1), Matrix4<double>(1));
    EXPECT_NE(Matrix4<double>(), Matrix4<double>(1));
    EXPECT_EQ(Matrix4<double>(1), Matrix4<double>(1));
}

TEST(Matrix4Double, sum) {
    EXPECT_EQ(Matrix4<double>()+Matrix4<double>(1)+Matrix4<double>(1), Matrix4<double>(2));
}

TEST(Matrix4Double, mul) {
    EXPECT_EQ(Matrix4<double>::identity()*Matrix4<double>(1), Matrix4<double>(1));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}