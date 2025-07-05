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
    float m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    float m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};

    EXPECT_EQ(Matrix4<float>(m1)*Matrix4<float>(m2), Matrix4<float>(m3));
    EXPECT_EQ(Matrix4f(m1)*Matrix4f(m2), Matrix4f(m3));

    EXPECT_EQ(Matrix4<float>(m1)*Vector4<float>(2,6,10,14), Vector4<float>(100,68,228,388));
    EXPECT_EQ(Matrix4f(m1)*Vector4f(2,6,10,14), Vector4f(100,68,228,388));
}

TEST(Matrix4Float, transpose) {
    EXPECT_EQ(Matrix4<float>::identity().transpose(), Matrix4<float>::identity());
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
    double m1[16] = {1,2,3,4,0,1,2,3,5,6,7,8,10,11,12,13};
    double m2[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    double m3[16] = {80,90,100,110,56,62,68,74,176,202,228,254,296,342,388,434};

    EXPECT_EQ(Matrix4<double>(m1)*Matrix4<double>(m2), Matrix4<double>(m3));
    EXPECT_EQ(Matrix4d(m1)*Matrix4d(m2), Matrix4d(m3));

    EXPECT_EQ(Matrix4<double>(m1)*Vector4<double>(2,6,10,14), Vector4<double>(100,68,228,388));
    EXPECT_EQ(Matrix4d(m1)*Vector4d(2,6,10,14), Vector4d(100,68,228,388));
}

TEST(Matrix4Double, transpose) {
    EXPECT_EQ(Matrix4<double>::identity().transpose(), Matrix4<double>::identity());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}