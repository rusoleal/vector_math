#pragma once

#include <string>
#include <vector_math/matrix4.hpp>
#include <vector_math/matrix4f.hpp>
#include <vector_math/matrix4d.hpp>

/// @file vector_math.hpp
/// @brief Main entry point for the vector_math library.
///
/// Include this single header to pull in the complete public API:
/// - Vec, Vector2, Vector3, Vector4, Vector4f, Vector4d
/// - Mat, Matrix2, Matrix3, Matrix4, Matrix4f, Matrix4d
/// - Quaternion
///
/// All types live in the `systems::leal::vector_math` namespace.

namespace systems::leal::vector_math {

    /// Returns the library version string in `"MAJOR.MINOR.PATCH"` format.
    std::string getVersion();

}
