#include <vector_math/vector_math.hpp>
#include "vector_math_config.h"

using namespace systems::leal::vector_math;

std::string systems::leal::vector_math::getVersion() {
    return std::to_string(vector_math_VERSION_MAJOR) + "." + std::to_string(vector_math_VERSION_MINOR) + "." + std::to_string(vector_math_VERSION_PATCH);
}
