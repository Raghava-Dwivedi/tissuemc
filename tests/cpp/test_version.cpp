#include <doctest.h>

#include <cstring>

#include "tissuemc/version.hpp"

TEST_CASE("version string is set by the build") {
    CHECK(std::strcmp(tissuemc::version, "unknown") != 0);
}
