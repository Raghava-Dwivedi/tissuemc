#pragma once
// Portability layer shared by the CPU (clang/gcc) and CUDA (nvcc) builds.
//
// TMC_HD marks a function as callable from both host and device code. Under a plain C++ compiler
// it expands to nothing, so the exact same physics source compiles into the CPU reference engine
// and into the GPU kernel.

#include <math.h>

#if defined(__CUDACC__)
#define TMC_HD __host__ __device__
#else
#define TMC_HD
#endif

namespace tissuemc::m {

// Overloads that resolve to the float or double C math function. Using the C names keeps them
// available in CUDA device code, where they map to device intrinsics.
TMC_HD inline float sqrt(float x) {
    return ::sqrtf(x);
}
TMC_HD inline double sqrt(double x) {
    return ::sqrt(x);
}
TMC_HD inline float log(float x) {
    return ::logf(x);
}
TMC_HD inline double log(double x) {
    return ::log(x);
}
TMC_HD inline float cos(float x) {
    return ::cosf(x);
}
TMC_HD inline double cos(double x) {
    return ::cos(x);
}
TMC_HD inline float sin(float x) {
    return ::sinf(x);
}
TMC_HD inline double sin(double x) {
    return ::sin(x);
}
TMC_HD inline float fabs(float x) {
    return ::fabsf(x);
}
TMC_HD inline double fabs(double x) {
    return ::fabs(x);
}

} // namespace tissuemc::m
