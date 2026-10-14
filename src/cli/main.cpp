#include <cstdio>
#include <cstring>

#include "tissuemc/version.hpp"

int main(int argc, char **argv) {
    if (argc > 1 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("tissuemc %s\n", tissuemc::version);
        return 0;
    }
    std::fprintf(stderr, "usage: tissuemc --version\n");
    return 2;
}
