#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>

#include "tissuemc/config.hpp"
#include "tissuemc/cpu_engine.hpp"
#include "tissuemc/mci.hpp"
#include "tissuemc/version.hpp"

namespace {

void usage(std::FILE *to) {
    std::fprintf(to, "usage:\n"
                     "  tissuemc --version\n"
                     "  tissuemc run <input.mci> [options]\n"
                     "\n"
                     "options for 'run':\n"
                     "  --photons N    override the photon count in the input file\n"
                     "  --batches B    number of batches for the standard error (default 10)\n"
                     "  --seed S       64-bit random seed (default 1)\n"
                     "  --threads T    CPU threads, 0 = one per hardware thread (default 1)\n"
                     "  --output FILE  write full results as JSON\n");
}

// Parse a non-negative integer option value; exits with a message on bad input.
unsigned long long parse_count(const char *name, const char *text) {
    char *end = nullptr;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || text[0] == '-') {
        std::fprintf(stderr, "tissuemc: %s expects a non-negative integer, got '%s'\n", name, text);
        std::exit(2);
    }
    return v;
}

int run_command(int argc, char **argv) {
    if (argc < 3) {
        usage(stderr);
        return 2;
    }
    const std::string input = argv[2];
    tissuemc::EngineOptions opt;
    std::string output;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char *name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tissuemc: %s needs a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--photons") {
            opt.photons = parse_count("--photons", value("--photons"));
        } else if (a == "--batches") {
            opt.batches = static_cast<int>(parse_count("--batches", value("--batches")));
        } else if (a == "--seed") {
            opt.seed = parse_count("--seed", value("--seed"));
        } else if (a == "--threads") {
            opt.threads = static_cast<int>(parse_count("--threads", value("--threads")));
        } else if (a == "--output") {
            output = value("--output");
        } else {
            std::fprintf(stderr, "tissuemc: unknown option '%s'\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }

    const auto runs = tissuemc::parse_mci_file(input);
    std::ofstream json;
    if (!output.empty()) {
        json.open(output);
        if (!json) {
            std::fprintf(stderr, "tissuemc: cannot write '%s'\n", output.c_str());
            return 1;
        }
        json << "{\"runs\":[";
    }
    int status = 0;
    for (size_t r = 0; r < runs.size(); ++r) {
        const tissuemc::RunResult res = tissuemc::run_cpu(runs[r], opt);
        const auto rd = res.rd(), tt = res.tt(), ab = res.absorbed();
        const auto cons = res.conservation_residual();
        std::printf("run %zu: %llu photons, %zu batches, %.2f s (%s, %s)\n", r + 1,
                    static_cast<unsigned long long>(res.photons()), res.batches.size(),
                    res.wall_seconds, res.engine.c_str(), res.precision.c_str());
        std::printf("  specular R_sp = %.6f\n", res.specular);
        std::printf("  diffuse  R_d  = %.6f +/- %.6f\n", rd.mean, rd.se);
        std::printf("  transmit T_t  = %.6f +/- %.6f\n", tt.mean, tt.se);
        std::printf("  absorbed  A   = %.6f +/- %.6f\n", ab.mean, ab.se);
        std::printf("  conservation residual = %+.2e +/- %.2e\n", cons.mean, cons.se);
        if (res.anomalies() != 0) {
            std::fprintf(stderr,
                         "tissuemc: %llu anomalies (non-finite weight or non-unit "
                         "direction); results are not trustworthy\n",
                         static_cast<unsigned long long>(res.anomalies()));
            status = 1;
        }
        if (json.is_open()) {
            if (r) {
                json << ',';
            }
            tissuemc::write_json(res, json);
        }
    }
    if (json.is_open()) {
        json << "]}\n";
        json.flush();
        if (!json) {
            std::fprintf(stderr, "tissuemc: error writing '%s'\n", output.c_str());
            return 1;
        }
    }
    return status;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc >= 2 && std::strcmp(argv[1], "--version") == 0) {
            std::printf("tissuemc %s\n", tissuemc::version);
            return 0;
        }
        if (argc >= 2 && std::strcmp(argv[1], "run") == 0) {
            return run_command(argc, argv);
        }
        usage(stderr);
        return 2;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "tissuemc: error: %s\n", e.what());
        return 1;
    }
}
