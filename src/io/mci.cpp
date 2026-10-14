#include "tissuemc/mci.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace tissuemc {
namespace {

struct Line {
    int number; // 1-based line number in the file, for error messages
    std::vector<std::string> tokens;
};

[[noreturn]] void fail(int line, const std::string &msg) {
    throw ConfigError("line " + std::to_string(line) + ": " + msg);
}

std::vector<Line> tokenize(std::istream &in) {
    std::vector<Line> lines;
    std::string raw;
    int number = 0;
    while (std::getline(in, raw)) {
        ++number;
        const size_t hash = raw.find('#');
        if (hash != std::string::npos) {
            raw.erase(hash);
        }
        std::istringstream words(raw);
        Line line{number, {}};
        std::string w;
        while (words >> w) {
            line.tokens.push_back(w);
        }
        if (!line.tokens.empty()) {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

class Cursor {
  public:
    explicit Cursor(std::vector<Line> lines) : lines_(std::move(lines)) {}

    // Next non-empty line, which must hold exactly `count` tokens.
    const Line &next(size_t count, const char *what) {
        if (pos_ >= lines_.size()) {
            const int last = lines_.empty() ? 0 : lines_.back().number;
            fail(last, std::string("unexpected end of file, expected ") + what);
        }
        const Line &line = lines_[pos_++];
        if (line.tokens.size() != count) {
            fail(line.number, std::string(what) + ": expected " + std::to_string(count) +
                                  " value(s), found " + std::to_string(line.tokens.size()));
        }
        return line;
    }

    bool done() const { return pos_ >= lines_.size(); }

    // Line number of the next unread line; only valid when !done().
    int line_of_next() const { return lines_[pos_].number; }

  private:
    std::vector<Line> lines_;
    size_t pos_ = 0;
};

double to_double(const Line &line, size_t i, const char *what) {
    const char *s = line.tokens[i].c_str();
    char *end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        fail(line.number, std::string(what) + ": '" + line.tokens[i] + "' is not a finite number");
    }
    return v;
}

// The manual requires integers for counts ("5", not "5.0"); mcml misreads floats there, so reject.
long long to_integer(const Line &line, size_t i, const char *what) {
    const char *s = line.tokens[i].c_str();
    char *end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE) {
        fail(line.number, std::string(what) + ": '" + line.tokens[i] + "' is not an integer");
    }
    return v;
}

} // namespace

std::vector<RunSpec> parse_mci(std::istream &in) {
    Cursor cur(tokenize(in));

    const Line &version = cur.next(1, "file version");
    if (to_double(version, 0, "file version") != 1.0) {
        fail(version.number, "unsupported file version '" + version.tokens[0] + "' (expected 1.0)");
    }
    const Line &nruns_line = cur.next(1, "number of runs");
    const long long nruns = to_integer(nruns_line, 0, "number of runs");
    if (nruns < 1) {
        fail(nruns_line.number, "number of runs must be >= 1");
    }

    std::vector<RunSpec> runs;
    for (long long r = 0; r < nruns; ++r) {
        RunSpec run;
        const Line &out = cur.next(2, "output filename and format");
        run.output_name = out.tokens[0];
        if (out.tokens[1] != "A") {
            fail(out.number, "output format must be 'A' (ASCII), found '" + out.tokens[1] + "'");
        }

        const Line &np = cur.next(1, "number of photons");
        const long long photons = to_integer(np, 0, "number of photons");
        if (photons < 1) {
            fail(np.number, "number of photons must be >= 1");
        }
        run.photons = static_cast<uint64_t>(photons);

        const Line &step = cur.next(2, "dz and dr");
        run.dz = to_double(step, 0, "dz");
        run.dr = to_double(step, 1, "dr");

        const Line &grid = cur.next(3, "grid counts (nz nr na)");
        const long long nz = to_integer(grid, 0, "nz");
        const long long nr = to_integer(grid, 1, "nr");
        const long long na = to_integer(grid, 2, "na");
        constexpr long long kMaxBins = 1'000'000;
        if (nz < 1 || nr < 1 || na < 1 || nz > kMaxBins || nr > kMaxBins || na > kMaxBins) {
            fail(grid.number, "grid counts must be integers in [1, 1000000]");
        }
        run.nz = static_cast<int>(nz);
        run.nr = static_cast<int>(nr);
        run.na = static_cast<int>(na);

        const Line &nl = cur.next(1, "number of layers");
        const long long nlayers = to_integer(nl, 0, "number of layers");
        if (nlayers < 1 || nlayers > kMaxLayers) {
            fail(nl.number, "number of layers must be in [1, " + std::to_string(kMaxLayers) + "]");
        }

        run.n_above = to_double(cur.next(1, "refractive index above"), 0, "n above");
        for (long long i = 0; i < nlayers; ++i) {
            const Line &ll = cur.next(5, "layer (n mua mus g d)");
            const std::string tag = "layer " + std::to_string(i + 1);
            run.layers.push_back(
                {to_double(ll, 0, (tag + " n").c_str()), to_double(ll, 1, (tag + " mua").c_str()),
                 to_double(ll, 2, (tag + " mus").c_str()), to_double(ll, 3, (tag + " g").c_str()),
                 to_double(ll, 4, (tag + " thickness").c_str())});
        }
        const Line &below = cur.next(1, "refractive index below");
        run.n_below = to_double(below, 0, "n below");

        try {
            validate(run);
        } catch (const ConfigError &e) {
            fail(below.number, std::string("invalid run: ") + e.what());
        }
        runs.push_back(std::move(run));
    }
    if (!cur.done()) {
        fail(cur.line_of_next(), "unexpected extra content after the last run");
    }
    return runs;
}

std::vector<RunSpec> parse_mci_file(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        throw ConfigError("cannot open '" + path + "'");
    }
    try {
        return parse_mci(f);
    } catch (const ConfigError &e) {
        throw ConfigError(path + ": " + e.what());
    }
}

} // namespace tissuemc
