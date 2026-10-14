#include <cmath>
#include <cstdio>

#include "tissuemc/result.hpp"
#include "tissuemc/version.hpp"

namespace tissuemc {

uint64_t RunResult::photons() const {
    uint64_t n = 0;
    for (const auto &b : batches) {
        n += b.photons;
    }
    return n;
}

uint64_t RunResult::events() const {
    uint64_t n = 0;
    for (const auto &b : batches) {
        n += b.events;
    }
    return n;
}

uint64_t RunResult::anomalies() const {
    uint64_t n = 0;
    for (const auto &b : batches) {
        n += b.anomalies;
    }
    return n;
}

namespace {

// Estimate of E[value per photon] from per-batch sums. Batches differ by at most one photon, so
// the standard error is the usual standard error of the mean of the per-batch means.
template <class ValueFn>
Estimate batch_estimate(const std::vector<BatchResult> &batches, ValueFn sum_of) {
    double total = 0;
    uint64_t photons = 0;
    for (const auto &b : batches) {
        total += sum_of(b);
        photons += b.photons;
    }
    Estimate e;
    e.mean = total / static_cast<double>(photons);
    const size_t n = batches.size();
    if (n < 2) {
        e.se = std::nan("");
        return e;
    }
    double mean_of_means = 0;
    for (const auto &b : batches) {
        mean_of_means += sum_of(b) / static_cast<double>(b.photons);
    }
    mean_of_means /= static_cast<double>(n);
    double ss = 0;
    for (const auto &b : batches) {
        const double d = sum_of(b) / static_cast<double>(b.photons) - mean_of_means;
        ss += d * d;
    }
    e.se = std::sqrt(ss / static_cast<double>(n - 1) / static_cast<double>(n));
    return e;
}

double absorbed_sum(const BatchResult &b) {
    double s = 0;
    for (double a : b.absorbed) {
        s += a;
    }
    return s;
}

} // namespace

Estimate RunResult::rd() const {
    return batch_estimate(batches, [](const BatchResult &b) { return b.rd; });
}

Estimate RunResult::tt() const {
    return batch_estimate(batches, [](const BatchResult &b) { return b.tt; });
}

Estimate RunResult::absorbed() const {
    return batch_estimate(batches, [](const BatchResult &b) { return absorbed_sum(b); });
}

Estimate RunResult::conservation_residual() const {
    Estimate e =
        batch_estimate(batches, [](const BatchResult &b) { return b.rd + b.tt + absorbed_sum(b); });
    e.mean = e.mean + specular - 1.0;
    return e;
}

namespace {

class Json {
  public:
    explicit Json(std::ostream &out) : out_(out) {}

    void raw(const char *s) { out_ << s; }

    void str(const std::string &s) {
        out_ << '"';
        for (char c : s) {
            if (c == '"' || c == '\\') {
                out_ << '\\' << c;
            } else if (static_cast<unsigned char>(c) < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(c));
                out_ << buf;
            } else {
                out_ << c;
            }
        }
        out_ << '"';
    }

    // 17 significant digits round-trip a double exactly. JSON has no NaN/Inf, so write null.
    void num(double v) {
        if (!std::isfinite(v)) {
            out_ << "null";
            return;
        }
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.17g", v);
        out_ << buf;
    }

    void integer(uint64_t v) { out_ << v; }

    void key(const char *k) {
        str(k);
        out_ << ':';
    }

    void array(const std::vector<double> &v) {
        out_ << '[';
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) {
                out_ << ',';
            }
            num(v[i]);
        }
        out_ << ']';
    }

    void estimate(const Estimate &e) {
        out_ << '{';
        key("mean");
        num(e.mean);
        out_ << ',';
        key("se");
        num(e.se);
        out_ << '}';
    }

  private:
    std::ostream &out_;
};

} // namespace

void write_json(const RunResult &run, std::ostream &out) {
    Json j(out);
    const RunSpec &s = run.spec;
    j.raw("{");
    j.key("tissuemc_version");
    j.str(tissuemc::version);
    j.raw(",");
    j.key("engine");
    j.str(run.engine);
    j.raw(",");
    j.key("precision");
    j.str(run.precision);
    j.raw(",");
    j.key("seed");
    j.integer(run.seed);
    j.raw(",");
    j.key("threads");
    j.integer(static_cast<uint64_t>(run.threads));
    j.raw(",");
    j.key("photons");
    j.integer(run.photons());
    j.raw(",");
    j.key("output_name");
    j.str(s.output_name);
    j.raw(",");
    j.key("n_above");
    j.num(s.n_above);
    j.raw(",");
    j.key("n_below");
    j.num(s.n_below);
    j.raw(",");
    j.key("layers");
    j.raw("[");
    for (size_t i = 0; i < s.layers.size(); ++i) {
        const LayerSpec &L = s.layers[i];
        j.raw(i ? ",{" : "{");
        j.key("n");
        j.num(L.n);
        j.raw(",");
        j.key("mua");
        j.num(L.mua);
        j.raw(",");
        j.key("mus");
        j.num(L.mus);
        j.raw(",");
        j.key("g");
        j.num(L.g);
        j.raw(",");
        j.key("d");
        j.num(L.d);
        j.raw("}");
    }
    j.raw("],");
    j.key("grid");
    j.raw("{");
    j.key("dz");
    j.num(s.dz);
    j.raw(",");
    j.key("dr");
    j.num(s.dr);
    j.raw(",");
    j.key("nz");
    j.integer(static_cast<uint64_t>(s.nz));
    j.raw(",");
    j.key("nr");
    j.integer(static_cast<uint64_t>(s.nr));
    j.raw("},");
    j.key("specular");
    j.num(run.specular);
    j.raw(",");
    j.key("wall_seconds");
    j.num(run.wall_seconds);
    j.raw(",");
    j.key("events");
    j.integer(run.events());
    j.raw(",");
    j.key("anomalies");
    j.integer(run.anomalies());
    j.raw(",");
    j.key("summary");
    j.raw("{");
    j.key("rd");
    j.estimate(run.rd());
    j.raw(",");
    j.key("tt");
    j.estimate(run.tt());
    j.raw(",");
    j.key("absorbed");
    j.estimate(run.absorbed());
    j.raw(",");
    j.key("conservation_residual");
    j.estimate(run.conservation_residual());
    j.raw("},");
    j.key("batches");
    j.raw("[");
    for (size_t i = 0; i < run.batches.size(); ++i) {
        const BatchResult &b = run.batches[i];
        j.raw(i ? ",{" : "{");
        j.key("first_photon");
        j.integer(b.first_photon);
        j.raw(",");
        j.key("photons");
        j.integer(b.photons);
        j.raw(",");
        j.key("rd");
        j.num(b.rd);
        j.raw(",");
        j.key("tt");
        j.num(b.tt);
        j.raw(",");
        j.key("absorbed");
        j.array(b.absorbed);
        j.raw(",");
        j.key("events");
        j.integer(b.events);
        j.raw(",");
        j.key("anomalies");
        j.integer(b.anomalies);
        j.raw("}");
    }
    j.raw("],");
    j.key("a_z");
    j.array(run.a_z);
    j.raw(",");
    j.key("rd_r");
    j.array(run.rd_r);
    j.raw("}");
}

} // namespace tissuemc
