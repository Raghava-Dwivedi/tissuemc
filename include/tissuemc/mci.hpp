#pragma once
// Reader for MCML .mci input files (L. Wang & S. L. Jacques, MCML manual, chapter 9.1).
//
// The format is line oriented: '#' starts a comment, blank lines are ignored, and parameters
// appear in a fixed order. Reading the same files as the original program lets the validation
// suite run identical inputs through mcml.c and through this code.

#include <istream>
#include <string>
#include <vector>

#include "tissuemc/config.hpp"

namespace tissuemc {

// Parse every run in the stream. Throws ConfigError with a line number on malformed input. Each
// returned run has already passed validate().
std::vector<RunSpec> parse_mci(std::istream &in);

// Convenience wrapper; throws ConfigError if the file cannot be opened.
std::vector<RunSpec> parse_mci_file(const std::string &path);

} // namespace tissuemc
