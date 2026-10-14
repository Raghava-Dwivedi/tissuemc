#pragma once

namespace tissuemc {

#ifdef TISSUEMC_VERSION
inline constexpr const char *version = TISSUEMC_VERSION;
#else
inline constexpr const char *version = "unknown";
#endif

} // namespace tissuemc
