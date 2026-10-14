# TissueMC

**Validated Monte Carlo simulation of light transport in layered biological tissue, with a measured GPU optimization study.**

> **Status: under construction.** No results are claimed yet. Every number that appears here will trace to a committed result file and the command that produced it.

TissueMC reimplements the layered-tissue model of [MCML](https://omlc.org/software/mc/) (Wang, Jacques & Zheng, 1995) as:

- a multithreaded C++ CPU engine (float64), and
- a CUDA engine (float32).

Both engines share one physics code path. They are validated against MCML's published reference values before any performance number is recorded.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design: validation strategy, optimization ladder, and benchmarking methodology.

## Quick start (developer)

```bash
uv sync                                  # Python analysis tools + build tools (cmake, ninja)
uv run cmake -S . -B build -G Ninja      # configure
uv run cmake --build build               # build engine + tests
uv run ctest --test-dir build --output-on-failure
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
