# Ariadne Neural

Ariadne Neural is the planned standalone C++20 framework for dense feed-forward
neural networks, learning, inference, transfer learning, and validated numerical
evaluation within Ariadne. It is intended to be portable, CPU-first, and free
from mandatory third-party machine-learning runtimes. Future optional backends
may support LibTorch and MLX.

**Status:** M1 canonical topology, frozen binary64 snapshots and backend contracts,
and M2 native-CPU one-sample and batched reference inference are implemented.
Gradients, training and formal certification are **not** implemented.
See [doc/neural_design.md](doc/neural_design.md) for the design and roadmap.

## Dependencies

Production submodules, pinned to explicit commits:

- [Function](https://github.com/ariadne-cps/function) (`ss/fun`): Ariadne
  function abstractions and the transitive validated numeric/algebra stack.
- [Threading](https://github.com/ariadne-cps/threading) (`ss/thr`): CPU
  scheduling, structured tasks, deterministic fixed-block reductions.
- [Configuration](https://github.com/ariadne-cps/configuration) (`ss/cfg`):
  shared CMake build configuration.

[Testing](https://github.com/ariadne-cps/testing) (`ss/tst`) is a test-only
dependency and is **not** linked into the production module. The build checks
shared transitive dependency commits for consistency.

Neural intentionally has no dependency on SMT, Reasoning, System, Evolution,
Linear Programming, or Nonlinear Programming. Reasoning is responsible for
combining Neural with SMT and dynamic-system verification.

## Layout

- `include/neural/` and `include/ariadne-neural.hpp`: public C++ API.
- `src/`: compiled Neural objects.
- `tests/`: standalone C++ tests using Ariadne Testing.
- `python/bindings/`, `python/tests/`: optional `pyariadne` bindings and tests.
- `tutorial/`: C++ and Python `tutorial_neural` examples together, plus
  standalone CMake files for the installed C++ consumer.
- `doc/neural_design.md`: goals, design decisions, and implementation roadmap.

## Build and test

```sh
git clone https://github.com/ariadne-cps/neural.git
cd neural
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel --target tests
ctest --test-dir build --output-on-failure
```

The CMake dependency system initializes required Git submodules as needed.
For a standalone release installation:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --parallel --target ariadne-neural bindings
cmake --install build-release --prefix /path/to/install
```

Optional Python bindings require Python 3.11+ with development headers;
set `-DBUILD_PYTHON_BINDINGS=OFF` to omit them. When enabled, the bindings
are composed into a single `pyariadne` module like the other Ariadne
components. Python bindings expose M2 CPU `infer()` and `infer_batch()` as well as
`neural_version()`.

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
