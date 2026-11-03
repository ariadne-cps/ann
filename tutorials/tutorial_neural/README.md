# Ariadne Neural standalone tutorial

This is a build/install smoke example, not a neural-network algorithm tutorial.
It deliberately exercises the public umbrella header and installed shared library.

After installing Neural with `cmake --install`:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
./build/tutorial_neural
```

A model and inference tutorial will replace this bootstrap example as the
Neural API is implemented.
