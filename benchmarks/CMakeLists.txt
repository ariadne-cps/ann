# Benchmarks are registered by setup_standalone_project via Benchmarking.
# The installed ANN library does not depend on the benchmark framework.
add_executable(benchmark_optimizers benchmark_optimizers.cpp)
target_link_libraries(benchmark_optimizers PRIVATE
    ariadne-ann ann ariadne-benchmarking)
add_dependencies(benchmarks benchmark_optimizers)
