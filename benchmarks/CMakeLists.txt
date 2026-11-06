# Standalone benchmark executable; ordinary build/test targets remain separate.
if(NOT TARGET benchmarks)
    add_custom_target(benchmarks)
endif()

add_executable(benchmark_optimizers benchmark_optimizers.cpp)
# setup_project_library() bundles all transitive submodule object targets
# (including Threading/Logging) into ariadne-ann for standalone builds.
# There is no independent ariadne-threading library in this project.
target_link_libraries(benchmark_optimizers PRIVATE
    ariadne-ann ann
)
add_dependencies(benchmarks benchmark_optimizers)
