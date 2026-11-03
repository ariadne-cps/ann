# Standalone benchmark executable; ordinary build/test targets remain separate.
if(NOT TARGET benchmarks)
    add_custom_target(benchmarks)
endif()

add_executable(benchmark_optimizers benchmark_optimizers.cpp)
# setup_project_library() bundles all transitive submodule object targets
# (including Threading/Logging) into ariadne-neural for standalone builds.
# There is no independent ariadne-threading library in this project.
target_link_libraries(benchmark_optimizers PRIVATE
    ariadne-neural neural
)
add_dependencies(benchmarks benchmark_optimizers)

# A tiny correctness/CLI smoke test, not a performance gate. Build it with
# the test target to ensure compiler warnings are checked by all Debug CI jobs.
if(TARGET tests)
    add_dependencies(tests benchmark_optimizers)
    add_test(NAME benchmark_optimizers_smoke
        COMMAND $<TARGET_FILE:benchmark_optimizers>
            --iterations 2 --samples 3 --batch-size 4 --width 4 --concurrency 0
    )
    set_property(TEST benchmark_optimizers_smoke PROPERTY LABELS "benchmark-smoke")
    if(WIN32)
        set_tests_properties(benchmark_optimizers_smoke PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:ariadne-neural>"
        )
    endif()
endif()
