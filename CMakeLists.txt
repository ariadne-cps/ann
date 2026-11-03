cmake_minimum_required(VERSION 3.15)

project(Neural VERSION 0.1.0)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(NOT TARGET ariadne-configuration)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    submodule update --init -- ss/cfg
                    RESULT_VARIABLE _result)
    if(_result)
        message(FATAL_ERROR "Cannot initialize configuration submodule")
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/ss/cfg/cmake/ProjectOptions.cmake")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Each direct submodule verifies its own internal diamonds. Check only
    # the shared dependencies newly brought together by Neural.
    require_same_dependency_commit(Configuration CHAINS
        "configuration"
        "function|configuration"
        "threading|configuration"
        "testing|configuration"
    )
    require_same_dependency_commit(Testing CHAINS
        "testing"
        "function|testing"
        "threading|testing"
    )
    require_same_dependency_commit(Logging CHAINS
        "threading|logging"
        "function|geometry|algebra|interval|numeric|logging"
    )
    require_same_dependency_commit(Utility CHAINS
        "threading|logging|utility"
        "function|geometry|algebra|interval|numeric|foundation|utility"
    )
endif()

setup_project()

include_directories("${PROJECT_SOURCE_DIR}/include")

add_subdirectory(ss)

add_library(neural INTERFACE)
target_include_directories(neural INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ariadne-neural>
)
target_link_libraries(neural INTERFACE function threading)

add_subdirectory(src)
target_link_libraries(NEURAL_SRC PRIVATE neural)

ariadne_register_public_headers(
    TARGET neural
    DIRECTORIES "${PROJECT_SOURCE_DIR}/include/neural"
    FILES "${PROJECT_SOURCE_DIR}/include/ariadne-neural.hpp"
)

setup_project_library(
    TARGET ariadne-neural
    OBJECTS NEURAL_SRC
    LINK_LIBRARIES neural
)

setup_standalone_project_tests(
    TARGET ariadne-neural
    TEST_DIRECTORY tests
    TEST_TARGET tests
    SOURCES
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/src"
)

# The testing orchestrator may auto-register an existing benchmarks directory.
# Register it explicitly only when its executable is not already present.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR
   AND NOT TARGET benchmark_optimizers)
    add_subdirectory(benchmarks)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    ariadne_install_dependency_bundle(
        TARGET neural
        DESTINATION include/ariadne-neural
    )
endif()
