cmake_minimum_required(VERSION 3.15)

project(ANN VERSION 0.1.0)

option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(NOT TARGET ariadne-build)
    find_package(Git REQUIRED)
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${CMAKE_CURRENT_SOURCE_DIR}"
                    submodule update --init -- ss/bui
                    RESULT_VARIABLE _result)
    if(_result)
        message(FATAL_ERROR "Cannot initialize build submodule")
    endif()
    include("${CMAKE_CURRENT_SOURCE_DIR}/ss/bui/cmake/ProjectOptions.cmake")
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Each direct submodule verifies its own internal diamonds. Check only
    # the shared dependencies newly brought together by ANN.
    require_same_dependency_commit(Build CHAINS
        "build"
        "threading|build"
        "testing|build"
    )
    require_same_dependency_commit(Testing CHAINS
        "testing"
        "threading|testing"
    )
endif()

setup_project()

include_directories("${PROJECT_SOURCE_DIR}/include")

add_subdirectory(ss)

add_library(ann INTERFACE)
target_include_directories(ann INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/ariadne-ann>
)
target_link_libraries(ann INTERFACE threading)

add_subdirectory(src)
target_link_libraries(ANN_SRC PRIVATE ann)

ariadne_register_public_headers(
    TARGET ann
    DIRECTORIES "${PROJECT_SOURCE_DIR}/include/ann"
    FILES "${PROJECT_SOURCE_DIR}/include/ariadne-ann.hpp"
)

setup_project_library(
    TARGET ariadne-ann
    OBJECTS ANN_SRC
    LINK_LIBRARIES ann
)

setup_standalone_project_tests(
    TARGET ariadne-ann
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
        TARGET ann
        DESTINATION include/ariadne-ann
    )
endif()
