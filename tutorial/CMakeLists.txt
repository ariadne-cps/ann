cmake_minimum_required(VERSION 3.15)
project(TutorialANN VERSION 3.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(AriadneANN REQUIRED)

add_executable(tutorial_ann tutorial_ann.cpp)
target_link_libraries(tutorial_ann PRIVATE AriadneANN::ariadne-ann)
