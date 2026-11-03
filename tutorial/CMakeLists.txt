cmake_minimum_required(VERSION 3.15)
project(TutorialNeural VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(AriadneNeural REQUIRED)

add_executable(tutorial_neural tutorial_neural.cpp)
target_link_libraries(tutorial_neural PRIVATE AriadneNeural::ariadne-neural)
