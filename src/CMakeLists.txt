add_library(NEURAL_SRC OBJECT
    neural.cpp
)
target_include_directories(NEURAL_SRC PRIVATE
    "${PROJECT_SOURCE_DIR}/include/neural"
)
