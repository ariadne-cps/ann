add_library(NEURAL_SRC OBJECT
    neural.cpp
    network.cpp
    backend.cpp
    inference.cpp
    training.cpp
    serialization.cpp
    checkpoint.cpp
    sha256.cpp
)
target_include_directories(NEURAL_SRC PRIVATE
    "${PROJECT_SOURCE_DIR}/include/neural"
)
