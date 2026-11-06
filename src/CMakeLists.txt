add_library(ANN_SRC OBJECT
    ann.cpp
    network.cpp
    backend.cpp
    inference.cpp
    training.cpp
    serialization.cpp
    checkpoint.cpp
    sha256.cpp
)
target_include_directories(ANN_SRC PRIVATE
    "${PROJECT_SOURCE_DIR}/include/ann"
)
