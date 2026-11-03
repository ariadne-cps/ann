find_library(ARIADNE_NEURAL_LIBRARY NAMES ariadne-neural)
find_path(ARIADNE_NEURAL_INCLUDE_DIR ariadne-neural.hpp PATH_SUFFIXES ariadne-neural)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AriadneNeural DEFAULT_MSG
    ARIADNE_NEURAL_LIBRARY
    ARIADNE_NEURAL_INCLUDE_DIR
)

if(AriadneNeural_FOUND)
    get_filename_component(ARIADNE_NEURAL_INCLUDE_PARENT_DIR "${ARIADNE_NEURAL_INCLUDE_DIR}" DIRECTORY)
    set(ARIADNE_NEURAL_INCLUDE_DIRS
        "${ARIADNE_NEURAL_INCLUDE_PARENT_DIR}"
        "${ARIADNE_NEURAL_INCLUDE_DIR}"
    )
    if(NOT TARGET AriadneNeural::ariadne-neural)
        add_library(AriadneNeural::ariadne-neural UNKNOWN IMPORTED)
        set_target_properties(AriadneNeural::ariadne-neural PROPERTIES
            IMPORTED_LOCATION "${ARIADNE_NEURAL_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ARIADNE_NEURAL_INCLUDE_DIRS}"
        )
    endif()
endif()
