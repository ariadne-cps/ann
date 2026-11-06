find_library(ARIADNE_ANN_LIBRARY NAMES ariadne-ann)
find_path(ARIADNE_ANN_INCLUDE_DIR ariadne-ann.hpp PATH_SUFFIXES ariadne-ann)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AriadneANN DEFAULT_MSG
    ARIADNE_ANN_LIBRARY
    ARIADNE_ANN_INCLUDE_DIR
)

if(AriadneANN_FOUND)
    get_filename_component(ARIADNE_ANN_INCLUDE_PARENT_DIR "${ARIADNE_ANN_INCLUDE_DIR}" DIRECTORY)
    set(ARIADNE_ANN_INCLUDE_DIRS
        "${ARIADNE_ANN_INCLUDE_PARENT_DIR}"
        "${ARIADNE_ANN_INCLUDE_DIR}"
    )
    if(NOT TARGET AriadneANN::ariadne-ann)
        add_library(AriadneANN::ariadne-ann UNKNOWN IMPORTED)
        set_target_properties(AriadneANN::ariadne-ann PROPERTIES
            IMPORTED_LOCATION "${ARIADNE_ANN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ARIADNE_ANN_INCLUDE_DIRS}"
        )
    endif()
endif()
