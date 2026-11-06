include_directories(${Python3_INCLUDE_DIRS})

get_target_property(PYTHON_COMMON_SOURCE_DIR pyariadne-python-common SOURCE_DIR)

add_library(pyariadne-ann-submodule-obj OBJECT ann_submodule.cpp)
target_include_directories(pyariadne-ann-submodule-obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PYTHON_COMMON_SOURCE_DIR}/include"
)
target_link_libraries(pyariadne-ann-submodule-obj PRIVATE
    pyariadne-python-common
    pybind11::module
    ann
)
set_target_properties(pyariadne-ann-submodule-obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyariadne-ann INTERFACE)
target_sources(pyariadne-ann INTERFACE
    $<TARGET_OBJECTS:pyariadne-ann-submodule-obj>
)
target_include_directories(pyariadne-ann INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(pyariadne-ann INTERFACE pyariadne-python-common)

ariadne_add_python_module(NAME ann MODULE_IMPORT_NAME pyariadne_ann MODULE_SOURCE ann_module.cpp)
