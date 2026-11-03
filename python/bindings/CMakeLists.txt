include_directories(${Python3_INCLUDE_DIRS})

get_target_property(PYTHON_COMMON_SOURCE_DIR pyariadne-python-common SOURCE_DIR)

add_library(pyariadne-neural-submodule-obj OBJECT neural_submodule.cpp)
target_include_directories(pyariadne-neural-submodule-obj PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PYTHON_COMMON_SOURCE_DIR}/include"
    $<TARGET_PROPERTY:pyariadne-function,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(pyariadne-neural-submodule-obj PRIVATE
    pyariadne-python-common
    pybind11::module
    neural
)
set_target_properties(pyariadne-neural-submodule-obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(pyariadne-neural INTERFACE)
target_sources(pyariadne-neural INTERFACE
    $<TARGET_OBJECTS:pyariadne-neural-submodule-obj>
)
target_include_directories(pyariadne-neural INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<TARGET_PROPERTY:pyariadne-function,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(pyariadne-neural INTERFACE pyariadne-function)

ariadne_add_python_module(NAME neural MODULE_SOURCE neural_module.cpp)
