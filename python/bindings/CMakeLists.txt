# ***************************************************************************
# *            python/bindings/CMakeLists.txt
# *
# *  Copyright  2026  Luca Geretti
# *
# ***************************************************************************
#
# This file is part of Ariadne Neural.
#
# Ariadne Neural is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ariadne Neural is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ariadne Neural.  If not, see <https://www.gnu.org/licenses/>.

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
