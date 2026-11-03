# ***************************************************************************
# *            python/CMakeLists.txt
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

set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)
set(PYBIND11_FINDPYTHON ON)

find_package(Python3 3.11 COMPONENTS Interpreter Development.Module)

option(PYTHON_ENFORCE_TESTS "Enforce Python tests" OFF)

if(Python3_FOUND)
    report_python_configuration_once()

    if(NOT TARGET pyariadne-function)
        message(FATAL_ERROR "The Function Python binding target is required to build pyariadne-neural.")
    endif()
    add_subdirectory(bindings)

    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        ariadne_finalize_python_module(NAME neural)

        if(WIN32)
            set(_python_install_dir "Lib/site-packages")
        else()
            set(_python_install_dir "lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
        endif()
        set(PYTHON_BINDINGS_INSTALL_DIR "${_python_install_dir}"
            CACHE STRING "Install destination for Python targets relative to the prefix")
        install(TARGETS pyariadne DESTINATION "${PYTHON_BINDINGS_INSTALL_DIR}")

        add_subdirectory(tests)
        add_dependencies(bindings pyariadne)
    endif()
else()
    if(PYTHON_ENFORCE_TESTS)
        message(FATAL_ERROR "Python >= 3.11 with Interpreter and Development.Module components is required when PYTHON_ENFORCE_TESTS=ON.")
    else()
        message(STATUS "Compatible Python development environment not found: Python bindings will not be built.")
    endif()
endif()
