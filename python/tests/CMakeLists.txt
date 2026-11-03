# ***************************************************************************
# *            python/tests/CMakeLists.txt
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

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pytest --version
    RESULT_VARIABLE PYTEST_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(PYTEST_result EQUAL 0)
    set(PYTHON_TESTS ON)
    message(STATUS "pytest found for ${Python3_EXECUTABLE}; Python tests will run.")
else()
    if(PYTHON_ENFORCE_TESTS)
        message(FATAL_ERROR "pytest is unavailable for ${Python3_EXECUTABLE}.")
    else()
        message(WARNING "pytest is unavailable; Neural Python tests will be skipped.")
    endif()
endif()

if(PYTHON_TESTS)
    add_custom_target(test_python_neural
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/test_neural.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME test_python_neural
        COMMAND "${Python3_EXECUTABLE}" -m pytest "${CMAKE_CURRENT_SOURCE_DIR}/test_neural.py"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_property(TEST test_python_neural PROPERTY
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:pyariadne>"
    )
    set_property(TEST test_python_neural PROPERTY LABELS "python")
    add_dependencies(test_python_neural pyariadne)
    add_dependencies(tests pyariadne)
endif()
