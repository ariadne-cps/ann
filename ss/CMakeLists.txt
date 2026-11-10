if(NOT TARGET threading)
    ariadne_add_submodule(threading)
endif()

# Bindings need only pybind11 and python-common, not Function bindings.
if(BUILD_PYTHON_BINDINGS AND NOT TARGET pyariadne-python-common)
    ariadne_add_submodule(python-common)
endif()

