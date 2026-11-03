if(NOT TARGET function)
    ariadne_add_submodule(function)
endif()
if(NOT TARGET threading)
    ariadne_add_submodule(threading)
endif()

# Testing is a test-only dependency and must not enter production linkage.
ariadne_setup_testing()
