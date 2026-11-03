include(CTest)

set(UNIT_TESTS
    test_neural
    test_network
    test_backend
    test_derivatives
    test_training
    test_umbrella
)

foreach(TEST IN LISTS UNIT_TESTS)
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} PRIVATE ariadne-neural neural ariadne-testing)
    add_test(NAME ${TEST} COMMAND $<TARGET_FILE:${TEST}> --concise)
    set_property(TEST ${TEST} PROPERTY LABELS "neural")
    if(WIN32)
        set_tests_properties(${TEST} PROPERTIES
            ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:$<TARGET_FILE_DIR:ariadne-neural>"
        )
    endif()
endforeach()

add_custom_target(tests)
add_dependencies(tests ${UNIT_TESTS})
