LIST(APPEND TESTS
	test_angle_quad.cpp
	test_sweep.cpp
)

foreach(test ${TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(${testname} ${test})
	target_link_libraries(${testname} PRIVATE sweep_lib GTest::gtest_main) 
	gtest_discover_tests(${testname})
endforeach()

