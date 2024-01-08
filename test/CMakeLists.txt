LIST(APPEND TESTS
	angular_quadrature.cpp
)

foreach(test ${TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(${testname} ${test})
	target_link_libraries(${testname} PRIVATE sweep_lib GTest::gtest GTest::gtest_main) 
	gtest_discover_tests(${testname})
endforeach()

