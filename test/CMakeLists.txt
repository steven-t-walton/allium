add_library(gtest_main_mpi gtest_mpi.cpp)
target_link_libraries(gtest_main_mpi PUBLIC alium GTest::gtest)

LIST(APPEND TESTS
	test_angle_quad.cpp
	test_sweep.cpp
	test_transport_mms.cpp
	test_integrators.cpp
	test_diffusion_mms.cpp
)

LIST(APPEND PAR_TESTS
	test_sweep.cpp
	test_transport_mms.cpp
	test_diffusion_mms.cpp
)

foreach(test ${TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(${testname} ${test})
	target_link_libraries(${testname} PRIVATE gtest_main_mpi) 
	gtest_discover_tests(${testname})
endforeach()

foreach(test ${PAR_TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(par_${testname} ${test})
	target_link_libraries(par_${testname} PRIVATE gtest_main_mpi) 
	add_test(NAME "${testname} (np = 4)" COMMAND mpirun -np 4 par_${testname})
endforeach()

