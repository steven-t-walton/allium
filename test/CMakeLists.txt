add_library(gtest_main_mpi gtest_mpi.cpp)
target_link_libraries(gtest_main_mpi PUBLIC allium GTest::gtest)

LIST(APPEND TESTS
	test_angle_quad.cpp
	test_integrators.cpp
	test_mfem.cpp
	test_phase_coef.cpp
	test_linalg.cpp
	test_cons_smm.cpp
)

LIST(APPEND PAR_TESTS
	test_sweep.cpp
	test_transport_mms.cpp
	test_diffusion_mms.cpp
	test_smm.cpp
)

foreach(test ${TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(${testname} ${test})
	target_link_libraries(${testname} PRIVATE gtest_main_mpi) 
	gtest_discover_tests(${testname} DISCOVERY_TIMEOUT 10)
endforeach()

foreach(test ${PAR_TESTS})
	get_filename_component(name ${test} NAME) 
	string (REPLACE ".cpp" "" testname ${name})
	add_executable(${testname} ${test})
	target_link_libraries(${testname} PRIVATE gtest_main_mpi) 
	gtest_discover_tests(${testname} DISCOVERY_TIMEOUT 10)
	add_test(NAME "${testname} (np = 4)" COMMAND mpirun -np 4 ${testname})
endforeach()

