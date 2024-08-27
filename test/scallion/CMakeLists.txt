set(EXE ${CMAKE_BINARY_DIR}/scallion)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test/scallion)

add_test(NAME "scallion.marshak1d.gci" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_gci.py 
	-e ${EXE} -i ${TEST_DIR}/marshak_thin.lua --np 2 --dt 5e-11 -o 1)
set_tests_properties("scallion.marshak1d.gci" PROPERTIES LABELS "scallion;parallel")