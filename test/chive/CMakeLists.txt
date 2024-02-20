set(CHIVE ${CMAKE_BINARY_DIR}/chive) # location of chive executable
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test/chive) # location of chive tests directory

# --- MMS tests for chive --- 
# consistent P1 SMM test 
add_test(NAME "chive.mms.p1smm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"p1smm\", consistent = true, solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.p1smm" PROPERTIES LABELS "chive;parallel")

# consistent LDG SMM test 
add_test(NAME "chive.mms.ldgsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ldgsmm\", consistent = true, solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.ldgsmm" PROPERTIES LABELS "chive;parallel")

# independent LDG SMM test 
add_test(NAME "chive.mms.indep_ldgsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ldgsmm\", consistent = false, solver = {type = \"direct\"}}" --phi-order 2 --J-order 2 -n 30)
set_tests_properties("chive.mms.indep_ldgsmm" PROPERTIES LABELS "chive;parallel")

# consistent IP SMM test 
add_test(NAME "chive.mms.ipsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ipsmm\", consistent = true, solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.ipsmm" PROPERTIES LABELS "chive;parallel")

# independent IP SMM test 
add_test(NAME "chive.mms.indep_ipsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ipsmm\", consistent = false, solver = {type = \"direct\"}}" --phi-order 2 --J-order 1 -n 15)
set_tests_properties("chive.mms.indep_ipsmm" PROPERTIES LABELS "chive;parallel")

# MIP test 
add_test(NAME "chive.mms.mip" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"mip\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.mip" PROPERTIES LABELS "chive;parallel")

# LDGSA test 
add_test(NAME "chive.mms.ldgsa" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"ldgsa\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.ldgsa" PROPERTIES LABELS "chive;parallel")

# P1SA test 
add_test(NAME "chive.mms.p1sa" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"p1sa\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.p1sa" PROPERTIES LABELS "chive;parallel")

# --- consistency checks for chive --- 
set(CONSISTENT_SMMS p1smm ldgsmm ipsmm)
foreach (smm ${CONSISTENT_SMMS})
	# serial 
	add_test(NAME "chive.consistency.${smm}_quad" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {8,8}, extents = {1,1}}")
	set_tests_properties("chive.consistency.${smm}_quad" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${smm}_tri" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {8,8}, extents = {1,1}, element_type = \"triangle\"}")
	set_tests_properties("chive.consistency.${smm}_tri" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${smm}_mixed" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {file = \"${TEST_DIR}/mixed.mesh\", refinements = 2}")
	set_tests_properties("chive.consistency.${smm}_mixed" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${smm}_hex" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l " driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {4,4,4}, extents = {1,1,1}}")
	set_tests_properties("chive.consistency.${smm}_hex" PROPERTIES LABELS "chive;serial")

	# repeat in parallel 
	add_test(NAME "chive.consistency.${smm}_quad_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {8,8}, extents = {1,1}}" --np 4)
	set_tests_properties("chive.consistency.${smm}_quad_par" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${smm}_tri_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {8,8}, extents = {1,1}, element_type = \"triangle\"}" --np 4)
	set_tests_properties("chive.consistency.${smm}_tri" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${smm}_mixed_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "driver.acceleration.type = \"${smm}\"; mesh = {file = \"${TEST_DIR}/mixed.mesh\", refinements = 2}" --np 4)
	set_tests_properties("chive.consistency.${smm}_mixed_par" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${smm}_hex_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l " driver.acceleration.type = \"${smm}\"; mesh = {num_elements = {4,4,4}, extents = {1,1,1}}" --np 4)
	set_tests_properties("chive.consistency.${smm}_hex_par" PROPERTIES LABELS "chive;parallel")
endforeach()

# --- has thick diffusion limit check --- 
# consistent SMMs
foreach(smm ${CONSISTENT_SMMS})
	add_test(NAME "chive.tdl.${smm}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_tdl.py -e ${CHIVE} -i ${TEST_DIR}/tdl.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"${smm}\", solver = {type = \"direct\"}}" --np 4)
	set_tests_properties("chive.tdl.${smm}" PROPERTIES LABELS "chive;parallel")
endforeach()

# independent SMMs
foreach(smm ldgsmm ipsmm)
	add_test(NAME "chive.tdl.indep_${smm}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_tdl.py -e ${CHIVE} -i ${TEST_DIR}/tdl.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"${smm}\", solver = {type = \"direct\"}, consistent=false}" --np 4)
	set_tests_properties("chive.tdl.indep_${smm}" PROPERTIES LABELS "chive;parallel")
endforeach()

# DSAs 
foreach(dsa p1sa ldgsa mip) 
	add_test(NAME "chive.tdl.${dsa}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_tdl.py -e ${CHIVE} -i ${TEST_DIR}/tdl.lua 
		-l "driver.preconditioner = {type = \"${dsa}\", solver = {type = \"direct\"}}" --np 4)
	set_tests_properties("chive.tdl.${dsa}" PROPERTIES LABELS "chive;parallel")
endforeach()

# --- test kinsol converges --- 
# consistent LDG SMM test 
add_test(NAME "chive.kinsol" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
	-l "driver.solver.type = \"kinsol\"; driver.solver.kdim = 5; driver.acceleration.type = \"ldgsmm\"; mesh = {num_elements = {8,8}, extents = {1,1}}" --np 4)
set_tests_properties("chive.kinsol" PROPERTIES LABELS "chive;parallel")
