set(CHIVE ${CMAKE_BINARY_DIR}/chive) # location of chive executable
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test/chive) # location of chive tests directory

# --- MMS tests for chive --- 
# consistent P1 SMM test 
add_test(NAME "chive.mms.p1smm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"p1smm\", consistent = true, solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.p1smm" PROPERTIES LABELS "chive;parallel")

foreach(bc full half)
	# consistent LDG SMM test 
	add_test(NAME "chive.mms.ldgsmm_${bc}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ldgsmm\", consistent = true, bc_type = \"${bc} range\"}" --phi-order 2 --J-order 2)
	set_tests_properties("chive.mms.ldgsmm_${bc}" PROPERTIES LABELS "chive;parallel")

	# independent LDG SMM test 
	add_test(NAME "chive.mms.indep_ldgsmm_${bc}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ldgsmm\", consistent = false, bc_type=\"${bc} range\"}" --Ne 15 --phi-order 2 --J-order 2)
	set_tests_properties("chive.mms.indep_ldgsmm_${bc}" PROPERTIES LABELS "chive;parallel")

	# consistent IP SMM test 
	add_test(NAME "chive.mms.ipsmm_${bc}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ipsmm\", consistent = true, bc_type=\"${bc} range\"}" --phi-order 2 --J-order 2)
	set_tests_properties("chive.mms.ipsmm_${bc}" PROPERTIES LABELS "chive;parallel")

	# independent IP SMM test 
	add_test(NAME "chive.mms.indep_ipsmm_${bc}" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
		-l "driver.solver.type = \"fp\"; driver.acceleration = {type = \"ipsmm\", consistent = false, bc_type=\"${bc} range\"}" --Ne 15 --phi-order 2 --J-order 1)
	set_tests_properties("chive.mms.indep_ipsmm_${bc}" PROPERTIES LABELS "chive;parallel")

endforeach()

# MIP test 
add_test(NAME "chive.mms.mip" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"mip\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.mip" PROPERTIES LABELS "chive;parallel")

# repeat with Abu Shumays quadrature 
add_test(NAME "chive.mms.abu_shumays" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.sn_order = 20; driver.sn_quadrature_type = 'abu shumays'; driver.preconditioner = {type = \"mip\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.abu_shumays" PROPERTIES LABELS "chive;parallel")

# LDGSA test 
add_test(NAME "chive.mms.ldgsa" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"ldgsa\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.ldgsa" PROPERTIES LABELS "chive;parallel")

# P1SA test 
add_test(NAME "chive.mms.p1sa" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_ooa.py -e ${CHIVE} -i ${TEST_DIR}/mms.lua 
	-l "driver.preconditioner = {type = \"p1sa\", solver = {type = \"direct\"}}" --phi-order 2 --J-order 2)
set_tests_properties("chive.mms.p1sa" PROPERTIES LABELS "chive;parallel")

# --- consistency checks for chive --- 
function(CheckConsistency name drv)
	# serial 
	add_test(NAME "chive.consistency.${name}_quad" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {num_elements = {3,3}, extents = {1,1}}")
	set_tests_properties("chive.consistency.${name}_quad" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${name}_tri" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {num_elements = {3,3}, extents = {1,1}, element_type = \"triangle\"}")
	set_tests_properties("chive.consistency.${name}_tri" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${name}_mixed" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {file = \"${TEST_DIR}/mixed.mesh\", serial_refinements = 2}")
	set_tests_properties("chive.consistency.${name}_mixed" PROPERTIES LABELS "chive;serial")

	add_test(NAME "chive.consistency.${name}_hex" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l " ${drv}; mesh = {num_elements = {2,2,2}, extents = {1,1,1}}")
	set_tests_properties("chive.consistency.${name}_hex" PROPERTIES LABELS "chive;serial")

	# repeat in parallel 
	add_test(NAME "chive.consistency.${name}_quad_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {num_elements = {3,3}, extents = {1,1}}" --np 4)
	set_tests_properties("chive.consistency.${name}_quad_par" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${name}_tri_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {num_elements = {3,3}, extents = {1,1}, element_type = \"triangle\"}" --np 4)
	set_tests_properties("chive.consistency.${name}_tri" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${name}_mixed_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l "${drv}; mesh = {file = \"${TEST_DIR}/mixed.mesh\", serial_refinements = 2}" --np 4)
	set_tests_properties("chive.consistency.${name}_mixed_par" PROPERTIES LABELS "chive;parallel")

	add_test(NAME "chive.consistency.${name}_hex_par" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_consistency.py -e ${CHIVE} -i ${TEST_DIR}/eps.lua 
		-l " ${drv}; mesh = {num_elements = {2,2,2}, extents = {1,1,1}}" --np 4)
	set_tests_properties("chive.consistency.${name}_hex_par" PROPERTIES LABELS "chive;parallel")
endfunction()

CheckConsistency("ldgsmm_half" "driver.acceleration.type = \"ldgsmm\"; driver.acceleration.bc_type = \"half range\"")
CheckConsistency("ldgsmm_half_ref" "driver.acceleration.type = \"ldgsmm\"; driver.acceleration.bc_type = \"half range reflect\"")
CheckConsistency("ldgsmm_full" "driver.acceleration.type = \"ldgsmm\"; driver.acceleration.bc_type = \"full range\"")

CheckConsistency("ipsmm_half" "driver.acceleration.type = \"ipsmm\"; driver.acceleration.bc_type = \"half range\"")
CheckConsistency("ipsmm_half_ref" "driver.acceleration.type = \"ipsmm\"; driver.acceleration.bc_type = \"half range reflect\"")
CheckConsistency("ipsmm_full" "driver.acceleration.type = \"ipsmm\"; driver.acceleration.bc_type = \"full range\"")

CheckConsistency("p1smm" "driver.acceleration.type = \"p1smm\"")

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

# --- GCI tests on "easy" crooked pipe problem --- 
# assesses pointwise convergence on realistic problem 
# compare scalar flux at outflow as mesh is refined, confirm get 1st order convergence 
add_test(NAME "chive.cp.mip" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_gci.py -e ${CHIVE} -i ${TEST_DIR}/easy_cp.lua 
	-l "driver.preconditioner = {type = 'mip', solver = cg_solver}" --nprocs 4 -o 1)
set_tests_properties("chive.cp.mip" PROPERTIES LABELS "chive;parallel")

add_test(NAME "chive.cp.ldgsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_gci.py -e ${CHIVE} -i ${TEST_DIR}/easy_cp.lua 
	-l "driver.acceleration = {type = 'ldgsmm', consistent=true, solver = cg_solver}; driver.solver.type = 'kinsol'" 
	--nprocs 4 -o 1)
set_tests_properties("chive.cp.ldgsmm" PROPERTIES LABELS "chive;parallel")

add_test(NAME "chive.cp.ipsmm" COMMAND ${Python3_EXECUTABLE} ${TEST_DIR}/test_gci.py -e ${CHIVE} -i ${TEST_DIR}/easy_cp.lua 
	-l "driver.acceleration = {type = 'ipsmm', consistent = true, solver = cg_solver}; driver.solver.type = 'kinsol'" --nprocs 4 -o 1)
set_tests_properties("chive.cp.ipsmm" PROPERTIES LABELS "chive;parallel")
