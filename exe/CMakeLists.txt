LIST(APPEND EXE
	sweep.cpp
	parsweep.cpp
	scratch.cpp
)

foreach(exe ${EXE})
	get_filename_component(name ${exe} NAME) 
	string (REPLACE ".cpp" "" exename ${name})
	add_executable(${exename} ${exe})
	target_link_libraries(${exename} PRIVATE sweep_lib) 
endforeach()

