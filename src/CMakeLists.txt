LIST(APPEND hdrs 
	${CMAKE_CURRENT_SOURCE_DIR}/tvector.hpp	
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/comment_stream.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/sweep.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/p1diffusion.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/transport_op.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/smm_integrators.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/phase_coefficient.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/linalg.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/dg_trace_coll.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/block_smm_op.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/cons_smm_op.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
)

LIST(APPEND srcs 
	${CMAKE_CURRENT_SOURCE_DIR}/tvector.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/sweep.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/p1diffusion.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/transport_op.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/smm_integrators.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/phase_coefficient.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/linalg.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/dg_trace_coll.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/block_smm_op.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/cons_smm_op.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
)

set(SOURCES ${SOURCES} ${srcs} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${hdrs} PARENT_SCOPE)