LIST(APPEND hdrs 
	${CMAKE_CURRENT_SOURCE_DIR}/tvector.hpp	
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/comment_stream.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/sweep.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/p1diffusion.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/transport_op.hpp
)

LIST(APPEND srcs 
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/sweep.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/p1diffusion.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/transport_op.cpp
)

set(SOURCES ${SOURCES} ${srcs} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${hdrs} PARENT_SCOPE)