LIST(APPEND hdrs 
	${CMAKE_CURRENT_SOURCE_DIR}/tvector.hpp	
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/comment_stream.hpp
)

LIST(APPEND srcs 
	${CMAKE_CURRENT_SOURCE_DIR}/angular_quadrature.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/mip.cpp
)

set(SOURCES ${SOURCES} ${srcs} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${hdrs} PARENT_SCOPE)