cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20) 
project(allium)

# test compiler has std::span
try_compile(HAS_STD_SPAN SOURCE_FROM_CONTENT "span_test.cpp"
	"#include <span>
	int main() { double *data = new double[5]; auto s = std::span<double>(data, 5); delete data; }"
	OUTPUT_VARIABLE output)
if (NOT ${HAS_STD_SPAN})
	message(FATAL_ERROR "compiler does not have std::span")
endif()

# get info about git commit, branch, tag 
if (EXISTS ${PROJECT_SOURCE_DIR}/.git)
	execute_process(
		COMMAND git describe --all --long --abbrev=40 --dirty --always
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		OUTPUT_VARIABLE ALLIUM_GIT_STRING 
		ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(
		COMMAND git rev-parse HEAD 
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		OUTPUT_VARIABLE ALLIUM_GIT_COMMIT 
		ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(
		COMMAND git rev-parse --abbrev-ref HEAD
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		OUTPUT_VARIABLE ALLIUM_GIT_BRANCH
		ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
	execute_process(
		COMMAND git describe --exact-match --tags
		WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
		OUTPUT_VARIABLE ALLIUM_GIT_TAG
		ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
	message(STATUS "allium git string: ${ALLIUM_GIT_STRING}")
endif()

# --- default value for build type --- 
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

# find MFEM 
find_package(mfem REQUIRED NAMES MFEM)
message(STATUS "found MFEM v${MFEM_VERSION}")
message(STATUS "MFEM git string: ${MFEM_GIT_STRING}")
if (NOT ${MFEM_USE_MPI})
	message(FATAL_ERROR "mfem must be built with MPI")
endif()
if (NOT ${MFEM_USE_HYPRE})
	message(FATAL_ERROR "hypre is required")
endif()
if (NOT ${MFEM_USE_METIS})
	message(FATAL_ERROR "metis is required")
endif()
if (NOT ${MFEM_USE_SUPERLU})
	message(WARNING "some functionality disabled when MFEM built without SuperLU")
endif()
if (NOT ${MFEM_USE_SUITESPARSE})
	message(WARNING "some functionality disabled when MFEM built without Suitesparse")
endif()
if (NOT ${MFEM_USE_SUNDIALS})
	message(WARNING "Anderson acceleration disabled when MFEM built without Sundials")
endif()
if (NOT ${MFEM_USE_GSLIB}) 
	message(WARNING "some functionality disabled when MFEM built without GSLIB")
endif()

# find igraph 
find_package(igraph REQUIRED)
message(STATUS "found igraph v${IGRAPH_VERSION}")

# find kokkos' mdspan 
find_package(mdspan REQUIRED)
message(STATUS "found mdspan v${mdspan_VERSION}")

# find yaml 
find_package(yaml-cpp REQUIRED)
message(STATUS "found yaml-cpp v${yaml-cpp_VERSION}")

# find lua 
find_package(Lua REQUIRED QUIET)
message(STATUS "found lua v${LUA_VERSION_STRING}")

# find sol2 
find_package(sol2 REQUIRED)
message(STATUS "found sol2 v${sol2_VERSION}")

# required for superlu 
find_package(OpenMP REQUIRED)

# set location of quadrature files 
set(LS_QUADRATURE_DIR "${CMAKE_SOURCE_DIR}/data/lsquad")
message(STATUS "angular quadrature dir: ${LS_QUADRATURE_DIR}")
set(AS_QUADRATURE_DIR "${CMAKE_SOURCE_DIR}/data/abu-shumays-quad")

# define transport vector layout 
# LEFT: stride fastest in group, then angle, then space 
# RIGHT: stride fastest in space, then angle, then group 
option(TRANSPORT_VECTOR_LAYOUT "mdspan layout")
if (NOT TRANSPORT_VECTOR_LAYOUT) 
	set(TRANSPORT_VECTOR_LAYOUT RIGHT)
endif()
string(TOUPPER ${TRANSPORT_VECTOR_LAYOUT} TRANSPORT_VECTOR_LAYOUT_UPPER)
if(${TRANSPORT_VECTOR_LAYOUT_UPPER} STREQUAL "LEFT") 
message(STATUS "transport vector layout: LEFT (stride in group, then angle, then space)")
set(TRANSPORT_VECTOR_LAYOUT_LEFT 1)
elseif(${TRANSPORT_VECTOR_LAYOUT_UPPER} STREQUAL "RIGHT") 
message(STATUS "transport vector layout: RIGHT (stride in space, then angle, then group)")
set(TRANSPORT_VECTOR_LAYOUT_RIGHT 1)
else()
message(FATAL_ERROR "TRANSPORT_VECTOR_LAYOUT = ${TRANSPORT_VECTOR_LAYOUT_UPPER} not supported")
endif()

# create config file 
set(CONFIG_LOCATION "${CMAKE_BINARY_DIR}/config")
configure_file(data/config.hpp.in ${CONFIG_LOCATION}/config.hpp)
message(STATUS "configure file generated in ${CONFIG_LOCATION}")

# add files from src directory 
set(SOURCES "")
set(HEADERS "") 
add_subdirectory(src) 

# create a library 
# includes all third party libraries 
add_library(allium ${SOURCES} ${HEADERS})
target_link_libraries(allium PUBLIC ${MFEM_LIBRARIES} igraph::igraph std::mdspan yaml-cpp::yaml-cpp ${LUA_LIBRARIES} sol2::sol2 OpenMP::OpenMP_CXX)
target_include_directories(allium PUBLIC ${MFEM_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/config ${PROJECT_SOURCE_DIR}/src ${LUA_INCLUDE_DIR})

# add executables in exe directory 
LIST(APPEND EXE
	exe/chive.cpp
	exe/scallion.cpp
)

foreach(exe ${EXE})
	get_filename_component(name ${exe} NAME) 
	string (REPLACE ".cpp" "" exename ${name})
	add_executable(${exename} ${exe})
	target_link_libraries(${exename} PRIVATE allium) 
endforeach()

# --- tests --- 
# put tests under option so GoogleTest is optional requirement
option(ENABLE_UNIT_TESTS "turn testing on, requires GoogleTest" OFF) 
if (${ENABLE_UNIT_TESTS})
	enable_testing()
	message(STATUS "unit tests enabled")
	# find GoogleTest 
	# specify location setting GTEST_ROOT to cmake install dir of gtest 
	find_package(GTest REQUIRED)
	# enable GTest specific CMake commands 
	include(GoogleTest)
	# add tests listed in test directory
	add_subdirectory(test)
else()
	message(STATUS "unit tests disabled")
endif()

option(ENABLE_CHIVE_TESTS "integrated tests for the chive driver" OFF)
if (${ENABLE_CHIVE_TESTS})
	enable_testing()
	message(STATUS "chive tests enabled")
	# requires python executable 
	set(Python3_FIND_UNVERSIONED_NAMES "FIRST") 
	find_package(Python3 REQUIRED COMPONENTS Interpreter)
	message(STATUS "python3 exe: ${Python3_EXECUTABLE}")
	add_subdirectory(test/chive)
else()
	message(STATUS "chive tests disabled")
endif()