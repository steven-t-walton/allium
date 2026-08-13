#!/bin/bash

# Script to download and install all third party libraries (TPLs)
# Designed for Apple Silicon Macs using Homebrew GCC/OpenMPI.
#
# Run from the Allium parent directory:
#   ./scripts/mac/gcc_build.sh
#
# Assumes cmake, git, OpenMPI, autoconf, automake, libtool, and lua (5.4) are installed.

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

nproc="$(sysctl -n hw.logicalcpu)"
ROOT="$(pwd)"
TPL="${ROOT}/tpl"
ARCH="$(uname -m)"
OS="$(uname -s)"

mkdir -p "${TPL}"

# Use the default macOS/Homebrew compiler wrappers.
#
# On macOS, gcc/g++ usually resolve to Apple Clang drivers, not GNU GCC.
# Likewise, Homebrew OpenMPI's mpicc/mpicxx wrappers normally use clang/clang++
# MPI wrappers should use their default Clang-based backend. 
# This keeps the C++ stack on a consistent libc++ runtime
# and avoids mixing Homebrew GCC/libstdc++ with Clang/libc++.
#
# For packages that require Fortran, use mpifort/gfortran as needed.

export CC=mpicc
export CXX=mpicxx
export FC=mpifort

echo "Using compiler wrappers:"
echo " CC  = ${CC}"
echo " CXX = ${CXX}"
echo " FC  = ${FC}"

echo "Compiler resolution:"
command -v gcc
command -v g++
command -v mpicc
command -v mpicxx
command -v mpifort

echo "MPI wrapper commands:"
mpicc -show
mpicxx -show
mpifort -show

LIBOMP_PREFIX="$(brew --prefix libomp)"

OPENMP_CMAKE_FLAGS=(
    -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
    -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
    -DOpenMP_C_LIB_NAMES="omp"
    -DOpenMP_CXX_LIB_NAMES="omp"
    -DOpenMP_omp_LIBRARY="${LIBOMP_PREFIX}/lib/libomp.dylib"
)

# -----------------------------------------------------------------------------
# Install roots
# -----------------------------------------------------------------------------
#
# Convention:
#   source tree:  ${TPL}/<package>
#   build tree:   ${TPL}/<package>/build
#   install tree: ${TPL}/<package>/install_dir
#
# ParMETIS is unpacked from mfem/tpls, so its root is different.

HYPRE_ROOT="${TPL}/hypre"
HYPRE_INSTALL="${HYPRE_ROOT}/src/hypre/"

PARMETIS_ROOT="${TPL}/parmetis-4.0.3"
PARMETIS_BUILD="${PARMETIS_ROOT}/build/${OS}-${ARCH}"
PARMETIS_INCLUDE_DIR="${PARMETIS_ROOT}/include"
PARMETIS_LIBRARY="${PARMETIS_BUILD}/libparmetis/libparmetis.a"
METIS_INCLUDE_DIR="${PARMETIS_ROOT}/metis/include"
METIS_LIBRARY="${PARMETIS_BUILD}/libmetis/libmetis.a"

SUPERLU_ROOT="${TPL}/superlu_dist"
SUPERLU_INSTALL="${SUPERLU_ROOT}/install_dir"

SUNDIALS_ROOT="${TPL}/sundials"
SUNDIALS_INSTALL="${SUNDIALS_ROOT}/install_dir"

GSLIB_ROOT="${TPL}/gslib"
GSLIB_BUILD="${GSLIB_ROOT}/build"

MFEM_ROOT="${TPL}/mfem"
MFEM_INSTALL="${MFEM_ROOT}/install_dir"

IGRAPH_ROOT="${TPL}/igraph"
IGRAPH_INSTALL="${IGRAPH_ROOT}/install_dir"

MDSPAN_ROOT="${TPL}/mdspan"
MDSPAN_INSTALL="${MDSPAN_ROOT}/install_dir"

YAML_CPP_ROOT="${TPL}/yaml-cpp"
YAML_CPP_INSTALL="${YAML_CPP_ROOT}/install_dir"

SOL2_ROOT="${TPL}/sol2"
SOL2_INSTALL="${SOL2_ROOT}/install_dir"

GTEST_ROOT="${TPL}/googletest"
GTEST_INSTALL="${GTEST_ROOT}/install_dir"

# -----------------------------------------------------------------------------
# Clone dependencies
# -----------------------------------------------------------------------------

cd "${TPL}"

repos=(
    "mfem/mfem"
    "hypre-space/hypre"
    "mfem/tpls"
    "libunwind/libunwind"
    "xiaoyeli/superlu_dist"
    "igraph/igraph"
    "kokkos/mdspan"
    "jbeder/yaml-cpp"
    "ThePhD/sol2"
    "google/googletest"
    "LLNL/sundials"
    "Nek5000/gslib"
)

for repo in "${repos[@]}"; do
    dir="$(basename "${repo}")"
    if [ ! -d "${dir}" ]; then
        git clone "https://github.com/${repo}.git"
    fi
done

# -----------------------------------------------------------------------------
# Checkout pinned commits
# -----------------------------------------------------------------------------

git -C hypre checkout 7e7fc8ce09153c60ae538a52a5f870f93b9608ca
git -C sundials checkout aaeab8d907c6b7dfca86041401fdc1448f35f826
git -C igraph checkout beebfcdcd707f50a31cf8eb3568cf09f8b7baf54
git -C mfem checkout 171eeb403f8d8deadfbd64d547de43c8705e616c
git -C superlu_dist checkout 4ae28e6f6d38f4f1fd3da88bd040f756d1275844
git -C mdspan checkout 80fc772eb812b45097c28fc0a46d8ff006138d69
git -C yaml-cpp checkout 8bcadb1509e63248e79c6dc09928f95cfae02b74
git -C sol2 checkout c1f95a773c6f8f4fde8ca3efe872e7286afe4444
git -C googletest checkout 94be250af7e14c58dcbf476972d2d7141551ff67
git -C gslib checkout 95acf5b42301d6cb48fda88d662f1d784b863089
git -C tpls checkout 486dd7171e67629fc85393479f710e821e0e8b77

# -----------------------------------------------------------------------------
# HYPRE
# -----------------------------------------------------------------------------

cd "${HYPRE_ROOT}/src"
rm -rf cmbuild
mkdir -p cmbuild
cd cmbuild

cmake ..
make && make install

# -----------------------------------------------------------------------------
# ParMETIS
# -----------------------------------------------------------------------------

cd "${TPL}"
tar -xvf tpls/parmetis-4.0.3.tar.gz

cd "${PARMETIS_ROOT}"
sed -i '' 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/g' CMakeLists.txt
sed -i '' 's/cmake_minimum_required(VERSION 2.4)/cmake_minimum_required(VERSION 3.5)/g' metis/CMakeLists.txt

make config \
    prefix=install \
    gklib_path=metis/GKlib \
    metis_path=metis

make install -j"${nproc}"

# -----------------------------------------------------------------------------
# SuperLU_DIST
# -----------------------------------------------------------------------------

cd "${SUPERLU_ROOT}"
rm -rf build "${SUPERLU_INSTALL}"
mkdir -p build "${SUPERLU_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${SUPERLU_INSTALL}" \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PARMETIS_ROOT}/include;${PARMETIS_ROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PARMETIS_LIBRARY};${METIS_LIBRARY}"

make install -j"${nproc}"

# -----------------------------------------------------------------------------
# SUNDIALS
# -----------------------------------------------------------------------------

cd "${SUNDIALS_ROOT}"
rm -rf build "${SUNDIALS_INSTALL}"
mkdir -p build "${SUNDIALS_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${SUNDIALS_INSTALL}" \
    -DENABLE_MPI=ON \
    -DEXAMPLES_INSTALL=OFF

make -j"${nproc}"
make install

# -----------------------------------------------------------------------------
# GSLIB
# -----------------------------------------------------------------------------

cd "${GSLIB_ROOT}"
make -j"${nproc}"

# -----------------------------------------------------------------------------
# MFEM
# -----------------------------------------------------------------------------

cd "${MFEM_ROOT}"
rm -rf build "${MFEM_INSTALL}"
mkdir -p build "${MFEM_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${MFEM_INSTALL}" \
    -DMFEM_USE_MPI=ON \
    -DMFEM_USE_OPENMP=ON \
    "${OPENMP_CMAKE_FLAGS[@]}" \
    -DHYPRE_DIR="${HYPRE_INSTALL}" \
    -DParMETIS_INCLUDE_DIR="${PARMETIS_INCLUDE_DIR}" \
    -DParMETIS_LIBRARY="${PARMETIS_LIBRARY}" \
    -DMETIS_INCLUDE_DIR="${METIS_INCLUDE_DIR}" \
    -DMETIS_LIBRARY="${METIS_LIBRARY}" \
    -DMFEM_USE_SUPERLU=ON \
    -DSuperLUDist_DIR="${SUPERLU_INSTALL}" \
    -DMFEM_USE_SUNDIALS=ON \
    -DSUNDIALS_DIR="${SUNDIALS_INSTALL}" \
    -DMFEM_USE_GSLIB=ON \
    -DGSLIB_DIR="${GSLIB_BUILD}" \
    -DMFEM_USE_LIBUNWIND=OFF

make
make install

# -----------------------------------------------------------------------------
# igraph
# -----------------------------------------------------------------------------

cd "${IGRAPH_ROOT}"
rm -rf build "${IGRAPH_INSTALL}"
mkdir -p build "${IGRAPH_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${IGRAPH_INSTALL}" \
    -DIGRAPH_WARNINGS_AS_ERRORS=OFF

make install

# -----------------------------------------------------------------------------
# mdspan
# -----------------------------------------------------------------------------

cd "${MDSPAN_ROOT}"
rm -rf build "${MDSPAN_INSTALL}"
mkdir -p build "${MDSPAN_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${MDSPAN_INSTALL}"

make install -j"${nproc}"

# -----------------------------------------------------------------------------
# yaml-cpp
# -----------------------------------------------------------------------------

cd "${YAML_CPP_ROOT}"
rm -rf build "${YAML_CPP_INSTALL}"
mkdir -p build "${YAML_CPP_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${YAML_CPP_INSTALL}" 

make install -j"${nproc}"

# -----------------------------------------------------------------------------
# sol2
# -----------------------------------------------------------------------------

cd "${SOL2_ROOT}" 
rm -rf build "${SOL2_INSTALL}" 
mkdir -p build "${SOL2_INSTALL}" 
cd build

cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX="${SOL2_INSTALL}" \
 -DSOL2_BUILD_LUA=FALSE \
 -DSOL2_LUA_VERSION=5.4

make install 

# -----------------------------------------------------------------------------
# googletest
# -----------------------------------------------------------------------------

cd "${GTEST_ROOT}"
rm -rf build "${GTEST_INSTALL}"
mkdir -p build "${GTEST_INSTALL}"
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${GTEST_INSTALL}" \

make install -j"${nproc}"

# -----------------------------------------------------------------------------
# Allium
# -----------------------------------------------------------------------------

cd "${ROOT}"
rm -rf build
mkdir -p build
cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dmfem_DIR="${MFEM_INSTALL}/lib/cmake/mfem" \
    -Digraph_DIR="${IGRAPH_INSTALL}/lib/cmake/igraph" \
    -Dmdspan_DIR="${MDSPAN_INSTALL}/lib/cmake/mdspan" \
    -Dyaml-cpp_DIR="${YAML_CPP_INSTALL}/lib/cmake/yaml-cpp" \
    -Dsol2_DIR="${SOL2_INSTALL}/share/cmake/sol2" \
    -DGTest_ROOT="${GTEST_INSTALL}/lib/cmake/GTest" \
    -DENABLE_UNIT_TESTS=TRUE \
    -DENABLE_CHIVE_TESTS=TRUE \
    "${OPENMP_CMAKE_FLAGS[@]}"

make

# -----------------------------------------------------------------------------
# Run tests
# -----------------------------------------------------------------------------

# explicitly set threads so SuperLU doesn't slow to a crawl
# darwin default must be to use 
# hyper threading 
export OMP_NUM_THREADS=1
# run tests in parallel 
# separating serial vs parallel to not 
# over prescribe threads 
ctest -j ${nproc} -L serial && ctest -j 4 -L parallel