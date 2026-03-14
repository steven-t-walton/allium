#!/bin/bash

# build all tpls + allium on M1/M2 mac. The script was tested on a system with strict permissions in place, 
# and so not all make & make install commands have -j$(nproc) to speed things up. If your system allows it, 
# you can place -j$(nproc) when "make" or "make install needs to be called."

# --- Configuration ---
nproc=$(sysctl -n hw.logicalcpu)
ROOT=$(pwd)
TPL=${ROOT}/tpl
ARCH=$(uname -m)
OS=$(uname -s)

# --- 1. System Prerequisites (macOS/Homebrew) ---
brew install cmake git libomp autoconf automake libtool lua@5.3
export OMP_PREFIX=$(brew --prefix libomp)
export LUA_PREFIX=$(brew --prefix lua@5.3)
export OMPI_CC=gcc-13
export OMPI_CXX=g++-13
export OMPI_FC=gfortran-13

export CC=mpicc
export CXX=mpicxx
export FC=mpifort

mkdir -p "${TPL}"

# --- 2. Clone & Checkout ---
cd "${TPL}"
repos=( "mfem/mfem"
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
        "Nek5000/gslib" )

for repo in "${repos[@]}"; do
    dir=$(basename "$repo")
    if [ ! -d "$dir" ]; then
        git clone "https://github.com/${repo}.git"
    fi
done

# Checkout specific SHAs
git -C hypre checkout 7e7fc8ce09153c60ae538a52a5f870f93b9608ca
git -C sundials checkout aaeab8d907c6b7dfca86041401fdc1448f35f826
git -C igraph checkout beebfcdcd707f50a31cf8eb3568cf09f8b7baf54
git -C mfem checkout 171eeb403f8d8deadfbd64d547de43c8705e616c
git -C superlu_dist checkout 4ae28e6f6d38f4f1fd3da88bd040f756d1275844
git -C libunwind checkout c19e28b0e8682155482d937131fd5b1553044a66
git -C mdspan checkout 80fc772eb812b45097c28fc0a46d8ff006138d69
git -C yaml-cpp checkout 8bcadb1509e63248e79c6dc09928f95cfae02b74
git -C sol2 checkout c1f95a773c6f8f4fde8ca3efe872e7286afe4444
git -C googletest checkout 94be250af7e14c58dcbf476972d2d7141551ff67
git -C gslib checkout 95acf5b42301d6cb48fda88d662f1d784b863089
git -C tpls checkout 486dd7171e67629fc85393479f710e821e0e8b77

# --- 3. Build TPLs ---

## HYPRE
cd "${TPL}/hypre/src"
rm -rf cmbuild
mkdir cmbuild && cd cmbuild
cmake ..
make && make install

## ParMETIS
cd "${TPL}"
tar -xf tpls/parmetis-4.0.3.tar.gz
PMROOT="${TPL}/parmetis-4.0.3"
cd "${PMROOT}"
sed -i '' 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/g' CMakeLists.txt
sed -i '' 's/cmake_minimum_required(VERSION 2.4)/cmake_minimum_required(VERSION 3.5)/g' metis/CMakeLists.txt
make config prefix=install
make install -j${nproc}
PM_INSTALL=${PMROOT}/build/${OS}-${ARCH}


## SuperLU_DIST
cd "${TPL}/superlu_dist"
rm -rf build && mkdir build && cd build
cmake .. \
    -DTPL_PARMETIS_INCLUDE_DIRS="${PMROOT}/include;${PMROOT}/metis/include" \
    -DTPL_PARMETIS_LIBRARIES="${PM_INSTALL}/libparmetis/libparmetis.a;${PM_INSTALL}/libmetis/libmetis.a"
make install -j${nproc}

## SUNDIALS
cd "${TPL}/sundials"
rm -rf build && mkdir build && cd build
cmake .. \
    -DENABLE_MPI=ON \
    -DEXAMPLES_INSTALL=OFF
make install -j${nproc}

## GSLIB
cd "${TPL}/gslib"
make -j${nproc}

## MFEM
cd "${TPL}/mfem"
rm -rf build install_dir && mkdir build install_dir && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../install_dir \
    -DMFEM_USE_MPI=ON \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DMFEM_USE_OPENMP=ON \
    -DHYPRE_DIR="${TPL}/hypre/src/hypre" \
    -DParMETIS_INCLUDE_DIR="${PM_INSTALL}/include" \
    -DParMETIS_LIBRARY="${PM_INSTALL}/libparmetis/libparmetis.a" \
    -DMETIS_LIBRARY="${PM_INSTALL}/libmetis/libmetis.a" \
    -DMFEM_USE_SUPERLU=ON \
    -DSuperLUDist_DIR="${TPL}/superlu_dist/install" \
    -DMFEM_USE_SUNDIALS=ON \
    -DSUNDIALS_DIR="${TPL}/sundials/" \
    -DMFEM_USE_GSLIB=ON \
    -DGSLIB_DIR="${TPL}/gslib/build/" \
    -DMFEM_USE_LIBUNWIND=ON
make && make install

cd "${TPL}/igraph"
rm -rf build install && mkdir build install && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
make install -j${nproc}

## Remaining C++ dependencies
for dep in mdspan yaml-cpp googletest; do
    cd "${TPL}/${dep}"
    rm -rf build install && mkdir build install && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx
    make install -j${nproc}
done

## Sol2
cd "${TPL}/sol2"
rm -rf build install && mkdir build install && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DSOL2_BUILD_LUA=FALSE \
    -DSOL2_LUA_VERSION=5.3 \
    -DLUA_INCLUDE_DIR="${LUA_PREFIX}/include/lua5.3" \
    -DLUA_LIBRARIES="${LUA_PREFIX}/lib/liblua5.3.dylib"
make install

# --- 4. Build Allium ---
cd "${ROOT}"
rm -rf build && mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -Dmfem_DIR="${TPL}/mfem/install_dir/lib/cmake/mfem" \
    -Digraph_DIR="${TPL}/igraph/install/lib/cmake/igraph" \
    -Dmdspan_DIR="${TPL}/mdspan/install/lib/cmake/mdspan" \
    -Dyaml-cpp_DIR="${TPL}/yaml-cpp/install/lib/cmake/yaml-cpp" \
    -Dsol2_DIR="${TPL}/sol2/install/share/cmake/sol2" \
    -DGTest_ROOT="${TPL}/googletest/install/lib/cmake/GTest" \
    -DENABLE_UNIT_TESTS=TRUE \
    -DENABLE_CHIVE_TESTS=TRUE
make

# -- run tests --
# explicitly set threads so SuperLU doesn't slow to a crawl
# darwin default must be to use 
# hyper threading 
export OMP_NUM_THREADS=1
# run tests in parallel 
# separating serial vs parallel to not 
# over prescribe threads 
ctest -j ${nproc} -L serial && ctest -j 4 -L parallel
