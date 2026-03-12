#!/bin/bash 

# script to download and install all third party libraries (TPLs)
# downloads TPLs into the directory specified by TPL variable 
# designed to work on darwin
# NOTE: compilation is done in parallel so allocate 
# a node with `salloc` first 

# number of processors to use with make 
nproc=36 
ROOT=$(pwd)
# where to download and build all third party libraries 
# this location can be external to allium 
TPL=${ROOT}/tpl

# create directory for tpls 
mkdir -p ${TPL}

# this compiler verified to work on darwin 
# NOTE: the openmpi module must be loaded to run the code as well!
module load openmpi/4.1.5-gcc_12.2.0 cmake/3.26.3

# --- clone dependencies into project source --- 
cd ${TPL}
git clone https://github.com/mfem/mfem.git 
git clone https://github.com/hypre-space/hypre.git
git clone https://github.com/mfem/tpls.git
git clone https://github.com/libunwind/libunwind.git
git clone https://github.com/xiaoyeli/superlu_dist.git
git clone https://github.com/igraph/igraph.git
git clone https://github.com/kokkos/mdspan.git
git clone https://github.com/jbeder/yaml-cpp.git
git clone https://github.com/ThePhD/sol2.git
git clone https://github.com/google/googletest.git 
git clone https://github.com/LLNL/sundials.git 
git clone https://github.com/Nek5000/gslib

# pin all repos to specific SHAs for reproducibility
git -C hypre        checkout 7e7fc8ce09153c60ae538a52a5f870f93b9608ca
git -C sundials     checkout aaeab8d907c6b7dfca86041401fdc1448f35f826
git -C igraph       checkout beebfcdcd707f50a31cf8eb3568cf09f8b7baf54
git -C mfem         checkout 171eeb403f8d8deadfbd64d547de43c8705e616c
git -C superlu_dist checkout 4ae28e6f6d38f4f1fd3da88bd040f756d1275844
git -C libunwind    checkout c19e28b0e8682155482d937131fd5b1553044a66
git -C mdspan       checkout 80fc772eb812b45097c28fc0a46d8ff006138d69
git -C yaml-cpp     checkout 8bcadb1509e63248e79c6dc09928f95cfae02b74
git -C sol2         checkout c1f95a773c6f8f4fde8ca3efe872e7286afe4444
git -C googletest   checkout 94be250af7e14c58dcbf476972d2d7141551ff67
git -C gslib        checkout 95acf5b42301d6cb48fda88d662f1d784b863089
git -C tpls         checkout 486dd7171e67629fc85393479f710e821e0e8b77

# --- build hypre --- 
cd ${TPL}/hypre/
cd src/cmbuild
cmake .. -DCMAKE_BUILD_TYPE=Release
# installs into hypre/src/hypre/lib64 
make install -j${nproc} 

# --- build parmetis --- 
cd ${TPL}
# unpack from mfem tpl repo 
tar -xvf tpls/parmetis-4.0.3.tar.gz 
PMROOT=${TPL}/parmetis-4.0.3/
cd ${PMROOT}
make config prefix=install 
make install -j${nproc} 

# --- build libunwind --- 
cd ${TPL}/libunwind
autoreconf -i
./configure --prefix=$(pwd)/install
make install -j${nproc} 

# --- build superlu --- 
cd ${TPL}/superlu_dist 
mkdir build install 
cd build
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../install \
 -DTPL_PARMETIS_INCLUDE_DIRS="${PMROOT}/include;${PMROOT}/metis/include/" \
 -DTPL_PARMETIS_LIBRARIES="${PMROOT}/build/Linux-x86_64/libparmetis/libparmetis.a;${PMROOT}/build/Linux-x86_64/libmetis/libmetis.a" 
make install -j${nproc} 

# --- sundials --- 
cd ${TPL}/sundials
mkdir build install 
cd build 
cmake .. \
 -DENABLE_MPI=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../install 
make install -j${nproc} 

# --- gslib --- 
cd ${TPL}/gslib 
make CC=mpicc 

# --- build mfem --- 
cd ${TPL}/mfem 
mkdir build install 
cd build
cmake .. \
 -DMFEM_USE_MPI=TRUE \
 -DHYPRE_DIR=${TPL}/hypre/src/hypre/ \
 -DMETIS_DIR=${PMROOT}/build/Linux-x86_64/libmetis \
 -DMETIS_INCLUDE_DIR=${PMROOT}/metis/include \
 -DMFEM_USE_LIBUNWIND=TRUE \
 -DLIBUNWIND_DIR=${TPL}/libunwind/install \
 -DMFEM_USE_SUPERLU=TRUE \
 -DSuperLUDist_DIR=${TPL}/superlu_dist/install \
 -DParMETIS_DIR=${PMROOT}/build/Linux-x86_64/libparmetis/ \
 -DParMETIS_INCLUDE_DIR=${PMROOT}/include/ \
 -DMFEM_USE_SUNDIALS=TRUE \
 -DSUNDIALS_DIR=${TPL}/sundials/install/ \
 -DMFEM_USE_GSLIB=ON \
 -DGSLIB_DIR=${TPL}/gslib/build \
 -DMFEM_USE_OPENMP=TRUE \
 -DCMAKE_INSTALL_PREFIX=../install \
 -DCMAKE_BUILD_TYPE=Release 
make install -j${nproc} 
# test linker by building examples 
make examples -j${nproc} 

# --- build igraph --- 
cd ${TPL}/igraph 
mkdir build install 
cd build 
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../install 
make install -j${nproc} 

# --- mdspan --- 
cd ${TPL}/mdspan 
mkdir build install 
cd build
cmake .. \
 -DCMAKE_INSTALL_PREFIX=../install 
make install 

# --- yaml ---
cd ${TPL}/yaml-cpp
mkdir build install 
cd build
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../install 
make install 

# --- sol2 --- 
cd ${TPL}/sol2
mkdir build install 
cd build 
cmake .. \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=../install \
 -DSOL2_BUILD_LUA=FALSE \
 -DSOL2_LUA_VERSION=5.3
make install 

# --- googletest --- 
cd ${TPL}/googletest 
mkdir build install 
cd build 
cmake .. \
 -DCMAKE_INSTALL_PREFIX=../install \
 -DCMAKE_BUILD_TYPE=Release 
make install 

# --- finally build allium --- 
cd ${ROOT}
mkdir build 
cd build 
cmake .. \
 -Dmfem_DIR=${TPL}/mfem/install/lib/cmake/mfem \
 -Digraph_DIR=${TPL}/igraph/install/lib64/cmake/igraph \
 -Dmdspan_DIR=${TPL}/mdspan/install/lib64/cmake/mdspan \
 -Dyaml-cpp_DIR=${TPL}/yaml-cpp/install/lib64/cmake/yaml-cpp \
 -Dsol2_DIR=${TPL}/sol2/install/share/cmake/sol2 \
 -DGTest_ROOT=${TPL}/googletest/install/lib64/cmake/GTest \
 -DENABLE_UNIT_TESTS=TRUE \
 -DENABLE_CHIVE_TESTS=TRUE \
 -DCMAKE_BUILD_TYPE=Release 
make -j${nproc} 
# -- run tests --
# explicitly set threads so SuperLU doesn't slow to a crawl
# darwin default must be to use 
# hyper threading 
export OMP_NUM_THREADS=1
# run tests in parallel 
# separating serial vs parallel to not 
# over prescribe threads 
ctest -j ${nproc} -L serial && ctest -j 4 -L parallel
