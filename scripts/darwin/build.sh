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

# --- build hypre --- 
cd ${TPL}/hypre/
git checkout v2.31.0
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
git checkout v6.7.0
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
git checkout smsolivier/ldg 
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
