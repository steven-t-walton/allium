# Building 
## Required Dependencies 
* [MFEM](https://github.com/mfem/mfem) built with MPI, [Hypre](https://github.com/hypre-space/hypre), [Metis](https://github.com/mfem/tpls/blob/gh-pages/metis-4.0.3.tar.gz) and optionally [Suitesparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), [Libunwind](https://github.com/libunwind/libunwind), and [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist) are useful 
* [igraph](https://github.com/igraph/igraph.git) 
* [Kokkos mdspan](https://github.com/kokkos/mdspan)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
* [sol2](https://github.com/ThePhD/sol2.git) and some version of Lua5.x 
	* use `-DSOL2_BUILD_LUA=FALSE` if using system lua 
	* use `-DSOL2_LUA_VERSION=5.X` if using another version besides 5.4 
## Optional Dependencies 
* [GoogleTest](https://github.com/google/googletest.git) 

## Installing 
1. A reasonably modern compiler is required (must support C++20). Verified to work with GCC 12 and 13. 
2. Use the same compiler for all dependencies 
3. MFEM must be built against the branch `smsolivier/ldg` (i.e. `git checkout smsolivier/ldg`)
4. Build MFEM's dependencies following their build instructions. Use CMake if possible. 
4. Build MFEM with 
```
cmake .. -DMFEM_USE_MPI=TRUE -DMFEM_USE_METIS=TRUE -DMFEM_USE_LIBUNWIND=TRUE -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<path to install> -DHYPRE_DIR=<path to hypre> -DMETIS_DIR=<path to Metis> -DLIBUNWIND_DIR=<path to libunwind>
make install 
```
1. Use CMake to build **and install** all other deps. In other words, run `make install` with `-DCMAKE_INSTALL_PREFIX` set if global installation not desired 
2. Invoke CMake on root directory pointing CMake to each dependency's CMake configuration files. An example looks like: 
``` 
cmake .. -Dmfem_DIR=<path to mfem install>/lib/cmake/mfem -Digraph_DIR=<path to igraph>/lib64/cmake/igraph/ -Dmdspan_DIR=<path to mdspan>/lib64/cmake/mdspan -Dyaml-cpp_DIR=<path to yaml-cpp>/lib64/cmake/yaml-cpp -Dsol2_DIR=<path to sol2>/share/cmake/sol2/ -DGTest_ROOT=<path to googletest>/lib64/cmake/GTest -DENABLE_TESTS=TRUE -DCMAKE_BUILD_TYPE=Release 
```
