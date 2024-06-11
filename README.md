# Overview
* `exe/`:
	* `chive.cpp`: driver for steady-state, mono-energetic, neutral particle transport supporting 
		* P1, Local Discontinuous Galerkin, and Modified Interior Penalty diffusion synthetic acceleration preconditioners for fixed-point and Krylov solvers
		* Independent Local Discontinuous Galerkin Second Moment Method 
		* arbitrary material descriptions and mesh composition 
		* Lua input system 
		* parallel decomposition with full upwind sweep 
		* output is YAML-parsable 
	* `scallion.cpp`: gray thermal radiative transfer 
		* backward Euler time integration 
		* Picard, Newton, one Newton algorithms 
	* `green.cpp`: gray thermal radiative transfer with consistent second moment methods
	* `spring.cpp`: gray diffusion thermal radiative transfer 
	* `garlic.cpp`: Lagrange hydro + radiation diffusion (under development)
* `inputs/`: example inputs for `chive` and `scallion` 
* `tests/`: collection of tests using the GoogleTest framework. Run with `ctest`. 
* `scripts/`: post-processing python files, build scripts 

# Building 
## Required Dependencies 
* [MFEM](https://github.com/mfem/mfem) built with 
	* MPI
	* [Hypre](https://github.com/hypre-space/hypre)
	* [Metis](https://github.com/mfem/tpls/blob/gh-pages/metis-4.0.3.tar.gz)
	* [Sundials](https://github.com/LLNL/sundials)
* [igraph](https://github.com/igraph/igraph.git) 
* [Kokkos mdspan](https://github.com/kokkos/mdspan)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp.git)
* [sol2](https://github.com/ThePhD/sol2.git) and some version of Lua5.x 
	* use `-DSOL2_BUILD_LUA=FALSE` if using system lua 
	* use `-DSOL2_LUA_VERSION=5.X` if using another version besides 5.4 
## Optional Dependencies 
* MFEM built with 
	* [Suitesparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)
	* [Libunwind](https://github.com/libunwind/libunwind)
	* [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist)
	* [GSLIB](https://github.com/Nek5000/gslib)
* [GoogleTest](https://github.com/google/googletest.git) 

## Installing 
### Darwin 
Steps to build on Darwin are documented in `scripts/darwin/build.sh`. From the root directory run `source scripts/darwin/build.sh`. By default this builds all dependencies in the directory `tpl` though this behavior can be changed. 
An example module file is provided in `scripts/darwin/module_file.lua`. Change the path to allium's build directory and copy into a place where Lmod knows to look for module files. 