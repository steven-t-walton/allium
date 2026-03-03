# allium: a high-performance library for deterministic thermal radiative transfer
*allium* is a software framework designed to facilitate the development and analysis of numerical algorithms relevant to radiation transport. Built on top of the MFEM finite element library, the package integrates a range of third-party dependencies, such as hypre, SUNDIALS, igraph, Lua, YAML, and GoogleTest, to provide an agile environment capable of rapid prototyping. The package is designed to scale well on massively parallel, CPU-based computer architectures through the utilization of MPI, OpenMP, and the high-performance implementations provided by the dependent packages. This allows testing novel algorithms at a computational scale not typically investigated in traditional academic literature. *allium* is comprised of a robust build system, core source code implementing both established and novel numerical methods, a suite of drivers that accept Lua-based input files and exercise the solution algorithms, and python scripts that facilitate the processing of the output from the drivers for publication in journal articles. In addition, a comprehensive set of tests ensures reliability and supports ongoing research and application development. Through its modular design and use of high-quality third party libraries, this package is uniquely suited to support academic research into the mathematical algorithms that underpin the simulation of radiation transport. 

## Overview
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

## Building 
### Required Dependencies 
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
### Optional Dependencies 
* MFEM built with 
	* [Suitesparse](https://github.com/DrTimothyAldenDavis/SuiteSparse)
	* [Libunwind](https://github.com/libunwind/libunwind)
	* [SuperLU_DIST](https://github.com/xiaoyeli/superlu_dist)
	* [GSLIB](https://github.com/Nek5000/gslib)
* [GoogleTest](https://github.com/google/googletest.git) 

### Installing 
#### Darwin 
Steps to build on Darwin are documented in `scripts/darwin/build.sh`. From the root directory run `source scripts/darwin/build.sh`. By default this builds all dependencies in the directory `tpl` though this behavior can be changed. 
An example module file is provided in `scripts/darwin/module_file.lua`. Change the path to allium's build directory and copy into a place where Lmod knows to look for module files. 

## Release 
* GPL 2.0 license
* LANL code designation: O5022

© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.