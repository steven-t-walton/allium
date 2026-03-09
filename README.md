# allium: a high-performance library for deterministic thermal radiative transfer
*allium* is a software framework designed to facilitate the development and analysis of numerical algorithms relevant to radiation transport. Built on top of the MFEM finite element library, the package integrates a range of third-party dependencies, such as hypre, SUNDIALS, igraph, Lua, YAML, and GoogleTest, to provide an agile environment capable of rapid prototyping. The package is designed to scale well on massively parallel, CPU-based computer architectures through the utilization of MPI, OpenMP, and the high-performance implementations provided by the dependent packages. This allows testing novel algorithms at a computational scale not typically investigated in traditional academic literature. *allium* is comprised of a robust build system, core source code implementing both established and novel numerical methods, a suite of drivers that accept Lua-based input files and exercise the solution algorithms, and python scripts that facilitate the processing of the output from the drivers for publication in journal articles. In addition, a comprehensive set of tests ensures reliability and supports ongoing research and application development. Through its modular design and use of high-quality third party libraries, this package is uniquely suited to support academic research into the mathematical algorithms that underpin the simulation of radiation transport. 

## Project Overview
*allium* is a software framework providing
* CMake build system 
* Lua-based input system 
* YAML-compatible terminal output system 
* upwind Discontinuous Galerkin transport discretization with **MPI-parallel fully upwind sweep** and OpenMP threading 
* a variety of diffusion-based acceleration and preconditioning techniques 
* support for fixed-point, Krylov solvers, and advanced nonlinear solvers 
* support for arbitrary meshes in 1, 2, and 3 dimensions 
* visualization via VisIt, Paraview, and GLVis 

This package has been designed under the following design considerations: 
* separation of concerns: the library is designed so that its components, such as input/output, data structures, and computations, are weakly coupled. 
* single responsibility principle: code units (classes, free functions, etc) do only one of input/output, storage of data, or computations, further separating concerns and reducing edge cases.  
* dependency injection/inversion: objects accept fully configured dependencies enabling access to low-level configuration without exposing low-level details. 
* composition over inheritance: functionality is assembled from small, focused components. Modularity is achieved through shallow inheritance hierarchies from simple and generic base classes. 

These design principles have resulted in a focused core library where the user has the control to compose core capabilities together to achieve a wide spectrum of functionality. 
This approach has allowed significant control of and access to low-level details important for academic research while keeping the scope creep of a globally accessible design at bay. 
In addition, weakly coupling components has allowed testing components independently and enables the ability to quickly create new physics drivers. 
In general, *allium* has been designed to have all input/output, configuration, management of data lifetime, and execution flow within the `main` of the driver allowing the core library to be simple and extensible. 
This design allows unexpected and out-of-scope concepts to be implemented with changes primarily restricted to the `main` function, reducing implementation burden. 

### Drivers 
Physics capabilities are organized into a suite of drivers in the `exe` directory. Example inputs and tests for each driver are provided in the correspondingly named directory in `inputs/` and `tests/`, respectively. 

### exe/chive
`chive` is a driver for steady-state, one-group, fixed-source transport problems often used as a proxy application for thermal radiative transfer. `chive` supports a variety of acceleration and preconditioning techniques including 
* "fully consistent" P1 diffusion synthetic acceleration (DSA)
* Local Discontinuous Galerkin (LDG) DSA 
* Modified Interior Penalty (MIP) DSA 
* Interior Penalty and LDG-based Second Moment Methods (SMMs)
Both fixed-point iteration and Krylov-based solvers are available. 

### exe/scallion 
`scallion` is a driver for multigroup, time-dependent thermal radiative transfer. Backward Euler time integration is used. `scallion` implements Picard, linearized, and full Newton solution algorithms. The linearized and Newton algorithms can be solved with Krylov or fixed-point iteration preconditioned with a subset of the algorithms implemented in `chive`. `scallion` supports
* trace plots (output solution at a point in space plotted over time)
* restarting from a previous simulation 
* analytic and tabular opacity data 
* implicit and explicit treatment of opacities 
* variable time steps defined by a function or table

### exe/green 
`green` is a variant of `scallion` that uses the Second Moment Method (SMM) to solve the equations of thermal radiative transfer. Where `scallion` evolves the intensity and material temperature, `green` evolves the intensity, low-order moments, and material temperature. This subtle difference motivated separating the two sets of methods into separate drivers. `green` is intended to parallel `scallion` and is thus compatible with all inputs to `scallion`. 

### exe/ramp 
`ramp` is simplification of `green` which only solves the low-order system (e.g. radiation diffusion). 

## Scripts
Post-processing scripts are provided in `scripts/`
* `inspect`: opens a YAML-compatible terminal output file and indexes into the YAML tree 
* `gridfunction.py`: class for loading and plotting 1D `mfem::GridFunction` 
* `plot_tracer.py`: plots tracers over time 
* `plot_visit.py`: uses `gridfunction.py` to plot MFEM's VisIt output format over time 
* `darwin/build.sh`: example build script (including all third party libraries) for linux-based systems 

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

## Release 
* [GPL 2.0 license](LICENSE.md)
* LANL code designation: O5022

© 2026. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.