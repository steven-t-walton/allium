-- Lmod module file for darwin 
-- symlink to common place for user module files 
-- with `ln -s module_file.lua ~/.local/modulefiles/allium.lua` 
-- add `module use ~/.local/modulefiles` to shell settings (bashrc, zshrc, etc) 
-- to tell Lmod where to look for local modules 
-- use `module load allium` to load the correct compiler, 
-- mpi environment, and put the executables and scripts into your path 
whatis("load modules and put allium exe into path")
depends_on("cmake/3.26.3")
depends_on("openmpi/4.1.5-gcc_12.2.0")
prepend_path("PATH", "~/allium/build/")
prepend_path("PATH", "~/allium/scripts/")
setenv("OMP_NUM_THREADS", 1)