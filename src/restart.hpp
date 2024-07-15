#pragma once 

#include "mfem.hpp"

// class that facilitates writing binary restart files 
// to disk in parallel 
// RestartWriter::Write writes the vector of solution 
// data to file along with meta data relating to simulation time,
// cycle number, and time step size 
// by default, 2 calls to Write are stored 
// the restarts are cycled so that "0" is the most recent, 
// "1" the second most recent, etc 
// storing at least 2 restart files ensures that 
// restart is possible even if an exit occurs 
// in the process of writing a restart file 
class RestartWriter 
{
private:
	MPI_Comm comm;
	const std::string root;
	std::string file_name;

	int num_procs=1, rank=0;

	int keep = 2;
	int cycle = 0;
	double time = 0.0; 
	double time_step = 0.0;
public:
	RestartWriter(MPI_Comm comm, std::string root);
	void SetNumRestartFiles(int k) { keep = k; }
	void SetCycle(int c) { cycle = c; }
	void SetTime(double t) { time = t; }
	void SetTimeStep(double step) { time_step = step; }
	void Write(const mfem::Vector &x) const;
private:
	void CycleRestartFiles(int current_level) const;
};

// load the vector x, cycle number, time, and time step size 
// from the provided restart directory and restart number 
void LoadFromRestart(MPI_Comm comm, const std::string root, int restart_num,
	mfem::Vector &x, int &cycle, double &time, double &time_step);