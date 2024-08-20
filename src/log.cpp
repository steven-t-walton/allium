#include "log.hpp"

ParMap<double,MAX> TimingLog(MPI_COMM_WORLD); 
ParMap<int,SUM> EventLog(MPI_COMM_WORLD); 	
ParMap<double,MAX> ValueLog(MPI_COMM_WORLD);