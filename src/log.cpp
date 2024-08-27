#include "log.hpp"

LogMap<double,SUM,MAX> TimingLog(MPI_COMM_WORLD); 
LogMap<int,SUM> EventLog(MPI_COMM_WORLD); 	
LogMap<double,MAX> ValueLog(MPI_COMM_WORLD);