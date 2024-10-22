#include "log.hpp"

LogMap<double,SUM,MAX> TimingLog(MPI_COMM_WORLD); 
LogMap<unsigned long int,SUM> EventLog(MPI_COMM_WORLD);
