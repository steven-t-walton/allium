#include "log.hpp"

LogMap<double,SUM,MAX> TimingLog(MPI_COMM_WORLD), TimingLogPersistent; 
LogMap<int,SUM> EventLog(MPI_COMM_WORLD), EventLogPersistent;
// LogMap<double,MAX> ValueLog(MPI_COMM_WORLD);