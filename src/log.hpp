#pragma once 
#include "mpi.h"
#include <string>
#include <map>
#include <vector>
#include <numeric>

enum LogOperation {
	SUM, 
	MIN, 
	MAX,
};

template<LogOperation F, typename T, typename U>
void Log(T &map, const std::string key, const U &val)
{
	auto it = map.find(key);
	if (it == map.end()) {
		map[key] = val;
	}

	else {
		auto &old_val = it->second;
		if constexpr (F == LogOperation::MAX) {
			old_val = std::max(old_val, val);
		} else if constexpr (F == LogOperation::MIN) {
			old_val = std::min(old_val, val);
		} else if constexpr (F == LogOperation::SUM) {
			old_val += val;
		}
	}
}

template<typename T, LogOperation Op, LogOperation ParOp=Op>
class LogMap : public std::map<std::string,T> 
{
private:
	MPI_Comm comm = MPI_COMM_NULL; 
	MPI_Datatype T_mpi; 
public:
	LogMap() = default;
	LogMap(MPI_Comm _comm) : comm(_comm) 
	{
		if constexpr (std::is_same<T,double>::value) {
			T_mpi = MPI_DOUBLE; 
		} else if constexpr (std::is_same<T,int>::value) {
			T_mpi = MPI_INT; 
		}
	}

	void Log(const std::string key, const T &val)
	{
		::Log<Op>(*this, key, val);
	}
	void Register(const std::string key) {
		static_assert(std::is_same<T,int>::value, "limiting register to int only");
		::Log<SUM>(*this, key, 1);
	}

	void Synchronize() {
		if (comm == MPI_COMM_NULL) return;
		int rank, size; 
		MPI_Comm_rank(comm, &rank); 
		MPI_Comm_size(comm, &size); 
		if (size == 1) return; 
		
		const int nkeys = this->size(); 
		int nkeys_global;
		MPI_Allreduce(&nkeys, &nkeys_global, 1, MPI_INT, MPI_SUM, comm); 
		if (nkeys_global == 0) return; 

		std::vector<int> key_offsets(this->size()+1);
		std::vector<T> value_data(this->size()); 
		int count = 0; 
		key_offsets[0] = 0; 
		for (const auto &it : *this) {
			value_data[count] = it.second; 
			key_offsets[++count] = it.first.size(); 
		} 
		std::partial_sum(key_offsets.begin(), key_offsets.end(), key_offsets.begin()); 
		char *key_data = new char[key_offsets.back()]; 
		count = 0; 
		for (const auto &it : *this) {
			for (int i=0; i<it.first.size(); i++) {
				key_data[key_offsets[count] + i] = it.first[i]; 			
			}
			count++; 
		} 

		MPI_Request send_request; 
		if (rank > 0) {
			MPI_Isend(key_offsets.data(), key_offsets.size(), MPI_INT, 0, 0, comm, &send_request); 
			MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
			MPI_Isend(key_data, key_offsets.back(), MPI_CHAR, 0, 1, comm, &send_request); 
			MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
			MPI_Isend(value_data.data(), this->size(), T_mpi, 0, 2, comm, &send_request); 
			MPI_Wait(&send_request, MPI_STATUS_IGNORE); 
		}
		else {
			count = 0; 
			while (count < size-1) {
				MPI_Status status[3]; 
				MPI_Probe(MPI_ANY_SOURCE, 0, comm, &status[0]); 
				int nbr_count; 
				MPI_Get_count(&status[0], MPI_INT, &nbr_count); 
				std::vector<int> nbr_key_offsets(nbr_count); 
				MPI_Recv(nbr_key_offsets.data(), nbr_count, MPI_INT, status[0].MPI_SOURCE, 0, comm, MPI_STATUS_IGNORE); 
				char *nbr_key_data = new char[nbr_key_offsets.back()]; 
				MPI_Recv(nbr_key_data, nbr_key_offsets.back(), MPI_CHAR, status[0].MPI_SOURCE, 1, comm, &status[1]); 
				std::vector<T> nbr_value_data(nbr_count-1); 
				MPI_Recv(nbr_value_data.data(), nbr_count-1, T_mpi, status[0].MPI_SOURCE, 2, comm, &status[2]); 

				for (int i=0; i<nbr_count-1; i++) {
					std::string key(nbr_key_data + nbr_key_offsets[i], nbr_key_offsets[i+1] - nbr_key_offsets[i]); 
					const T &nbr = nbr_value_data[i]; 
					::Log<ParOp>(*this, key, nbr);
				}

				delete[] nbr_key_data; 
				count++; 
			}
		}

		delete[] key_data; 
	}
};

extern LogMap<double,SUM,MAX> TimingLog; 
extern LogMap<int,SUM> EventLog; 