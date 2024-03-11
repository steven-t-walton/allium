#pragma once 
#include "mpi.h"
#include <string>
#include <map>
#include <vector>
#include <numeric>

enum SynchronizeOperation {
	SUM, 
	MIN, 
	MAX,
};

template<typename T, SynchronizeOperation Op>
class ParMap : public std::map<std::string,T> 
{
private:
	MPI_Comm comm; 
	MPI_Datatype T_mpi; 
public:
	ParMap(MPI_Comm _comm) : comm(_comm) 
	{
		if constexpr (std::is_same<T,double>::value) {
			T_mpi = MPI_DOUBLE; 
		} else if constexpr (std::is_same<T,int>::value) {
			T_mpi = MPI_INT; 
		}
	}

	void Synchronize() {
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
					auto it = this->find(key); 
					if (it == this->end()) {
						this->operator[](key) = 0; 
					}
					T &my_value = this->at(key); 
					if constexpr (Op == SynchronizeOperation::SUM) {
						my_value += nbr; 
					} else if constexpr (Op == SynchronizeOperation::MIN) {
						my_value = std::min(my_value, nbr); 
					} else if constexpr (Op == SynchronizeOperation::MAX) {
						my_value = std::max(my_value, nbr); 
					} else {
						static_assert(false, "operation not defined"); 
					}
				}

				count++; 
			}
		}
	}
};

extern ParMap<double,MAX> TimingLog; 
extern ParMap<int,SUM> EventLog; 
