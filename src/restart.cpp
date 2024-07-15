#include "restart.hpp"
#include <filesystem>

struct RestartMetaData
{
	int num_procs;
	int cycle; 
	double time; 
	double time_step;
	int size; 
};

RestartWriter::RestartWriter(MPI_Comm comm, std::string root)
	: comm(comm), root(root)
{
	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &rank);
	if (rank == 0) {
		std::filesystem::path path(root);
		std::filesystem::create_directories(path);		
	}
	file_name = mfem::MakeParFilename(root + "/restart_0.", rank, ".dat");
}

void RestartWriter::Write(const mfem::Vector &x) const
{
	CycleRestartFiles(keep);
	std::ofstream out(file_name, std::ios::binary);
	RestartMetaData data(num_procs, cycle, time, time_step, x.Size());
	out.write(reinterpret_cast<char*>(&data), sizeof(data));
	out.write(reinterpret_cast<char*>(x.GetData()), x.Size() * sizeof(double));
}

void RestartWriter::CycleRestartFiles(int max_lvl) const
{
	if (rank == 0) {
		for (int lvl=max_lvl-1; lvl >= 1; lvl--) {
			const auto old_root = root + "/restart_" + std::to_string(lvl-1) + ".";
			const auto new_root = root + "/restart_" + std::to_string(lvl) + ".";
			for (int p=0; p<num_procs; p++) {
				const auto old_str = mfem::MakeParFilename(old_root, p, ".dat");
				const auto new_str = mfem::MakeParFilename(new_root, p, ".dat");
				std::filesystem::path old_path(old_str), new_path(new_str);
				if (std::filesystem::exists(old_path))
					std::filesystem::rename(old_path, new_path);
			}			
		}
	}
	MPI_Barrier(comm);
}

void LoadFromRestart(MPI_Comm comm, const std::string base, int restart_num,
	mfem::Vector &x, int &cycle, double &time, double &time_step)
{
	int rank, num_procs; 
	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &rank);
	const bool root = rank == 0;
	const auto file_name = mfem::MakeParFilename(
		base + "/restart_" + std::to_string(restart_num) + ".", rank, ".dat");
	std::ifstream inp(file_name, std::ios::binary);
	RestartMetaData data;
	inp.read(reinterpret_cast<char*>(&data), sizeof(data));
	if (data.num_procs != num_procs and root) MFEM_ABORT("MPI ranks does not match restart data");
	const bool size_match = data.size == x.Size(); 
	bool size_match_global;
	MPI_Reduce(&size_match, &size_match_global, 1, MPI_C_BOOL, MPI_LAND, 0, comm);
	if (!size_match_global and root)
		MFEM_ABORT("restart file size wrong");
	inp.read(reinterpret_cast<char*>(x.GetData()), x.Size() * sizeof(double));

	cycle = data.cycle; 
	time = data.time; 
	time_step = data.time_step;
}