#include "mfem.hpp"

using namespace mfem; 

int main(int argc, char *argv[]) {
	// initialize MPI 
	// automatically calls MPI_Finalize 
	mfem::Mpi::Init(argc, argv); 
	const auto rank = mfem::Mpi::WorldRank(); 
	const bool root = rank == 0; 

	// stream for output to terminal
	mfem::OutStream par_out(std::cout);
	// enable only for root so non-root procs don't clutter cout 
	if (rank!=0) par_out.Disable();

	std::string input1, input2, var;
	int cycle1=1, cycle2=1;
	OptionsParser args(argc, argv); 
	args.AddOption(&input1, "-a", "--input_one", "first dataset", true);
	args.AddOption(&input2, "-b", "--input_two", "second dataset", true);
	args.AddOption(&var, "-v", "--var", "variable to compare", true);
	args.AddOption(&cycle1, "-c", "--cycle1", "cycle number to load");
	args.AddOption(&cycle2, "-d", "--cycle2", "cycle number to load");
	args.Parse(); 
	if (!args.Good()) {
		args.PrintUsage(par_out); 
		return 1; 
	}
	// if (root) args.PrintOptions(par_out);

	VisItDataCollection collection1(MPI_COMM_WORLD, input1);
	VisItDataCollection collection2(MPI_COMM_WORLD, input2);
	collection1.Load(0);
	const auto dt1 = collection1.GetTimeStep();
	collection1.Load(cycle1);
	collection2.Load(cycle2);

	const auto time1 = collection1.GetTime();
	const auto time2 = collection2.GetTime();
	if (std::fabs(time1 - time2) > 1e-14) 
		MFEM_ABORT("data collections not at same final time");

	// const auto dt1 = collection1.GetTimeStep();
	const auto dt2 = collection2.GetTimeStep();

	if (dt2 > dt1) MFEM_WARNING("reference time step too large");

	auto *gf_a = collection1.GetField(var);
	auto *gf_b = collection2.GetField(var);
	GridFunctionCoefficient coef_b(gf_b);
	const auto err = mfem::GlobalLpNorm(2.0, gf_a->ComputeL2Error(coef_b), MPI_COMM_WORLD);
	double mag = sqrt(mfem::InnerProduct(MPI_COMM_WORLD, *gf_b, *gf_b));

	if (root)
		printf("%e %.3e\n", dt1, err/mag);
}