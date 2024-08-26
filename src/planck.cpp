#include "planck.hpp"

void CheckPlanckSpectrumCovered(MPI_Comm comm, double Emin, double Emax, 
	const mfem::Vector &temperature, double tol)
{
	int rank; 
	MPI_Comm_rank(comm, &rank);
	const auto Tmax = temperature.Max(); 
	const auto Tmin = temperature.Min(); 
	bool warn[2];
	warn[0] = (IntegrateNormalizedPlanck(Emax, Tmin) < 1.0 - tol)
		or (IntegrateNormalizedPlanck(Emax, Tmax) < 1.0 - tol); 
	warn[1] = (IntegrateNormalizedPlanck(Emin, Tmax) > tol)
		or (IntegrateNormalizedPlanck(Emin, Tmin) > tol);

	bool warn_global[2];
	MPI_Reduce(warn, warn_global, 2, MPI_C_BOOL, MPI_LOR, 0, comm);

	if (rank == 0) {
		if (warn_global[0])
			MFEM_WARNING("max group bound not high enough to integrate planck to within " << tol);
		if (warn_global[1])
			MFEM_WARNING("min group bound not low enough to integrate planck to within " << tol);
	}
}