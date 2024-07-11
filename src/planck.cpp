#include "planck.hpp"

void EvalPlanckSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	const auto G = energy_grid.Size() - 1;
	spectrum.SetSize(G);
	double prev = 0.0; 
	for (int g=0; g<G; g++) {
		double next = IntegrateNormalizedPlanck(energy_grid[g+1], T);
		spectrum(g) = next - prev;
		prev = next;
	}
}

void EvalRosselandSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	const auto G = energy_grid.Size() - 1;
	spectrum.SetSize(G);
	double prev = 0.0;
	for (int g=0; g<G; g++) {
		double next = IntegrateNormalizedRosseland(energy_grid[g+1], T);
		spectrum(g) = next - prev;
		prev = next;
	}
}