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

MultiGroupPlanckCoefficient::MultiGroupPlanckCoefficient(
	const mfem::Array<double> &grid, mfem::Coefficient &T)
	: energy_grid(grid), T(T), mfem::VectorCoefficient(grid.Size()-1)
{
}

void MultiGroupPlanckCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	v.SetSize(vdim);
	const double temperature = T.Eval(trans, ip);
	const double gray = constants::StefanBoltzmann * std::pow(temperature, 4);
	double planck_prev = 0.0;
	double planck_new = 0.0;
	for (int g=0; g<vdim; g++) {
		planck_new = IntegrateNormalizedPlanck(energy_grid[g+1], temperature);
		v(g) = (planck_new - planck_prev) * gray;
		planck_prev = planck_new;
	}
}

MultiGroupRosselandCoefficient::MultiGroupRosselandCoefficient(
	const mfem::Array<double> &grid)
	: energy_grid(grid), mfem::VectorCoefficient(grid.Size()-1)
{
}

void MultiGroupRosselandCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	assert(T);
	v.SetSize(vdim);
	mfem::GridFunctionCoefficient coef(T);
	const double temperature = coef.Eval(trans, ip);
	const double gray = 4.0 * constants::StefanBoltzmann * std::pow(temperature, 3);
	double ross_prev = 0.0;
	double ross_new = 0.0;
	for (int g=0; g<vdim; g++) {
		ross_new = IntegrateNormalizedRosseland(energy_grid[g+1], temperature);
		v(g) = (ross_new - ross_prev) * gray;
		ross_prev = ross_new;
	}
}