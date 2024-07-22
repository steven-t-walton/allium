#include "multigroup.hpp"
#include "planck.hpp"

MultiGroupEnergyGrid::MultiGroupEnergyGrid(const mfem::Array<double> &bounds, const mfem::Array<double> &midpoints)
	: bounds(bounds), midpoints(midpoints)
{
	assert(bounds.Size() - 1 == midpoints.Size());
	widths.SetSize(midpoints.Size());
	for (int g=0; g<Size(); g++) {
		widths[g] = bounds[g+1] - bounds[g];
	}
}

MultiGroupEnergyGrid MultiGroupEnergyGrid::MakeGray(double Emin, double Emax)
{
	mfem::Array<double> bounds(2);
	mfem::Array<double> midpoints(1);
	bounds[0] = Emin; 
	bounds[1] = Emax; 
	midpoints[0] = std::sqrt(std::max(Emin, 1e-5) * Emax);
	return MultiGroupEnergyGrid(bounds, midpoints);
}

MultiGroupEnergyGrid MultiGroupEnergyGrid::MakeLogSpaced(double Emin, double Emax, int G, bool extend_to_zero)
{
	mfem::Array<double> bounds(G+1);
	mfem::Array<double> midpoints(G);
	const double Emin_log = std::log(Emin);
	const double Emax_log = std::log(Emax);

	if (extend_to_zero) {
		const double dE_log = (Emax_log - Emin_log)/(G-1);
		bounds[0] = 0.0; 
		bounds[1] = Emin; 
		for (int g=1; g<bounds.Size()-1; g++) {
			const double exponent = Emin_log + g*dE_log; 
			bounds[g+1] = std::exp(exponent);
		}
	}

	else {
		const double dE_log = (Emax_log - Emin_log)/G;
		bounds[0] = Emin;
		for (int g=1; g<bounds.Size(); g++) {
			const double exponent = Emin_log + g*dE_log;
			bounds[g] = std::exp(exponent);
		}		
	}

	for (int g=0; g<bounds.Size()-1; g++) {
		if (g==0 and extend_to_zero)
			midpoints[g] = (bounds[g] + bounds[g+1])/2;
		else
			midpoints[g] = std::sqrt(bounds[g] * bounds[g+1]);
	}

	return MultiGroupEnergyGrid(bounds, midpoints);
}

MultiGroupEnergyGrid MultiGroupEnergyGrid::MakeEqualSpaced(double Emin, double Emax, int G, bool extend_to_zero)
{
	mfem::Array<double> bounds(G+1);
	mfem::Array<double> midpoints(G);

	if (extend_to_zero) {
		const double dE = (Emax - Emin)/(G-1);
		bounds[0] = 0.0; 
		bounds[1] = Emin; 
		for (int g=1; g<bounds.Size(); g++) {
			bounds[g+1] = Emin + g * dE;
		}
	}

	else {
		const double dE = (Emax - Emin)/G;
		bounds[0] = 0.0; 
		for (int g=1; g<bounds.Size(); g++) {
			bounds[g] = Emin + g*dE;
		}
	}

	for (int g=0; g<bounds.Size()-1; g++) {
		midpoints[g] = (bounds[g] + bounds[g+1])/2;
	}
	return MultiGroupEnergyGrid(bounds, midpoints);
}

WeightedGroupCollapseOperator::WeightedGroupCollapseOperator(
	const mfem::FiniteElementSpace &fes, 
	const MomentVectorExtents &phi_ext,
	mfem::VectorCoefficient &f)
	: fes(fes), phi_ext(phi_ext), f(f)
{
	height = fes.GetVSize(); 
	width = TotalExtent(phi_ext);

	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	spectrum.SetSize(G);
}

void WeightedGroupCollapseOperator::Mult(const mfem::Vector &mg, mfem::Vector &gray) const
{
	assert(mg.Size() == width);
	assert(gray.Size() == height);

	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	gray = 0.0;

	auto mg_view = ConstMomentVectorView(mg.GetData(), phi_ext);
	mfem::Array<int> dofs;
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dofs); 
		auto &trans = *fes.GetMesh()->GetElementTransformation(e); 
		const auto &fe = *fes.GetFE(e); 
		const auto &ir = fe.GetNodes(); 
		for (int n=0; n<ir.Size(); n++) {
			f.Eval(spectrum, trans, ir.IntPoint(n)); 
			for (int g=0; g<G; g++) {
				gray(dofs[n]) += spectrum(g) * mg_view(g,0,dofs[n]);
			}
		}
	}
}

void WeightedGroupCollapseOperator::MultTranspose(const mfem::Vector &gray, mfem::Vector &mg) const 
{
	assert(mg.Size() == width);
	assert(gray.Size() == height);

	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	auto mg_view = MomentVectorView(mg.GetData(), phi_ext);
	mfem::Array<int> dofs;
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dofs);
		auto &trans = *fes.GetMesh()->GetElementTransformation(e); 
		const auto &fe = *fes.GetFE(e); 
		const auto &ir = fe.GetNodes(); 
		for (int n=0; n<ir.Size(); n++) {
			f.Eval(spectrum, trans, ir.IntPoint(n)); 
			for (int g=0; g<G; g++) {
				mg_view(g,0,dofs[n]) = spectrum(g) * gray(dofs[n]);
			}
		}
	}
}

PlanckSpectrumMGCoefficient::PlanckSpectrumMGCoefficient(
	const mfem::Array<double> &grid, mfem::Coefficient &T)
	: energy_grid(grid), T(T), MultiGroupCoefficient(grid.Size()-1)
{
}

double PlanckSpectrumMGCoefficient::Eval(
	int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	const double temperature = T.Eval(trans, ip);
	const double planck_low = IntegrateNormalizedPlanck(energy_grid[g], temperature);
	const double planck_high = IntegrateNormalizedPlanck(energy_grid[g+1], temperature);
	return planck_high - planck_low;
}

void PlanckSpectrumMGCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	v.SetSize(vdim);
	const double temperature = T.Eval(trans, ip);
	EvalPlanckSpectrum(energy_grid, temperature, v);
}

RosselandSpectrumMGCoefficient::RosselandSpectrumMGCoefficient(
	const mfem::Array<double> &grid, mfem::Coefficient &T)
	: energy_grid(grid), T(T), MultiGroupCoefficient(grid.Size()-1)
{
}

double RosselandSpectrumMGCoefficient::Eval(
	int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	const double temperature = T.Eval(trans, ip);
	const double ross_low = IntegrateNormalizedRosseland(energy_grid[g], temperature);
	const double ross_high = IntegrateNormalizedRosseland(energy_grid[g+1], temperature);
	return (ross_high - ross_low);
}

void RosselandSpectrumMGCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	v.SetSize(vdim);
	const double temperature = T.Eval(trans, ip);
	EvalRosselandSpectrum(energy_grid, temperature, v);
}