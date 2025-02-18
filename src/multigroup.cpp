#include "multigroup.hpp"
#include "planck.hpp"

MultiGroupEnergyGrid::MultiGroupEnergyGrid(const mfem::Array<double> &bounds)
	: bounds(bounds)
{
	midpoints.SetSize(bounds.Size()-1);
	for (int i=0; i<midpoints.Size(); i++) {
		if (bounds[i] > 0.0) {
			midpoints[i] = std::sqrt(bounds[i] * bounds[i+1]);
		} else {
			midpoints[i] = (bounds[i] + bounds[i+1])/2;
		}
	}

	widths.SetSize(Size());
	for (int g=0; g<Size(); g++) {
		widths[g] = bounds[g+1] - bounds[g];
	}
}

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
	bounds[0] = Emin; 
	bounds[1] = Emax; 
	return MultiGroupEnergyGrid(bounds);
}

MultiGroupEnergyGrid MultiGroupEnergyGrid::MakeLogSpaced(double Emin, double Emax, int G)
{
	mfem::Array<double> bounds(G+1);
	mfem::Array<double> midpoints(G);
	const double Emin_log = std::log(Emin);
	const double Emax_log = std::log(Emax);

	const double dE_log = (Emax_log - Emin_log)/G;
	bounds[0] = Emin;
	for (int g=1; g<bounds.Size(); g++) {
		const double exponent = Emin_log + g*dE_log;
		bounds[g] = std::exp(exponent);
	}		

	for (int g=0; g<bounds.Size()-1; g++) {
		midpoints[g] = std::sqrt(bounds[g] * bounds[g+1]);
	}

	return MultiGroupEnergyGrid(bounds, midpoints);
}

MultiGroupEnergyGrid MultiGroupEnergyGrid::MakeEqualSpaced(double Emin, double Emax, int G)
{
	mfem::Array<double> bounds(G+1);
	mfem::Array<double> midpoints(G);

	const double dE = (Emax - Emin)/G;
	bounds[0] = Emin; 
	for (int g=1; g<bounds.Size(); g++) {
		bounds[g] = Emin + g*dE;
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