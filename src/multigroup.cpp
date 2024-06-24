#include "multigroup.hpp"

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
	midpoints[0] = std::sqrt(Emin * Emax);
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

void OpacityGroupCollapseOperator::Mult(const mfem::Vector &sigma_mf, mfem::Vector &sigma_gray) const 
{
	const int G = fes.GetVDim(); 
	const auto ordering = fes.GetOrdering(); 
	assert(ordering == mfem::Ordering::Type::byNODES); 
	assert(sigma_gray.Size() == height); 
	sigma_gray = 0.0; 
	mfem::Vector F(G); 
	for (int g=0; g<G; g++) {
		F(g) = energy_grid[g+1] - energy_grid[g];
	}
	mfem::Array<int> dof; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dof); 
		auto &trans = *fes.GetMesh()->GetElementTransformation(e); 
		const auto &fe = *fes.GetFE(e); 
		const auto &ir = fe.GetNodes(); 
		for (int n=0; n<ir.Size(); n++) {
			if (f) 
				f->Eval(F, trans, ir.IntPoint(n)); 
			double denom = 0.0; 
			for (int g=0; g<G; g++) {
				double dE = energy_grid[g+1] - energy_grid[g]; 
				sigma_gray(dof[n]) += F(g) * sigma_mf(dof[n] + g*height); 
				denom += F(g); 
			}
			sigma_gray(dof[n]) /= denom; 
		}
	}
}

GroupCollapseOperator::GroupCollapseOperator(const MomentVectorExtents &mg_ext)
	: mg_ext(mg_ext), gr_ext(1, mg_ext.extent(MomentIndex::MOMENT), mg_ext.extent(MomentIndex::SPACE))
{
	// maps multigroup to single group
	width = TotalExtent(mg_ext);
	height = TotalExtent(gr_ext);
}

void GroupCollapseOperator::Mult(const mfem::Vector &mg, mfem::Vector &gray) const
{
	assert(mg.Size() == width);
	assert(gray.Size() == height);
	gray = 0.0;

	auto mg_view = MomentVectorView(mg.GetData(), mg_ext);
	auto gr_view = MomentVectorView(gray.GetData(), gr_ext);
	for (int g=0; g<mg_ext.extent(MomentIndex::ENERGY); g++) {
		for (int m=0; m<mg_ext.extent(MomentIndex::MOMENT); m++) {
			for (int s=0; s<mg_ext.extent(MomentIndex::SPACE); s++) {
				gr_view(0,m,s) += mg_view(g,m,s);
			}
		}
	}
}

void GroupCollapseOperator::MultTranspose(const mfem::Vector &gray, mfem::Vector &mg) const
{
	assert(mg.Size() == width);
	assert(gray.Size() == height);
	auto mg_view = MomentVectorView(mg.GetData(), mg_ext);
	auto gr_view = MomentVectorView(gray.GetData(), gr_ext);
	for (int g=0; g<mg_ext.extent(MomentIndex::ENERGY); g++) {
		for (int m=0; m<mg_ext.extent(MomentIndex::MOMENT); m++) {
			for (int s=0; s<mg_ext.extent(MomentIndex::SPACE); s++) {
				mg_view(g,m,s) = gr_view(0,m,s);
			}
		}
	}	
}

MomentVectorMultiGroupCoefficient::MomentVectorMultiGroupCoefficient(
	const mfem::FiniteElementSpace &fes, const MomentVectorExtents &phi_ext, const mfem::Vector &data)
	: fes(fes), phi_ext(phi_ext), data(data), mfem::VectorCoefficient(phi_ext.extent(MomentIndex::ENERGY))
{
}

void MomentVectorMultiGroupCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	auto view = MomentVectorView(data.GetData(), phi_ext);
	v.SetSize(vdim);
	mfem::Array<int> dofs;
	const auto *fe = fes.GetFE(trans.ElementNo);
	const auto dof = fe->GetDof();
	shape.SetSize(dof);
	local_data.SetSize(dof);
	fe->CalcShape(ip, shape);
	fes.GetElementDofs(trans.ElementNo, dofs);
	for (int g=0; g<vdim; g++) {
		for (int i=0; i<dof; i++) {
			local_data(i) = view(g,moment_id,dofs[i]);
		}
		v(g) = shape * local_data;
	}
}