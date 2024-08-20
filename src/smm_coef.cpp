#include "smm_coef.hpp"

SMMCorrectionTensorCoefficient::SMMCorrectionTensorCoefficient(
	mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi)
	: fes(_fes), quad(_quad), psi(_psi), mfem::MatrixArrayCoefficient(_fes.GetMesh()->Dimension())
{
	const auto dim = height; 
	gfs.SetSize(dim*dim); 
	for (auto i=0; i<dim*dim; i++) {
		gfs[i] = new mfem::ParGridFunction(&fes); 
		(*gfs[i]) = 0.0; 
	}

	mfem::DenseMatrix OmegaOmega(dim); 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		mfem::MultVVt(Omega, OmegaOmega); 
		for (auto d=0; d<dim; d++) {
			OmegaOmega(d,d) -= 1./3; 
		}
		OmegaOmega *= quad.GetWeight(a); 
		for (auto i=0; i<psi.extent(2); i++) {
			auto psi_local = psi(0,a,i); 
			for (auto d=0; d<dim; d++) {
				for (auto e=0; e<dim; e++) {
					auto idx = d + e*dim; 
					(*gfs[idx])(i) += OmegaOmega(d,e) * psi_local; 
				}
			}
		}
	}

	for (auto i=0; i<dim; i++) {
		for (auto j=0; j<dim; j++) {
			gfs[i*dim + j]->ExchangeFaceNbrData(); 
			Set(i,j, new mfem::GridFunctionCoefficient(gfs[i*dim+j]), true); 
		}
	}
}

SMMCorrectionTensorCoefficient::~SMMCorrectionTensorCoefficient() 
{
	for (auto i=0; i<gfs.Size(); i++) { delete gfs[i]; }
}

SecondMomentTensorCoefficient::SecondMomentTensorCoefficient(
	mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi)
	: fes(_fes), quad(_quad), psi(_psi), mfem::MatrixArrayCoefficient(_fes.GetMesh()->Dimension())
{
	const auto dim = height; 
	gfs.SetSize(dim*dim); 
	for (auto i=0; i<dim*dim; i++) {
		gfs[i] = new mfem::ParGridFunction(&fes); 
		(*gfs[i]) = 0.0; 
	}

	mfem::DenseMatrix OmegaOmega(dim); 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		mfem::MultVVt(Omega, OmegaOmega); 
		OmegaOmega *= quad.GetWeight(a); 
		for (auto i=0; i<psi.extent(2); i++) {
			auto psi_local = psi(0,a,i); 
			for (auto d=0; d<dim; d++) {
				for (auto e=0; e<dim; e++) {
					auto idx = d + e*dim; 
					(*gfs[idx])(i) += OmegaOmega(d,e) * psi_local; 
				}
			}
		}
	}

	for (auto i=0; i<dim; i++) {
		for (auto j=0; j<dim; j++) {
			gfs[i*dim + j]->ExchangeFaceNbrData(); 
			Set(i,j, new mfem::GridFunctionCoefficient(gfs[i*dim+j]), true); 
		}
	}
}

SecondMomentTensorCoefficient::~SecondMomentTensorCoefficient() 
{
	for (auto i=0; i<gfs.Size(); i++) { delete gfs[i]; }
}

void MatrixDivergenceGridFunctionCoefficient::Eval(mfem::Vector &v, 
	mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	mfem::Vector g; 
	for (int i=0; i<vdim; i++) {
		for (int j=0; j<vdim; j++) {
			const auto *gf = dynamic_cast<mfem::GridFunctionCoefficient*>(T.GetCoeff(i,j))->GetGridFunction();
			mfem::GradientGridFunctionCoefficient grad_coef(gf); 
			grad.GetColumnReference(i*vdim + j, g); 
			grad_coef.Eval(g, trans, ip); 
		}
	}

	v.SetSize(vdim); 
	v = 0.0; 
	for (auto i=0; i<vdim; i++) {
		for (auto j=0; j<vdim; j++) {
			const auto idx = j + i*vdim; 
			v(i) += grad(j, idx); 
		}
	}
}

SMMBdrCorrectionFactorCoefficient::SMMBdrCorrectionFactorCoefficient(
	mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
	ConstTransportVectorView _psi, double _alpha)
	: fes(_fes), quad(_quad), psi(_psi), alpha(_alpha)
{
	dim = fes.GetMesh()->Dimension(); 
	nor.SetSize(dim); 
}

double SMMBdrCorrectionFactorCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
{
	auto *ftrans = dynamic_cast<mfem::FaceElementTransformations*>(&trans); 
	if (!ftrans) { MFEM_ABORT("must call on face"); }
	ftrans->SetAllIntPoints(&ip); 
	if (dim==1) {
		nor(0) = 2*ftrans->GetElement1IntPoint().x - 1.0;
	} else {
		mfem::CalcOrtho(ftrans->Jacobian(), nor); 
	}

	nor.Set(1./nor.Norml2(), nor); 	

	mfem::Array<int> dofs; 
	fes.GetElementDofs(ftrans->Elem1No, dofs); 
	mfem::Vector psi_local(dofs.Size()), shape(dofs.Size()); 
	const auto &el = *fes.GetFE(ftrans->Elem1No); 
	const auto &eip = ftrans->GetElement1IntPoint(); 
	el.CalcShape(eip, shape); 
	double beta = 0.0; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		for (auto i=0; i<dofs.Size(); i++) { psi_local(i) = psi(0,a,dofs[i]); }
		double psi_at_ip = psi_local * shape; 
		beta += (std::fabs(Omega*nor) - alpha) * psi_at_ip * quad.GetWeight(a); 
	}
	return beta; 
}

double GrayInflowPartialCurrentCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	assert(trans.ElementType == mfem::ElementTransformation::BDR_FACE); 
	auto *ftrans = dynamic_cast<mfem::FaceElementTransformations*>(&trans); 
	assert(ftrans); 
	ftrans->SetAllIntPoints(&ip); 
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	if (dim==1) {
		nor(0) = 2*ftrans->GetElement1IntPoint().x - 1.0;
	} else {
		CalcOrtho(ftrans->Jacobian(), nor); 				
	}
	nor.Set(1./nor.Norml2(), nor); 

	double Jin = 0.0; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		phase_coef.SetAngle(Omega); 
		double psi_in = phase_coef.Eval(trans, ip); 
		double dot = Omega*nor; 
		if (dot <= 0) {
			Jin += dot * psi_in * quad.GetWeight(a); 
		}
	}
	return Jin * scale; 
}

void InflowPartialCurrentCoefficient::Eval(mfem::Vector &v, 
	mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	assert(trans.ElementType == mfem::ElementTransformation::BDR_FACE); 
	auto *ftrans = dynamic_cast<mfem::FaceElementTransformations*>(&trans); 
	assert(ftrans); 
	ftrans->SetAllIntPoints(&ip); 
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	if (dim==1) {
		nor(0) = 2*ftrans->GetElement1IntPoint().x - 1.0;
	} else {
		CalcOrtho(ftrans->Jacobian(), nor); 				
	}
	nor.Set(1./nor.Norml2(), nor); 

	v.SetSize(vdim);
	v = 0.0;
	for (int g=0; g<vdim; g++) {
		phase_coef.SetEnergy(energy_grid.LowerBound(g), energy_grid.UpperBound(g), energy_grid.MeanEnergy(g));
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			phase_coef.SetAngle(Omega); 
			double psi_in = phase_coef.Eval(trans, ip); 
			double dot = Omega*nor; 
			if (dot <= 0) {
				v(g) += dot * psi_in * quad.GetWeight(a); 
			}
		}
		v(g) *= scale;
	}
}

void OpacityCorrectionCoefficient::Eval(mfem::Vector &v, 
	mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	v.SetSize(vdim);
	const auto gray_eval = gray.Eval(trans, ip);
	mg.Eval(mg_eval, trans, ip);
	F.Eval(Feval, trans, ip);
	for (int d=0; d<vdim; d++) {
		mfem::Vector Fd;
		Feval.GetColumnReference(d, Fd);
		v(d) = gray_eval * Fd.Sum() - (mg_eval * Fd);
	}
}

void NormalizedOpacityCorrectionCoefficient::Eval(mfem::Vector &v, 
	mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
{
	v.SetSize(vdim);
	const auto gray_eval = gray.Eval(trans, ip);
	mg.Eval(mg_eval, trans, ip);
	E.Eval(E_eval, trans, ip);
	const auto gray_E = E_eval.Sum();
	F.Eval(Feval, trans, ip);
	for (int d=0; d<vdim; d++) {
		mfem::Vector Fd;
		Feval.GetColumnReference(d, Fd);
		v(d) = ((mg_eval * Fd) - gray_eval * Fd.Sum()) / gray_E;
	}
}
