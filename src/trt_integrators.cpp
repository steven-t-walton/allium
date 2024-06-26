#include "trt_integrators.hpp"
#include "lumping.hpp"
#include "constants.hpp"
#include "planck.hpp"

void BlackBodyEmissionNFI::AssembleElementVector(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Vector &elvec) 
{
	const auto dof = el.GetDof(); 

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	elvec.SetSize(dof); 
	shape.SetSize(dof); 
	elvec = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		double T = elfun * shape; // interpolate T 
		double B_at_ip = sigma.Eval(trans, ip) * constants::StefanBoltzmann * pow(T, 4); 
		elvec.Add(B_at_ip * ip.weight * trans.Weight(), shape); 
	}
}

void BlackBodyEmissionNFI::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::DenseMatrix &elmat) 
{
	const auto dof = el.GetDof(); 

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	shape.SetSize(dof); 
	elmat.SetSize(dof); 
	elmat = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		double T = shape * elfun; 
		double dB_at_ip = sigma.Eval(trans, ip) * 4.0 * constants::StefanBoltzmann * pow(T, 3); 
		AddMult_a_VVt(dB_at_ip * ip.weight * trans.Weight(), shape, elmat); 
	}
}

void PlanckEmissionNFI::AssembleElementVector(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Vector &elvec)
{
	const auto dof = el.GetDof(); 
	const auto G = group_bnds.Size() - 1;

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	elvec.SetSize(dof*G); 
	shape.SetSize(dof); 
	bg_shape.SetSize(dof);
	elvec = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		const double T = elfun * shape; // interpolate T 
		const double w = trans.Weight() * ip.weight * constants::StefanBoltzmann * pow(T,4);
		EvalPlanckSpectrum(group_bnds, T, spectrum);
		sigma_coef.Eval(sigma_g, trans, ip);
		for (int g=0; g<G; g++) {
			bg_shape.Set(w * spectrum(g) * sigma_g(g), shape);
			elvec.AddSubVector(bg_shape, g*dof);
		}
	}
}

void PlanckEmissionNFI::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
	const auto dof = el.GetDof(); 
	const auto G = group_bnds.Size() - 1;

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	shape.SetSize(dof); 
	elmat.SetSize(dof*G); 
	elmat = 0.0; 
	elmat_g.SetSize(dof);

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n);
		trans.SetIntPoint(&ip);
		el.CalcShape(ip, shape);
		const double T = elfun * shape; 
		const double w = trans.Weight() * ip.weight * 4.0 * constants::StefanBoltzmann * pow(T, 3);
		EvalRosselandSpectrum(group_bnds, T, spectrum);
		sigma_coef.Eval(sigma_g, trans, ip);
		MultVVt(shape, elmat_g);
		for (int g=0; g<G; g++) {
			elmat.AddMatrix(sigma_g(g) * spectrum(g) * w, elmat_g, dof*g, dof*g);
		}
	}
}

void PlanckEmissionNFI::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Array<mfem::DenseMatrix*> &elmats)
{
	const auto dof = el.GetDof(); 
	const auto G = group_bnds.Size() - 1;

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	shape.SetSize(dof); 
	for (int g=0; g<G; g++) {
		elmats[g]->SetSize(dof);
		*elmats[g] = 0.0;
	}
	elmat_g.SetSize(dof);

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n);
		trans.SetIntPoint(&ip);
		el.CalcShape(ip, shape);
		const double T = elfun * shape; 
		const double w = trans.Weight() * ip.weight * 4.0 * constants::StefanBoltzmann * pow(T, 3);
		EvalRosselandSpectrum(group_bnds, T, spectrum);
		sigma_coef.Eval(sigma_g, trans, ip);
		MultVVt(shape, elmat_g);
		for (int g=0; g<G; g++) {
			elmats[g]->Add(sigma_g(g) * spectrum(g) * w, elmat_g);
		}
	}
}

void GrayPlanckEmissionNFI::AssembleElementVector(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Vector &elvec)
{
	const auto dof = el.GetDof(); 
	const auto G = group_bnds.Size() - 1;

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	elvec.SetSize(dof); 
	shape.SetSize(dof); 
	bg_shape.SetSize(dof);
	elvec = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		const double T = elfun * shape; // interpolate T 
		const double w = trans.Weight() * ip.weight * constants::StefanBoltzmann * pow(T,4);
		EvalPlanckSpectrum(group_bnds, T, spectrum);
		sigma_coef.Eval(sigma_g, trans, ip);
		elvec.Add((spectrum * sigma_g) * w, shape);
	}
}

void GrayPlanckEmissionNFI::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::DenseMatrix &elmat)
{
	const auto dof = el.GetDof(); 

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa*el.GetOrder() + ob); 
	}

	shape.SetSize(dof); 
	elmat.SetSize(dof); 
	elmat = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		double T = shape * elfun; 
		sigma_coef.Eval(sigma_g, trans, ip);
		EvalRosselandSpectrum(group_bnds, T, spectrum);
		const double gray_dBg = (sigma_g * spectrum) * 4.0 * constants::StefanBoltzmann * pow(T, 3);
		AddMult_a_VVt(gray_dBg * ip.weight * trans.Weight(), shape, elmat);
	}
}

PlanckEmissionNonlinearForm::PlanckEmissionNonlinearForm(const mfem::FiniteElementSpace &fes, 
	const MomentVectorExtents &phi_ext, PlanckEmissionNFI &nfi, bool lump)
	: fes(fes), phi_ext(phi_ext), nfi(nfi), lump(lump)
{
	height = TotalExtent(phi_ext);
	width = fes.GetVSize();

	row_offsets.SetSize(phi_ext.extent(MomentIndex::ENERGY)+1);
	row_offsets[0] = 0;
	for (int g=1; g<row_offsets.Size(); g++) {
		row_offsets[g] = phi_ext.extent(MomentIndex::SPACE);
	}
	row_offsets.PartialSum();

	col_offsets.SetSize(2);
	col_offsets[0] = 0; 
	col_offsets[1] = fes.GetVSize();
}

void PlanckEmissionNonlinearForm::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	mfem::Array<int> vdofs;
	mfem::Vector el_x, el_y;
	const mfem::FiniteElement *fe;
	mfem::ElementTransformation *T;
	auto &mesh = *fes.GetMesh();

	y = 0.0;
	auto y_view = MomentVectorView(y.GetData(), phi_ext);

	for (int e=0; e<fes.GetNE(); e++) {
		fe = fes.GetFE(e);
		T = mesh.GetElementTransformation(e);
		fes.GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, el_x);
		LumpedIntegrationRule rule(T->GetGeometryType()); 
		if (lump)
			nfi.SetIntegrationRule(rule); 
		nfi.AssembleElementVector(*fe, *T, el_x, el_y);
		const auto dof = fe->GetDof(); 
		for (int g=0; g<G; g++) {
			for (int i=0; i<dof; i++) {
				y_view(g, 0, vdofs[i]) += el_y(i + g*dof);
			}			
		}
	}
}

mfem::BlockOperator &PlanckEmissionNonlinearForm::GetGradient(const mfem::Vector &x) const 
{
	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	mfem::Array<int> vdofs;
	mfem::Vector el_x, el_y;
	const mfem::FiniteElement *fe;
	mfem::ElementTransformation *T;
	auto &mesh = *fes.GetMesh();

	mfem::Array<mfem::SparseMatrix*> spmats(G);
	mfem::Array<mfem::DenseMatrix*> elmats(G);
	for (int g=0; g<G; g++) {
		elmats[g] = new mfem::DenseMatrix;
		spmats[g] = new mfem::SparseMatrix(fes.GetVSize());
	}

	for (int e=0; e<fes.GetNE(); e++) {
		fe = fes.GetFE(e);
		T = mesh.GetElementTransformation(e);
		fes.GetElementVDofs(e, vdofs);
		x.GetSubVector(vdofs, el_x);
		LumpedIntegrationRule rule(T->GetGeometryType()); 
		if (lump)
			nfi.SetIntegrationRule(rule); 
		nfi.AssembleElementGrad(*fe, *T, el_x, elmats);	
		for (int g=0; g<G; g++) {
			spmats[g]->AddSubMatrix(vdofs, vdofs, *elmats[g]);
		}
	}

	delete grad;
	grad = new mfem::BlockOperator(row_offsets, col_offsets);
	for (int g=0; g<G; g++) {
		delete elmats[g];
		grad->SetBlock(g, 0, spmats[g]);
	}
	grad->owns_blocks = 1;

	return *grad;
}