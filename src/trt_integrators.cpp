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
	MFEM_ABORT("not implemented");
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