#include "trt_integrators.hpp"
#include "lumped_intrule.hpp"
#include "constants.hpp"

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