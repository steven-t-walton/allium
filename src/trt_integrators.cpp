#include "trt_integrators.hpp"

void EnergyBalanceNonlinearFormIntegrator::AssembleElementVector(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::Vector &elvec) 
{
	const auto dof = el.GetDof(); 
	BlackBodyEmissionCoefficient planck_emission(sigma, el, elfun); 

	const mfem::IntegrationRule *ir = IntRule; 
	if (rules) { // lump with nodal quadrature 
		ir = &rules->Get(el.GetGeomType(), el.GetOrder()); 
	} else {
		ir = &mfem::IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + sigma_fe_order); 
	}

	elvec.SetSize(dof); 
	shape.SetSize(dof); 
	elvec = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		double T = elfun * shape; // interpolate T 
		double B_at_ip = planck_emission.Eval(trans, ip); 
		elvec.Add(B_at_ip * ip.weight * trans.Weight(), shape); 
	}
}

void EnergyBalanceNonlinearFormIntegrator::AssembleElementGrad(
	const mfem::FiniteElement &el, mfem::ElementTransformation &trans, 
	const mfem::Vector &elfun, mfem::DenseMatrix &elmat) 
{
	const auto dof = el.GetDof(); 
	GradBlackBodyEmissionCoefficient grad_plank_emission(sigma, el, elfun); 

	const mfem::IntegrationRule *ir = IntRule; 
	if (rules) { // lump with nodal quadrature 
		ir = &rules->Get(el.GetGeomType(), el.GetOrder()); 
	} else {
		ir = &mfem::IntRules.Get(el.GetGeomType(), 2*el.GetOrder() + sigma_fe_order); 
	}

	shape.SetSize(dof); 
	elmat.SetSize(dof); 
	elmat = 0.0; 

	for (int n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 
		el.CalcShape(ip, shape); 
		double dB_at_ip = grad_plank_emission.Eval(trans, ip); 
		AddMult_a_VVt(dB_at_ip * ip.weight * trans.Weight(), shape, elmat); 
	}
}