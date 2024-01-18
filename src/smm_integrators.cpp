#include "smm_integrators.hpp"

void WeakTensorDivergenceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::ElementTransformation &trans, mfem::Vector &elvec)
{
	int dof = el.GetDof(); 
	const auto dim = el.GetDim(); 
	gshape.SetSize(dof,dim); 
	gradT.SetSize(dof); 
	T.SetSize(dim); 
	T_flat.SetSize(dim*dim); 
	elvec.SetSize(dof*dim); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetIntPoint(&ip); 

		el.CalcPhysDShape(trans, gshape); 
		Tcoef.Eval(T, trans, ip); 
		T.GradToDiv(T_flat); 

		for (auto d=0; d<dim; d++) {
			mfem::Vector T_flat_ref(T_flat, d*dim, dim); 
			gshape.Mult(T_flat_ref, gradT); 
			gradT *= ip.weight * trans.Weight(); 
			elvec.AddSubVector(gradT, d*dof); 
		}
	}
}