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

void VectorJumpTensorAverageLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto ndof = el.GetDof(); 
	const auto dim = el.GetDim(); 
	shape1.SetSize(ndof); 
	T.SetSize(dim); 
	Tn1.SetSize(dim); 
	nor.SetSize(dim); 
	elvec.SetSize(dim*ndof); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		ir = &mfem::IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		el.CalcShape(eip1, shape1); 

		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}

		Tcoef.Eval(T, trans.GetElement1Transformation(), eip1);
		T.Mult(nor, Tn1); 
		Tn1 *= ip.weight; 

		for (auto d=0; d<dim; d++) {
			for (auto i=0; i<ndof; i++) {
				elvec(i + d*ndof) -= shape1(i) * Tn1(d); 
			}
		}
	}
}

void VectorJumpTensorAverageLFIntegrator::AssembleRHSElementVect(
	const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto ndof1 = el1.GetDof(); 
	const auto ndof2 = el2.GetDof(); 
	const auto dim = el1.GetDim(); 
	shape1.SetSize(ndof1); 
	shape2.SetSize(ndof2); 
	T.SetSize(dim); 
	Tn1.SetSize(dim); Tn2.SetSize(dim); 
	nor.SetSize(dim); 
	elvec.SetSize(dim*(ndof1+ndof2)); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		ir = &mfem::IntRules.Get(el1.GetGeomType(), oa * std::max(el1.GetOrder(),el2.GetOrder()) + ob);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 
		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 

		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}

		Tcoef.Eval(T, trans.GetElement1Transformation(), eip1);
		T.Mult(nor, Tn1); 
		Tcoef.Eval(T, trans.GetElement2Transformation(), eip2); 
		T.Mult(nor, Tn2); 
		Tn1 += Tn2; 
		Tn1 *= 0.5 * ip.weight; 

		for (auto d=0; d<dim; d++) {
			for (auto i=0; i<ndof1; i++) {
				elvec(i + d*ndof1) -= shape1(i) * Tn1(d); 
			}
		}

		for (auto d=0; d<dim; d++) {
			for (auto i=0; i<ndof2; i++) {
				elvec(dim*ndof1 + i * d*ndof2) += shape2(i) * Tn1(d); 
			}
		}
	}
}

void BoundaryNormalFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
	mfem::Vector &elvec) 
{
	const auto ndof = el.GetDof(); 
	const auto dim = el.GetDim(); 
	shape.SetSize(ndof); 
	nor.SetSize(dim); 
	elvec.SetSize(ndof*dim); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(el.GetGeomType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const mfem::IntegrationPoint &eip = trans.GetElement1IntPoint(); 
		if (dim>1) {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}
		else {
			nor(0) = 1.0; 
		}

		double inf = inflow.Eval(*trans.Face, ip); 
		el.CalcShape(eip, shape); 
		for (int d=0; d<dim; d++) {
			for (int i=0; i<ndof; i++) {
				elvec(i + d*ndof) += shape(i) * nor(d) * ip.weight * inf; 
			}
		}
	}
}

SMMCorrectionTensorCoefficient::SMMCorrectionTensorCoefficient(
	mfem::ParFiniteElementSpace &_fes, AngularQuadrature &_quad, ConstTransportVectorView _psi)
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

SMMBdrCorrectionFactorCoefficient::SMMBdrCorrectionFactorCoefficient(
	mfem::ParFiniteElementSpace &_fes, AngularQuadrature &_quad, 
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