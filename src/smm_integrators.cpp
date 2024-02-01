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
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), oa * el.GetOrder() + ob);
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

	mfem::Vector elvec1(elvec, 0, dim*ndof1); 
	mfem::Vector elvec2(elvec, dim*ndof1, dim*ndof2); 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), oa * std::max(el1.GetOrder(),el2.GetOrder()) + ob);
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
				elvec1(i + d*ndof1) -= shape1(i) * Tn1(d); 
			}
		}

		for (auto d=0; d<dim; d++) {
			for (auto i=0; i<ndof2; i++) {
				elvec2(i + d*ndof2) += shape2(i) * Tn1(d); 
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
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
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

		double inf = inflow.Eval(trans, ip); 
		el.CalcShape(eip, shape); 
		for (int d=0; d<dim; d++) {
			for (int i=0; i<ndof; i++) {
				elvec(i + d*ndof) += shape(i) * nor(d) * ip.weight * inf; 
			}
		}
	}
}

void ProjectedCoefBoundaryNormalLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec)
{
	const auto ndof = el.GetDof(); 
	const auto dim = el.GetDim(); 
	shape.SetSize(ndof); 
	nor.SetSize(dim); 
	elvec.SetSize(ndof*dim); 
	elvec = 0.0; 

	const auto &tr_el = *fec.TraceFiniteElementForGeometry(trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	Qnodes.SetSize(tr_dof); 
	tr_el.Project(Q, trans, Qnodes); 
	tr_shape.SetSize(tr_dof); 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
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

		tr_el.CalcShape(ip, tr_shape); 
		double q = tr_shape * Qnodes; 
		el.CalcShape(eip, shape); 
		for (int d=0; d<dim; d++) {
			for (int i=0; i<ndof; i++) {
				elvec(i + d*ndof) += shape(i) * nor(d) * ip.weight * q; 
			}
		}
	}
}

void ProjectedCoefBoundaryLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto dof = el.GetDof(); 
	const auto &tr_el = *fec.TraceFiniteElementForGeometry(trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	Qnodes.SetSize(tr_dof); 
	tr_el.Project(Q, trans, Qnodes); 
	tr_shape.SetSize(tr_dof); 
	shape.SetSize(dof); 
	elvec.SetSize(dof); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip = trans.GetElement1IntPoint(); 
		el.CalcShape(eip, shape); 
		tr_el.CalcShape(ip, tr_shape); 
		double val = trans.Weight() * ip.weight * (tr_shape * Qnodes); 
		elvec.Add(val, shape); 
	}
}

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

void CSMMZerothMomentFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto face_el_no = trans.mesh->GetBdrElementFaceIndex(trans.ElementNo); 
	// extract beta values at face 
	const auto &tr_fes = *beta.FESpace();
	tr_fes.GetElementDofs(trans.Elem1No, beta_dofs1);  
	beta.GetSubVector(beta_dofs1, beta_all1); 
	const auto &tr_el = *tr_fes.GetTraceElement(face_el_no, trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	auto info = trans.mesh->GetFaceInformation(face_el_no); 
	assert(info.IsBoundary()); 
	auto local_face_id = info.element[0].local_face_id; 
	beta_trace1.SetSize(tr_dof); 
	for (int i=0; i<tr_dof; i++) { beta_trace1(i) = beta_all1(local_face_id * tr_dof + i); }
	tr_shape1.SetSize(tr_dof); 

	const auto dof = el.GetDof(); 
	shape1.SetSize(dof); 
	elvec.SetSize(dof); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip = trans.GetElement1IntPoint(); 
		el.CalcShape(eip, shape1); 
		tr_el.CalcShape(ip, tr_shape1); 
		double val = trans.Weight() * ip.weight * (tr_shape1 * beta_trace1); 
		elvec.Add(-val/2, shape1); 
	}
}

void CSMMZerothMomentFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	// extract beta data for this face 
	const auto &tr_fes = *beta.ParFESpace();
	const auto &tr_el = *tr_fes.GetTraceElement(trans.ElementNo, trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	auto info = trans.mesh->GetFaceInformation(trans.ElementNo); 
	assert(trans.Elem1No < trans.mesh->GetNE()); 
	const auto nbr_el = trans.Elem2No - trans.mesh->GetNE(); 

	// get element "1" data 
	tr_fes.GetElementDofs(trans.Elem1No, beta_dofs1);  
	beta.GetSubVector(beta_dofs1, beta_all1); 
	auto local_face_id1 = info.element[0].local_face_id; 
	auto local_face_id2 = info.element[1].local_face_id; 
	beta_trace1.SetSize(tr_dof); 
	for (int i=0; i<tr_dof; i++) { beta_trace1(i) = beta_all1(local_face_id1 * tr_dof + i); }
	tr_shape1.SetSize(tr_dof); 

	// get element "2" data 
	// look into parallel data structure if shared face
	if (nbr_el >= 0) {
		tr_fes.GetFaceNbrElementVDofs(nbr_el, beta_dofs2); 
		const auto &fnbr_data = beta.FaceNbrData(); 
		fnbr_data.GetSubVector(beta_dofs2, beta_all2); 
	}

	else {
		tr_fes.GetElementDofs(trans.Elem2No, beta_dofs2); 
		beta.GetSubVector(beta_dofs2, beta_all2); 
	}
	beta_trace2.SetSize(tr_dof); 
	for (int i=0; i<tr_dof; i++) { beta_trace2(i) = beta_all2(local_face_id2 * tr_dof + i); }
	tr_shape2.SetSize(tr_dof); 		

	const auto dof1 = el1.GetDof(); 
	const auto dof2 = el2.GetDof(); 
	assert(dof1 == dof2); 
	shape1.SetSize(dof1); 
	shape2.SetSize(dof2); 
	elvec.SetSize(dof1 + dof2); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * std::max(el1.GetOrder(), el2.GetOrder()) + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	mfem::Vector elvec1(elvec, 0, dof1); 
	mfem::Vector elvec2(elvec, dof1, dof2); 
	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 
		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 
		tr_el.CalcShape(ip, tr_shape1); 
		tr_el.CalcShape(ip, tr_shape2); // <-- should there be a second trace element?
		double jump = (tr_shape1 * beta_trace1) - (tr_shape2 * beta_trace2); 
		double val = trans.Weight() * ip.weight * jump; 
		elvec1.Add(-val/2, shape1); 
		elvec2.Add(val/2, shape2); 
	}
}

void CSMMFirstMomentFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto face_el_no = trans.mesh->GetBdrElementFaceIndex(trans.ElementNo); 
	const auto dim = trans.GetSpaceDim(); 
	const auto &tr_vfes = *tensor.FESpace(); 
	tr_vfes.GetElementVDofs(trans.Elem1No, tr_vdofs1); 
	tensor.GetSubVector(tr_vdofs1, tensor_all1); 

	const auto &tr_el = *tr_vfes.GetTraceElement(face_el_no, trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	const auto info = trans.mesh->GetFaceInformation(face_el_no); 
	const auto local_face_id1 = info.element[0].local_face_id; 

	const auto dofs_per_el = tr_vdofs1.Size() / dim; 
	tensor_tr1.SetSize(dim*tr_dof); 
	for (int d=0; d<dim; d++) {
		for (int i=0; i<tr_dof; i++) {
			tensor_tr1(tr_dof*d + i) = tensor_all1(local_face_id1*tr_dof*dim + i + d*tr_dof); 
		}
	}
	tr_shape.SetSize(tr_dof); 

	const auto dof = el.GetDof(); 
	shape1.SetSize(dof); 
	elvec.SetSize(dim*dof); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip = trans.GetElement1IntPoint(); 

		el.CalcShape(eip, shape1); 
		tr_el.CalcShape(ip, tr_shape); 
		for (auto d=0; d<dim; d++) {
			mfem::Vector tensor_tr1_d(tensor_tr1, tr_dof*d, tr_dof); 
			mfem::Vector elvec_d(elvec, dof*d, dof); 
			elvec_d.Add(-(tensor_tr1_d * tr_shape)*ip.weight*trans.Weight(), shape1); 
		}
	}
}

void CSMMFirstMomentFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec)
{
	assert(trans.Elem1No < trans.mesh->GetNE()); 
	const auto nbr_el = trans.Elem2No - trans.mesh->GetNE(); 

	const auto dim = trans.GetSpaceDim(); 
	const auto &tr_vfes = *tensor.ParFESpace(); 
	const auto &tr_el = *tr_vfes.GetTraceElement(trans.ElementNo, trans.GetGeometryType()); 
	const auto tr_dof = tr_el.GetDof(); 
	const auto info = trans.mesh->GetFaceInformation(trans.ElementNo); 
	const auto local_face_id1 = info.element[0].local_face_id; 
	const auto local_face_id2 = info.element[1].local_face_id; 

	tr_vfes.GetElementVDofs(trans.Elem1No, tr_vdofs1); 
	tensor.GetSubVector(tr_vdofs1, tensor_all1); 

	if (nbr_el >= 0) {
		tr_vfes.GetFaceNbrElementVDofs(nbr_el, tr_vdofs2); 
		const auto &fnbr_data = tensor.FaceNbrData(); 
		fnbr_data.GetSubVector(tr_vdofs2, tensor_all2); 
	}

	else {
		tr_vfes.GetElementVDofs(trans.Elem2No, tr_vdofs2); 
		tensor.GetSubVector(tr_vdofs2, tensor_all2); 
	}

	const auto dofs_per_el = tr_vdofs1.Size() / dim; 
	tensor_tr1.SetSize(dim*tr_dof); 
	tensor_tr2.SetSize(dim*tr_dof); 
	for (int d=0; d<dim; d++) {
		for (int i=0; i<tr_dof; i++) {
			tensor_tr1(tr_dof*d + i) = tensor_all1(local_face_id1*tr_dof*dim + i + d*tr_dof); 
			tensor_tr2(tr_dof*d + i) = tensor_all2(local_face_id2*tr_dof*dim + i + d*tr_dof); 
		}
	}
	tr_shape.SetSize(tr_dof); 

	const auto dof1 = el1.GetDof(); 
	const auto dof2 = el2.GetDof(); 
	shape1.SetSize(dof1);
	shape2.SetSize(dof2);  
	elvec.SetSize(dim*(dof1 + dof2)); 
	mfem::Vector elvec1(elvec, 0, dim*dof1), elvec2(elvec, dim*dof1, dim*dof2); 
	elvec = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * std::max(el1.GetOrder(), el2.GetOrder()) + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 

		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 
		tr_el.CalcShape(ip, tr_shape); 
		double w = ip.weight * trans.Weight(); 
		for (auto d=0; d<dim; d++) {
			mfem::Vector tensor_tr1_d(tensor_tr1, tr_dof*d, tr_dof); 
			mfem::Vector tensor_tr2_d(tensor_tr2, tr_dof*d, tr_dof); 
			double upw_tensor_d = (tensor_tr1_d*tr_shape) - (tensor_tr2_d*tr_shape); 

			mfem::Vector elvec1_d(elvec1, d*dof1, dof1), elvec2_d(elvec2, d*dof2, dof2); 
			elvec1_d.Add(-upw_tensor_d * w, shape1); 
			elvec2_d.Add(upw_tensor_d * w, shape2); 
		}
	}
}