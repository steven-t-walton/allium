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
		elvec.Add(-val, shape1); 
	}
}

// at parallel interfaces, each processor sets the orientation of it's owned element to 
// zero with the orientation of "Elem2" set relative to "Elem1" 
// the data in the consistent SMM closures then has the wrong orientation on a parallel face 
// this function sets an integration point's "point matrix" so that the transformed integration points 
// undo the orientation misalignment on parallel interfaces 
// this is achieved by reording the columns of the point matrix from orientation 0 to the orientation 
// described in info.element[1] 
// NOTE: this is intended to be used to map from the reference element to a different point in the 
// reference element so this should still be useful and work correctly even on high-order meshes 
void ReorientPointMat(const mfem::Geometry::Type geom_type, const int face_orient, mfem::DenseMatrix &pm) {
	// nothing to do in 1D 
	if (geom_type == mfem::Geometry::POINT) {
		return; 
	}

	// face geometry is a segment 
	// only two orientations exists for segments => just have to swap the columns 
	if (geom_type == mfem::Geometry::SEGMENT) {
		assert(pm.Height() == 1 and pm.Width() == 2); 
		std::swap(pm(0,0), pm(0,1)); 
		return; 
	} 

	// in 3D, faces are quads or tris which have 8 and 6 possible orientations, respectively 
	// use mfem::Geometry::Constants to map from orientation 0 to the orientation in face_orient 
	const int *map;
	if (geom_type == mfem::Geometry::SQUARE) {
		assert(pm.Height() == 2 and pm.Width() == 4); 		
		map = mfem::Geometry::Constants<mfem::Geometry::SQUARE>::Orient[face_orient];
	} 

	else if (geom_type == mfem::Geometry::TRIANGLE) {
		assert(pm.Height() == 2 and pm.Width() == 3); 		
		map = mfem::Geometry::Constants<mfem::Geometry::TRIANGLE>::Orient[face_orient]; 
	}

	// permute columns corresponding to map 
	mfem::DenseMatrix pm2(pm); 
	for (auto i=0; i<pm.Width(); i++) {
		for (auto j=0; j<pm.Height(); j++) {
			pm(j,map[i]) = pm2(j,i); 				
		}
	}
}

void CSMMZerothMomentFaceLFIntegrator::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	// extract beta data for this face 
	const auto dim = trans.GetSpaceDim();
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

	mfem::IntegrationPointTransformation *iptrans = nullptr; 
	if (nbr_el >= 0 and dim>1) {
		iptrans = new mfem::IntegrationPointTransformation;
		iptrans->Transf.SetIdentityTransformation(trans.GetGeometryType()); 
		ReorientPointMat(trans.GetGeometryType(), info.element[1].orientation, iptrans->Transf.GetPointMat()); 		
	}

	const auto dof1 = el1.GetDof(); 
	const auto dof2 = el2.GetDof(); 
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
	mfem::IntegrationPoint ip2; 
	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 
		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 
		tr_el.CalcShape(ip, tr_shape1); 
		if (iptrans) iptrans->Transform(ip, ip2); 
		else ip2 = ip;
		tr_el.CalcShape(ip2, tr_shape2); // <-- should there be a second trace element?
		double jump = (tr_shape1 * beta_trace1) - (tr_shape2 * beta_trace2); 
		double val = trans.Weight() * ip.weight * jump; 
		elvec1.Add(-val, shape1); 
		elvec2.Add(val, shape2); 
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
	tr_shape1.SetSize(tr_dof); 

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
		tr_el.CalcShape(ip, tr_shape1); 
		for (auto d=0; d<dim; d++) {
			mfem::Vector tensor_tr1_d(tensor_tr1, tr_dof*d, tr_dof); 
			mfem::Vector elvec_d(elvec, dof*d, dof); 
			elvec_d.Add(-(tensor_tr1_d * tr_shape1)*ip.weight*trans.Weight(), shape1); 
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
	tr_shape1.SetSize(tr_dof); 
	tr_shape2.SetSize(tr_dof); 

	mfem::IntegrationPointTransformation *iptrans = nullptr; 
	if (nbr_el >= 0 and dim>1) {
		iptrans = new mfem::IntegrationPointTransformation;
		iptrans->Transf.SetIdentityTransformation(trans.GetGeometryType()); 
		ReorientPointMat(trans.GetGeometryType(), info.element[1].orientation, iptrans->Transf.GetPointMat()); 		
	}

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

	mfem::IntegrationPoint ip2; 
	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 

		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 
		tr_el.CalcShape(ip, tr_shape1); 
		if (iptrans) iptrans->Transform(ip, ip2);
		else ip2 = ip;
		tr_el.CalcShape(ip2, tr_shape2); 
		double w = ip.weight * trans.Weight(); 
		for (auto d=0; d<dim; d++) {
			mfem::Vector tensor_tr1_d(tensor_tr1, tr_dof*d, tr_dof); 
			mfem::Vector tensor_tr2_d(tensor_tr2, tr_dof*d, tr_dof); 
			double upw_tensor_d = (tensor_tr1_d*tr_shape1) - (tensor_tr2_d*tr_shape2); 

			mfem::Vector elvec1_d(elvec1, d*dof1, dof1), elvec2_d(elvec2, d*dof2, dof2); 
			elvec1_d.Add(-upw_tensor_d * w, shape1); 
			elvec2_d.Add(upw_tensor_d * w, shape2); 
		}
	}
}


void LDGTraceIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1,
	const mfem::FiniteElement &tr_fe2,
	const mfem::FiniteElement &te_fe1, 
	const mfem::FiniteElement &te_fe2,
	mfem::FaceElementTransformations &T, 
	mfem::DenseMatrix &elmat)
{
	int dim = tr_fe1.GetDim();
	nor.SetSize(dim); 
	int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2, tr_ndofs, te_ndofs;
	tr_ndof1 = tr_fe1.GetDof();
	te_ndof1 = te_fe1.GetDof();
	tr_shape1.SetSize(tr_ndof1); 
	te_shape1.SetSize(te_ndof1); 

	if (T.Elem2No >= 0)
	{
		tr_ndof2 = tr_fe2.GetDof();
		te_ndof2 = te_fe2.GetDof();
		tr_shape2.SetSize(tr_ndof2); 
		te_shape2.SetSize(te_ndof2); 
	}
	else
	{
		tr_ndof2 = 0;
		te_ndof2 = 0;
	}

	tr_ndofs = tr_ndof1 + tr_ndof2;
	te_ndofs = te_ndof1 + te_ndof2;
	elmat.SetSize(te_ndofs, tr_ndofs*dim);
	elmat = 0.0;

	const auto *ir = IntRule;
	if (ir == NULL)
	{
		int order;
		if (tr_ndof2)
		{
			order = std::max(tr_fe1.GetOrder(), tr_fe2.GetOrder()) 
				+ std::max(te_fe1.GetOrder(), te_fe2.GetOrder());
		}
		else
		{
			order = tr_fe1.GetOrder() + te_fe1.GetOrder();
		}
		ir = &mfem::IntRules.Get(T.GetGeometryType(), order);
	}

	A11.SetSize(te_ndof1, tr_ndof1); 
	A12.SetSize(te_ndof1, tr_ndof2); 
	A21.SetSize(tr_ndof2, te_ndof1); 
	A22.SetSize(te_ndof2, tr_ndof2); 

	double w, sign;
	for (int n=0; n<ir->GetNPoints(); n++)
	{
		const auto &ip = ir->IntPoint(n);
		T.SetAllIntPoints(&ip);
		const auto &eip1 = T.GetElement1IntPoint();
		const auto &eip2 = T.GetElement2IntPoint();
		if (dim == 1) {
			nor(0) = 2*eip1.x - 1.0;
		} else {
			mfem::CalcOrtho(T.Jacobian(), nor);			
		}
		w = ip.weight;
		if (tr_ndof2) { w /= 2; }
		// set sign = sign(beta . normal)
		// use factor of half in weight to get sign/2
		if (beta and tr_ndof2) { sign = (*beta * nor >= 0 ? 1.0 : -1.0); }
		else { sign = 0.0; }

		if (limit > 0) {
			double c = coef->Eval(*T.Elem1, eip1); 
			if (tr_ndof2) {
				c += coef->Eval(*T.Elem2, eip2); 
				c /= 2; 
			}
			double k = kappa * T.Weight() / T.Elem1->Weight() * c; 
			if (k < limit) {
				sign = 0.0; 
			}
		}

		tr_fe1.CalcShape(eip1, tr_shape1);
		te_fe1.CalcShape(eip1, te_shape1);
		mfem::MultVWt(te_shape1, tr_shape1, A11);
		for (int d=0; d<dim; d++)
		{
			elmat.AddMatrix(w*nor(d)*(1.0 + sign), A11, 0, d*tr_ndof1);
		}

		if (tr_ndof2)
		{
			tr_fe2.CalcShape(eip2, tr_shape2);
			te_fe2.CalcShape(eip2, te_shape2);
			mfem::MultVWt(te_shape1, tr_shape2, A12);
			mfem::MultVWt(te_shape2, tr_shape1, A21);
			mfem::MultVWt(te_shape2, tr_shape2, A22);
			for (int d=0; d<dim; d++)
			{
				elmat.AddMatrix(w*nor(d)*(1.0 - sign), A12, 0, dim*tr_ndof1 + d*tr_ndof2);
				elmat.AddMatrix(-w*nor(d)*(1.0 + sign), A21, te_ndof1, d*tr_ndof1);
				elmat.AddMatrix(w*nor(d)*(-1.0 + sign), A22, te_ndof1, dim*tr_ndof1 + d*tr_ndof2);
			}
		}
	}
}