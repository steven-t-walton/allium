#include "cons_smm_op.hpp"
#include "smm_integrators.hpp"

ConsistentSMMSourceOperator::ConsistentSMMSourceOperator(mfem::ParFiniteElementSpace &_fes, mfem::ParFiniteElementSpace &_vfes, 
		const AngularQuadrature &_quad, const TransportVectorExtents &_psi_ext, ConstTransportVectorView source_vec, double _alpha)
	: fes(_fes), vfes(_vfes), quad(_quad), psi_ext(_psi_ext), alpha(_alpha) 
{
	offsets.SetSize(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum(); 

	width = TotalExtent(psi_ext); 
	height = offsets.Last(); 

	const auto dim = fes.GetMesh()->Dimension(); 
	Q0.SetSize(fes.GetVSize()); 
	Q0 = 0.0; 
	Q1.SetSize(vfes.GetVSize()); 
	Q1 = 0.0; 

	mfem::Array<int> dofs, vdofs; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		for (auto e=0; e<fes.GetNE(); e++) {
			fes.GetElementDofs(e, dofs); 
			vfes.GetElementVDofs(e, vdofs); 
			for (const auto &dof : dofs) {
				Q0(dof) += source_vec(0,a,dof) * quad.GetWeight(a); 
			}
			for (auto d=0; d<dim; d++) {
				for (auto i=0; i<dofs.Size(); i++) {
					Q1(vdofs[i + dofs.Size()*d]) += Omega(d) * source_vec(0,a,dofs[i]) * quad.GetWeight(a); 
				}
			}
		}
	}
	Q1 *= 3.0; 

	auto face_to_element = std::unique_ptr<const mfem::Table>(fes.GetParMesh()->GetFaceToAllElementTable()); 
	element_to_face = std::unique_ptr<const mfem::Table>(mfem::Transpose(*face_to_element)); 

	if (fes.IsVariableOrder()) { MFEM_ABORT("variable order not supported for consistent SMM"); }
	trace_coll = std::make_unique<DGTrace_FECollection>(fes.GetOrder(0), dim); 
	auto &mesh = *fes.GetParMesh(); 
	trace_fes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get()); 
	trace_vfes = std::make_unique<mfem::ParFiniteElementSpace>(&mesh, trace_coll.get(), dim); 
	beta.SetSpace(trace_fes.get());
	tensor.SetSpace(trace_vfes.get());  
}

void ConsistentSMMSourceOperator::Mult(const mfem::Vector &psi, mfem::Vector &source) const 
{
	mfem::BlockVector bv(source.GetData(), offsets); 
	ConstTransportVectorView psi_view(psi.GetData(), psi_ext); 
	SMMCorrectionTensorCoefficient T(fes, quad, psi_view);
	ProjectClosuresToFaces(fes, quad, psi_view, alpha, beta, tensor); 

	mfem::ParLinearForm fform(&fes, bv.GetBlock(1).GetData()); 
	fform.AddInteriorFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta));
	fform.AddBdrFaceIntegrator(new CSMMZerothMomentFaceLFIntegrator(beta)); 
	// fform.AddInteriorFaceIntegrator(new CSMMFaceIntegrator0(fes, quad, psi_view, alpha)); 
	// fform.AddBdrFaceIntegrator(new CSMMFaceIntegrator0(fes, quad, psi_view, alpha)); 
	fform.Assemble(); 

	mfem::ParLinearForm gform(&vfes, bv.GetBlock(0).GetData()); 
	gform.AddDomainIntegrator(new WeakTensorDivergenceLFIntegrator(T)); 
	gform.AddInteriorFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 
	gform.AddBdrFaceIntegrator(new CSMMFirstMomentFaceLFIntegrator(tensor)); 
	// gform.AddBdrFaceIntegrator(new CSMMFaceIntegrator1(fes, quad, psi_view, alpha)); 
	// gform.AddInteriorFaceIntegrator(new CSMMFaceIntegrator1(fes, quad, psi_view, alpha)); 
	gform.Assemble(); 
	gform *= 3.0; 

	fform += Q0; 
	gform += Q1; 
}

void ProjectClosuresToFaces(const mfem::ParFiniteElementSpace &fes, const AngularQuadrature &quad, ConstTransportVectorView psi, 
	double alpha, mfem::ParGridFunction &beta, mfem::ParGridFunction &tensor)
{
	auto &mesh = *fes.GetParMesh(); 
	std::unique_ptr<const mfem::Table> face_to_element(mesh.GetFaceToAllElementTable()); 
	std::unique_ptr<const mfem::Table> element_to_face(mfem::Transpose(*face_to_element));  
	const auto dim = mesh.Dimension(); 
	const auto &beta_fes = *beta.FESpace(); 
	const auto &tensor_fes = *tensor.FESpace(); 
	mfem::Vector nor(dim), shape, psi_local; 
	mfem::Array<int> psi_dof, beta_dof, tensor_dof; 
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &psi_el = *fes.GetFE(e); 
		shape.SetSize(psi_el.GetDof()); 
		psi_local.SetSize(psi_el.GetDof()); 
		fes.GetElementDofs(e, psi_dof);
		beta_fes.GetElementDofs(e, beta_dof);  
		tensor_fes.GetElementVDofs(e, tensor_dof); 
		const auto *faces = element_to_face->GetRow(e); 
		for (auto f=0; f<element_to_face->RowSize(e); f++) {
			const auto face = faces[f]; 
			const auto &info = mesh.GetFaceInformation(face); 
			mfem::FaceElementTransformations *trans; 
			if (info.IsShared()) {
				trans = mesh.GetSharedFaceTransformationsByLocalIndex(face, true); 
			} else {
				trans = mesh.GetFaceElementTransformations(face); 
			}			
			bool keep_order = e == trans->Elem1No; 
			const auto local_face_id = info.element[keep_order ? 0 : 1].local_face_id; 
			const auto &tr_el = *beta_fes.GetTraceElement(face, mesh.GetFaceGeometry(face)); 
			const auto &ir = tr_el.GetNodes(); 

			mfem::IntegrationPoint eip; 
			for (auto n=0; n<ir.GetNPoints(); n++) {
				const auto &ip = ir.IntPoint(n); 
				trans->SetAllIntPoints(&ip); 
				if (dim==1) {
					nor(0) = 2*trans->GetElement1IntPoint().x - 1.0;
				} else {
					mfem::CalcOrtho(trans->Jacobian(), nor); 				
				}
				nor.Set((keep_order ? 1.0 : -1.0)/nor.Norml2(), nor); 

				if (keep_order) {
					eip = trans->GetElement1IntPoint(); 
				} else {
					eip = trans->GetElement2IntPoint(); 
				}

				psi_el.CalcShape(eip, shape); 

				double val = 0.0, phi = 0.0, Jn = 0.0; 
				mfem::Vector ten_val(dim); 
				ten_val = 0.0; 
				for (auto a=0; a<quad.Size(); a++) {
					const auto &Omega = quad.GetOmega(a); 
					double w = quad.GetWeight(a); 
					double dot = Omega * nor; 
					for (auto i=0; i<psi_dof.Size(); i++) { psi_local[i] = psi(0,a,psi_dof[i]); }
					double psi_at_ip = shape * psi_local; 
					val += (std::fabs(Omega * nor) - alpha) * psi_at_ip * w; 
					phi += psi_at_ip * w; 
					Jn += dot * psi_at_ip * w; 
					if (dot > 0) {
						for (auto d=0; d<dim; d++) {
							ten_val(d) += Omega(d) * dot * psi_at_ip * w; 
						}
					}
				}
				beta(beta_dof[local_face_id*tr_el.GetDof() + n]) = val; 
				for (auto d=0; d<dim; d++) {
					tensor(tensor_dof[local_face_id*tr_el.GetDof()*dim + n + d*tr_el.GetDof()]) 
						= ten_val(d) - nor(d)/6*phi - nor(d)/6/alpha*Jn; 
				}
			}
		}
	}

	beta.ExchangeFaceNbrData(); 
	tensor.ExchangeFaceNbrData(); 
}

void CSMMFaceIntegrator0::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
	mfem::Vector &elvec) 
{
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	const auto dof = el.GetDof(); 
	shape1.SetSize(dof); 
	elvec.SetSize(dof); 
	elvec = 0.0; 

	mfem::Array<int> psi_dof; 
	fes.GetElementDofs(trans.Elem1No, psi_dof); 
	mfem::Vector psi_local(psi_dof.Size()); 

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
		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}
		nor.Set(1./nor.Norml2(), nor); 

		double beta = 0.0; 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			const auto w = quad.GetWeight(a); 
			const auto dot = Omega * nor; 
			for (int i=0; i<psi_dof.Size(); i++) {
				psi_local(i) = psi(0,a,psi_dof[i]); 
			}
			double psi_at_ip = shape1 * psi_local; 
			beta += (std::fabs(dot) - alpha) * psi_at_ip * w; 
		}

		elvec.Add(-beta/2*ip.weight*trans.Weight(), shape1); 
	}
}

void CSMMFaceIntegrator0::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
		mfem::FaceElementTransformations &trans, mfem::Vector &elvec) 
{
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	const auto dof1 = el1.GetDof(); 
	const auto dof2 = el2.GetDof(); 
	shape1.SetSize(dof1); 
	shape2.SetSize(dof2); 
	elvec.SetSize(dof1 + dof2); 
	elvec = 0.0; 
	mfem::Vector elvec1(elvec, 0, dof1), elvec2(elvec, dof1, dof2); 

	mfem::Array<int> psi_dof1, psi_dof2; 
	fes.GetElementDofs(trans.Elem1No, psi_dof1); 
	fes.GetElementDofs(trans.Elem2No, psi_dof2); 
	mfem::Vector psi_local1(psi_dof1.Size()), psi_local2(psi_dof2.Size()); 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * std::max(el1.GetOrder(),el2.GetOrder()) + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 
		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}
		nor.Set(1./nor.Norml2(), nor); 

		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 

		double beta1 = 0.0, beta2 = 0.0; 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			const auto w = quad.GetWeight(a); 
			const auto dot = Omega * nor; 
			for (auto i=0; i<psi_dof1.Size(); i++) { psi_local1(i) = psi(0,a,psi_dof1[i]); }
			for (auto i=0; i<psi_dof2.Size(); i++) { psi_local2(i) = psi(0,a,psi_dof2[i]); }
			double psi1 = shape1 * psi_local1; 
			double psi2 = shape2 * psi_local2; 
			beta1 += (std::fabs(dot) - alpha) * psi1 * w; 
			beta2 += (std::fabs(dot) - alpha) * psi2 * w; 
		}

		double jump = beta1 - beta2; 
		const auto w = ip.weight * trans.Weight(); 
		elvec1.Add(-jump/2 * w, shape1); 
		elvec2.Add(jump/2 * w, shape2); 
	}
}

void CSMMFaceIntegrator1::AssembleRHSElementVect(const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, 
	mfem::Vector &elvec)
{
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	const auto dof = el.GetDof(); 
	shape1.SetSize(dof); 
	elvec.SetSize(dim*dof); 
	elvec = 0.0; 

	mfem::Array<int> psi_dof; 
	fes.GetElementDofs(trans.Elem1No, psi_dof); 
	mfem::Vector psi_local(psi_dof.Size()); 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * el.GetOrder() + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	mfem::Vector Pout(dim); 
	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip = trans.GetElement1IntPoint(); 
		el.CalcShape(eip, shape1); 
		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}
		nor.Set(1./nor.Norml2(), nor); 

		Pout = 0.0; 
		double phi = 0.0; 
		double Jn = 0.0; 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			const auto w = quad.GetWeight(a); 
			const auto dot = Omega * nor; 
			for (int i=0; i<psi_dof.Size(); i++) {
				psi_local(i) = psi(0,a,psi_dof[i]); 
			}
			double psi_at_ip = shape1 * psi_local; 
			phi += psi_at_ip * w; 
			Jn += dot * psi_at_ip * w; 
			if (dot > 0) {
				for (auto d=0; d<dim; d++) {
					Pout(d) += Omega(d) * dot * psi_at_ip * w; 
				}
			}
		}

		for (auto d=0; d<dim; d++) {
			mfem::Vector elvec_d(elvec, d*dof, dof); 
			elvec_d.Add((-Pout(d) + nor(d)*phi/6 + nor(d)*Jn/6/alpha)*ip.weight*trans.Weight(), shape1); 
		}
	}
}

void CSMMFaceIntegrator1::AssembleRHSElementVect(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &trans, mfem::Vector &elvec)
{
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	const auto dof1 = el1.GetDof(); 
	const auto dof2 = el2.GetDof(); 
	shape1.SetSize(dof1); 
	shape2.SetSize(dof2); 
	elvec.SetSize(dim*(dof1 + dof2)); 
	elvec = 0.0; 
	mfem::Vector elvec1(elvec, 0, dim*dof1), elvec2(elvec, dim*dof1, dim*dof2); 

	mfem::Array<int> psi_dof1, psi_dof2; 
	fes.GetElementDofs(trans.Elem1No, psi_dof1); 
	fes.GetElementDofs(trans.Elem2No, psi_dof2); 
	mfem::Vector psi_local1(psi_dof1.Size()), psi_local2(psi_dof2.Size()); 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int intorder = oa * std::max(el1.GetOrder(),el2.GetOrder()) + ob;  
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), intorder);
	}

	mfem::Vector P1(dim), P2(dim); 
	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const auto &eip1 = trans.GetElement1IntPoint(); 
		const auto &eip2 = trans.GetElement2IntPoint(); 
		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}
		nor.Set(1./nor.Norml2(), nor); 

		el1.CalcShape(eip1, shape1); 
		el2.CalcShape(eip2, shape2); 

		P1 = 0.0; 
		P2 = 0.0; 
		double phi1 = 0.0, phi2 = 0.0; 
		double Jn1 = 0.0, Jn2 = 0.0; 
		for (auto a=0; a<quad.Size(); a++) {
			const auto &Omega = quad.GetOmega(a); 
			const auto w = quad.GetWeight(a); 
			const auto dot = Omega * nor; 
			for (auto i=0; i<psi_dof1.Size(); i++) { psi_local1(i) = psi(0,a,psi_dof1[i]); }
			for (auto i=0; i<psi_dof2.Size(); i++) { psi_local2(i) = psi(0,a,psi_dof2[i]); }
			double psi1 = shape1 * psi_local1; 
			double psi2 = shape2 * psi_local2; 
			phi1 += psi1 * w; 
			phi2 += psi2 * w; 
			Jn1 += dot * psi1 * w; 
			Jn2 += dot * psi2 * w; 
			if (dot > 0) {
				for (auto d=0; d<dim; d++) {
					P1(d) += Omega(d) * dot * psi1 * w; 
				}
			} else {
				for (auto d=0; d<dim; d++) {
					P2(d) += Omega(d) * dot * psi2 * w; 
				}
			}
		}

		P1 += P2; 
		phi1 += phi2; 
		Jn1 -= Jn2; 

		const auto w = ip.weight * trans.Weight(); 
		for (auto d=0; d<dim; d++) {
			mfem::Vector elvec1_d(elvec1, d*dof1, dof1), elvec2_d(elvec2, d*dof2, dof2); 
			auto val_d = -P1(d) + nor(d)*phi1/6 + nor(d)*Jn1/6/alpha; 
			elvec1_d.Add(val_d * w, shape1); 
			elvec2_d.Add(-val_d * w, shape2); 
		}
	}
}