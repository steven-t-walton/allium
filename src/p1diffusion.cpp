#include "p1diffusion.hpp"

void PenaltyIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat)
{
	int dim, ndof1, ndof2, ndofs; 
	dim = el1.GetDim(); 
	nor.SetSize(dim); 
	ndof1 = el1.GetDof(); 
	shape1.SetSize(ndof1); 

	if (trans.Elem2No >= 0) {
		ndof2 = el2.GetDof(); 
		shape2.SetSize(ndof2); 
	} 
	else {
		ndof2 = 0; 
	}

	ndofs = ndof1 + ndof2; 
	elmat.SetSize(ndofs); 
	elmat = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		// a simple choice for the integration order; is this OK?
		int order;
		if (ndof2)
		{
		   order = 2*std::max(el1.GetOrder(), el2.GetOrder());
		}
		else
		{
		   order = 2*el1.GetOrder();
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const mfem::IntegrationPoint &eip1 = trans.GetElement1IntPoint(); 
		const mfem::IntegrationPoint &eip2 = trans.GetElement2IntPoint(); 

		double w; 
		if (scale) {
			w = kappa * ip.weight * trans.Weight() * trans.Weight() / trans.Elem1->Weight(); 
		}
		else
			w = kappa * ip.weight * trans.Weight(); 
		el1.CalcShape(eip1, shape1); 
		for (int i=0; i<ndof1; i++) {
			for (int j=0; j<ndof1; j++) {
				elmat(i,j) += shape1(i) * shape1(j) * w; 
			}
		}
		if (ndof2) {
			el2.CalcShape(eip2, shape2); 
			for (int i=0; i<ndof1; i++) {
				for (int j=0; j<ndof2; j++) {
					elmat(i,j+ndof1) -= shape1(i) * shape2(j) * w; 
				}
			}
			for (int i=0; i<ndof2; i++) {
				for (int j=0; j<ndof1; j++) {
					elmat(i+ndof1,j) -= shape2(i) * shape1(j) * w; 
				}
			}
			for (int i=0; i<ndof2; i++) {
				for (int j=0; j<ndof2; j++) {
					elmat(i+ndof1, j+ndof1) += shape2(i) * shape2(j) * w; 
				}
			}
		}
	}
}

void DGJumpAverageIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1, const mfem::FiniteElement &tr_fe2,
	const mfem::FiniteElement &te_fe1, const mfem::FiniteElement &te_fe2, 
	mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat) 
{
	int dim, te_ndof1, te_ndof2, te_ndofs, tr_ndof1, tr_ndof2, tr_ndofs; 
	dim = tr_fe1.GetDim(); 
	nor.SetSize(dim); 
	tr_ndof1 = tr_fe1.GetDof(); 
	te_ndof1 = te_fe1.GetDof(); 
	tr_shape1.SetSize(tr_ndof1); 
	te_shape1.SetSize(te_ndof1); 

	if (trans.Elem2No >= 0) {
		tr_ndof2 = tr_fe2.GetDof(); 
		te_ndof2 = te_fe2.GetDof(); 
		tr_shape2.SetSize(tr_ndof2); 
		te_shape2.SetSize(te_ndof2); 
	} 
	else {
		te_ndof2 = 0; 
		tr_ndof2 = 0; 
	}

	te_ndofs = te_ndof1 + te_ndof2; 
	tr_ndofs = tr_ndof1 + tr_ndof2; 

	elmat.SetSize(te_ndofs, tr_ndofs*dim); 
	elmat = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		int order;
		if (te_ndof2)
		{
		   order = 2*std::max(te_fe1.GetOrder(), te_fe2.GetOrder());
		}
		else
		{
		   order = 2*te_fe1.GetOrder();
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const mfem::IntegrationPoint &eip1 = trans.GetElement1IntPoint(); 
		const mfem::IntegrationPoint &eip2 = trans.GetElement2IntPoint(); 	

		double w = ip.weight; 
		if (te_ndof2) w /= 2; 

		te_fe1.CalcShape(eip1, te_shape1); 
		tr_fe1.CalcShape(eip1, tr_shape1); 

		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}

		for (int i=0; i<te_ndof1; i++) {
			for (int d=0; d<dim; d++) {
				for (int j=0; j<tr_ndof1; j++) {
					elmat(i,j+d*tr_ndof1) += te_shape1(i) * tr_shape1(j) * nor(d) * w; 
				}
			}
		}

		if (te_ndof2) {
			te_fe2.CalcShape(eip2, te_shape2); 
			tr_fe2.CalcShape(eip2, tr_shape2); 

			for (int i=0; i<te_ndof1; i++) {
				for (int d=0; d<dim; d++) {
					for (int j=0; j<tr_ndof2; j++) {
						elmat(i,j+dim*tr_ndof1+d*tr_ndof2) += te_shape1(i) * tr_shape2(j) * nor(d) * w; 
					}
				}
			}

			for (int i=0; i<te_ndof2; i++) {
				for (int d=0; d<dim; d++) {
					for (int j=0; j<tr_ndof1; j++) {
						elmat(i+te_ndof1,j+d*tr_ndof1) -= te_shape2(i) * tr_shape1(j) * nor(d) * w; 
					}
				}
			}

			for (int i=0; i<te_ndof2; i++) {
				for (int d=0; d<dim; d++) {
					for (int j=0; j<tr_ndof2; j++) {
						elmat(i+te_ndof1,j+dim*tr_ndof1+d*tr_ndof2) -= te_shape2(i) * tr_shape2(j) * nor(d) * w; 
					}
				}
			}
		}
	}
	elmat *= alpha; 
}

void DGVectorJumpJumpIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2, 
	mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat)
{
	int dim, ndof1, ndof2, ndofs; 
	dim = el1.GetDim(); 
	nor.SetSize(dim); 
	ndof1 = el1.GetDof(); 
	shape1.SetSize(ndof1); 

	if (trans.Elem2No >= 0) {
		ndof2 = el2.GetDof(); 
		shape2.SetSize(ndof2); 
	} 
	else {
		ndof2 = 0; 
	}

	ndofs = ndof1 + ndof2; 
	elmat.SetSize(ndofs*dim); 
	elmat = 0.0; 

	const mfem::IntegrationRule *ir = IntRule;
	if (ir == NULL)
	{
		// a simple choice for the integration order; is this OK?
		int order;
		if (ndof2)
		{
		   order = 2*std::max(el1.GetOrder(), el2.GetOrder());
		}
		else
		{
		   order = 2*el1.GetOrder();
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
	}

	for (int n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const mfem::IntegrationPoint &eip1 = trans.GetElement1IntPoint(); 
		const mfem::IntegrationPoint &eip2 = trans.GetElement2IntPoint(); 

		if (dim==1) {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		} else {
			mfem::CalcOrtho(trans.Jacobian(), nor); 
		}		
		double w = ip.weight * trans.Weight(); 
		nor.Set(1./nor.Norml2(), nor); 

		el1.CalcShape(eip1, shape1); 
		for (int e=0; e<dim; e++) {
			for (int i=0; i<ndof1; i++) {
				for (int d=0; d<dim; d++) {
					for (int j=0; j<ndof1; j++) {
						elmat(i+e*ndof1,j+d*ndof1) += shape1(i) * nor(e) * shape1(j) * nor(d) * w; 
					}
				}
			}
		}

		if (ndof2) {
			el2.CalcShape(eip2, shape2); 
			for (int e=0; e<dim; e++) {
				for (int i=0; i<ndof1; i++) {
					for (int d=0; d<dim; d++) {
						for (int j=0; j<ndof2; j++) {
							elmat(i+e*ndof1,j+dim*ndof1+d*ndof2) -= shape1(i) * nor(e) * shape2(j) * nor(d) * w; 
						}
					}
				}
			}

			for (int e=0; e<dim; e++) {
				for (int i=0; i<ndof2; i++) {
					for (int d=0; d<dim; d++) {
						for (int j=0; j<ndof1; j++) {
							elmat(i+dim*ndof1+e*ndof2,j+d*ndof1) -= shape2(i) * nor(e) * shape1(j) * nor(d) * w; 
						}
					}
				}
			}

			for (int e=0; e<dim; e++) {
				for (int i=0; i<ndof2; i++) {
					for (int d=0; d<dim; d++) {
						for (int j=0; j<ndof2; j++) {
							elmat(i+dim*ndof1+e*ndof2,j+dim*ndof1+d*ndof2) += shape2(i) * nor(e) * shape2(j) * nor(d) * w; 
						}
					}
				}
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

mfem::BlockOperator *CreateP1DiffusionDiscretization(mfem::ParFiniteElementSpace &fes, mfem::ParFiniteElementSpace &vfes, 
		mfem::Coefficient &total, mfem::Coefficient &absorption, double alpha)
{
	mfem::Array<int> offsets(3); 
	offsets[0] = 0; 
	offsets[1] = vfes.GetVSize(); 
	offsets[2] = fes.GetVSize(); 
	offsets.PartialSum();

	mfem::ParBilinearForm Mtform(&vfes);
	mfem::ProductCoefficient total3(3.0, total); 
	Mtform.AddDomainIntegrator(new mfem::VectorMassIntegrator(total3));
	Mtform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.AddBdrFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Mtform.Assemble(); 
	Mtform.Finalize();  
	mfem::HypreParMatrix *Mt = Mtform.ParallelAssemble(); 

	mfem::ParBilinearForm Maform(&fes); 
	mfem::ConstantCoefficient alpha_c(alpha); 
	Maform.AddDomainIntegrator(new mfem::MassIntegrator(absorption)); 
	Maform.AddInteriorFaceIntegrator(new PenaltyIntegrator(alpha, false)); 
	Maform.AddBdrFaceIntegrator(new mfem::BoundaryMassIntegrator(alpha_c)); 
	Maform.Assemble(); 
	Maform.Finalize(); 
	mfem::HypreParMatrix *Ma = Maform.ParallelAssemble();

	mfem::ParMixedBilinearForm Dform(&vfes, &fes); 
	mfem::ConstantCoefficient neg_one(-1.0); 
	Dform.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::GradientIntegrator(neg_one))); 
	Dform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Dform.AddBdrFaceIntegrator(new DGJumpAverageIntegrator(0.5)); 
	Dform.Assemble(); 
	Dform.Finalize(); 
	mfem::HypreParMatrix *D = Dform.ParallelAssemble(); 
	mfem::HypreParMatrix *DT = D->Transpose(); 
	(*DT) *= -1.0; 

	auto *op = new mfem::BlockOperator(offsets); 
	op->SetBlock(0,0, Mt); 
	op->SetBlock(0,1, DT); 
	op->SetBlock(1,0, D); 
	op->SetBlock(1,1, Ma); 
	op->owns_blocks = 1; 
	return op; 
}

mfem::HypreParMatrix *BlockOperatorToMonolithic(const mfem::BlockOperator &bop) 
{
	mfem::Array2D<mfem::HypreParMatrix*> blocks(bop.NumRowBlocks(), bop.NumColBlocks()); 
	for (auto row=0; row<blocks.NumRows(); row++) {
		for (auto col=0; col<blocks.NumCols(); col++) {
			const mfem::Operator *op = &bop.GetBlock(row,col); 
			const auto *hypre_op = dynamic_cast<const mfem::HypreParMatrix*>(op); 
			if (!hypre_op) MFEM_ABORT("blocks must be HypreParMatrix"); 
			blocks(row,col) = const_cast<mfem::HypreParMatrix*>(hypre_op); 
		}
	}
	return mfem::HypreParMatrixFromBlocks(blocks); 
}