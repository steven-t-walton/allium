#include "moment_integrators.hpp"

void MIPDiffusionIntegrator::AssembleFaceMatrix(
	const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
	mfem::FaceElementTransformations &Trans, mfem::DenseMatrix &elmat)
{
	using namespace mfem; 
	int dim, ndof1, ndof2, ndofs;
	bool kappa_is_nonzero = (kappa != 0.);
	double w, wq = 0.0;

	dim = el1.GetDim();
	ndof1 = el1.GetDof();

	nor.SetSize(dim);
	nh.SetSize(dim);
	ni.SetSize(dim);
	adjJ.SetSize(dim);
	if (MQ) {
		mq.SetSize(dim);
	}

	shape1.SetSize(ndof1);
	dshape1.SetSize(ndof1, dim);
	dshape1dn.SetSize(ndof1);
	if (Trans.Elem2No >= 0) {
		ndof2 = el2.GetDof();
		shape2.SetSize(ndof2);
		dshape2.SetSize(ndof2, dim);
		dshape2dn.SetSize(ndof2);
	}
	else {
		ndof2 = 0;
	}

	ndofs = ndof1 + ndof2;
	elmat.SetSize(ndofs);
	elmat = 0.0;
	if (kappa_is_nonzero) {
		jmat.SetSize(ndofs);
		jmat = 0.;
	}

	const IntegrationRule *ir = IntRule;
	if (ir == NULL) {
		// a simple choice for the integration order; is this OK?
		int order;
		if (ndof2) {
			order = 2*std::max(el1.GetOrder(), el2.GetOrder());
		}
		else {
			order = 2*el1.GetOrder();
		}
		ir = &IntRules.Get(Trans.GetGeometryType(), order);
	}

	// assemble: < {(Q \nabla u).n},[v] >      --> elmat
	//           kappa < {h^{-1} Q} [u],[v] >  --> jmat
	for (int p = 0; p < ir->GetNPoints(); p++) {
		const IntegrationPoint &ip = ir->IntPoint(p);

		// Set the integration point in the face and the neighboring elements
		Trans.SetAllIntPoints(&ip);

		// Access the neighboring elements' integration points
		// Note: eip2 will only contain valid data if Elem2 exists
		const IntegrationPoint &eip1 = Trans.GetElement1IntPoint();
		const IntegrationPoint &eip2 = Trans.GetElement2IntPoint();

		if (dim == 1) {
			nor(0) = 2*eip1.x - 1.0;
		}
		else {
			CalcOrtho(Trans.Jacobian(), nor);
		}

		el1.CalcShape(eip1, shape1);
		el1.CalcDShape(eip1, dshape1);
		w = ip.weight/Trans.Elem1->Weight();
		if (ndof2) {
			w /= 2;
		}
		if (!MQ) {
			if (Q) {
				w *= Q->Eval(*Trans.Elem1, eip1);
			}
			ni.Set(w, nor);
		}
		else {
			nh.Set(w, nor);
			MQ->Eval(mq, *Trans.Elem1, eip1);
			mq.MultTranspose(nh, ni);
		}
		CalcAdjugate(Trans.Elem1->Jacobian(), adjJ);
		adjJ.Mult(ni, nh);
		if (kappa_is_nonzero) {
			wq = ni * nor;
		}
		// Note: in the jump term, we use 1/h1 = |nor|/det(J1) which is
		// independent of Loc1 and always gives the size of element 1 in
		// direction perpendicular to the face. Indeed, for linear transformation
		//     |nor|=measure(face)/measure(ref. face),
		//   det(J1)=measure(element)/measure(ref. element),
		// and the ratios measure(ref. element)/measure(ref. face) are
		// compatible for all element/face pairs.
		// For example: meas(ref. tetrahedron)/meas(ref. triangle) = 1/3, and
		// for any tetrahedron vol(tet)=(1/3)*height*area(base).
		// For interior faces: q_e/h_e=(q1/h1+q2/h2)/2.

		dshape1.Mult(nh, dshape1dn);
		for (int i = 0; i < ndof1; i++)
			for (int j = 0; j < ndof1; j++) {
				elmat(i, j) += shape1(i) * dshape1dn(j);
			}

		if (ndof2) {
			el2.CalcShape(eip2, shape2);
			el2.CalcDShape(eip2, dshape2);
			w = ip.weight/2/Trans.Elem2->Weight();
			if (!MQ) {
				if (Q) {
					w *= Q->Eval(*Trans.Elem2, eip2);
				}
				ni.Set(w, nor);
			}
			else {
				nh.Set(w, nor);
				MQ->Eval(mq, *Trans.Elem2, eip2);
				mq.MultTranspose(nh, ni);
			}
			CalcAdjugate(Trans.Elem2->Jacobian(), adjJ);
			adjJ.Mult(ni, nh);
			if (kappa_is_nonzero) {
				wq += ni * nor;
			}

			dshape2.Mult(nh, dshape2dn);

			for (int i = 0; i < ndof1; i++)
				for (int j = 0; j < ndof2; j++) {
					elmat(i, ndof1 + j) += shape1(i) * dshape2dn(j);
				}

			for (int i = 0; i < ndof2; i++)
				for (int j = 0; j < ndof1; j++) {
					elmat(ndof1 + i, j) -= shape2(i) * dshape1dn(j);
				}

			for (int i = 0; i < ndof2; i++)
				for (int j = 0; j < ndof2; j++) {
					elmat(ndof1 + i, ndof1 + j) -= shape2(i) * dshape2dn(j);
				}
		}

		if (kappa_is_nonzero) {
			// only assemble the lower triangular part of jmat
			wq *= kappa;
			if (wq < alpha * ip.weight * Trans.Weight()) 
				wq = alpha * ip.weight * Trans.Weight(); 

			for (int i = 0; i < ndof1; i++) {
				const double wsi = wq*shape1(i);
				for (int j = 0; j <= i; j++)
				{
					jmat(i, j) += wsi * shape1(j);
				}
			}
			if (ndof2) {
				for (int i = 0; i < ndof2; i++) {
					const int i2 = ndof1 + i;
					const double wsi = wq*shape2(i);
					for (int j = 0; j < ndof1; j++) {
						jmat(i2, j) -= wsi * shape1(j);
					}
					for (int j = 0; j <= i; j++) {
						jmat(i2, ndof1 + j) += wsi * shape2(j);
					}
				}
			}
		}
	}

	// elmat := -elmat + sigma*elmat^t + jmat
	if (kappa_is_nonzero) {
		for (int i = 0; i < ndofs; i++) {
			for (int j = 0; j < i; j++) {
				double aij = elmat(i,j), aji = elmat(j,i), mij = jmat(i,j);
				elmat(i,j) = sigma*aji - aij + mij;
				elmat(j,i) = sigma*aij - aji + mij;
			}
			elmat(i,i) = (sigma - 1.)*elmat(i,i) + jmat(i,i);
		}
	}
	else {
		for (int i = 0; i < ndofs; i++) {
			for (int j = 0; j < i; j++) {
				double aij = elmat(i,j), aji = elmat(j,i);
				elmat(i,j) = sigma*aji - aij;
				elmat(j,i) = sigma*aij - aji;
			}
			elmat(i,i) *= (sigma - 1.);
		}
	}
}

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
	if (ir == NULL) {
		// a simple choice for the integration order; is this OK?
		int order;
		if (ndof2) {
		   order = 2*std::max(el1.GetOrder(), el2.GetOrder());
		}
		else {
		   order = 2*el1.GetOrder();
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const mfem::IntegrationPoint &ip = ir->IntPoint(n); 
		trans.SetAllIntPoints(&ip); 
		const mfem::IntegrationPoint &eip1 = trans.GetElement1IntPoint(); 
		const mfem::IntegrationPoint &eip2 = trans.GetElement2IntPoint(); 

		double Davg = 1.0; 
		if (D) {
			double D1 = D->Eval(*trans.Elem1, eip1); 
			double D2 = D->Eval(*trans.Elem2, eip2); 
			Davg = (D1 + D2)/2; 
		}

		double w; 
		if (scale) {
			double val = Davg * kappa * trans.Weight() / trans.Elem1->Weight(); 
			if (val < limit) val = limit; 
			w = ip.weight * trans.Weight() * val; 
		}
		else {
			w = Davg * kappa * ip.weight * trans.Weight(); 
		}
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

void DGJumpAverageIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1, const mfem::FiniteElement &te_fe1,
	const mfem::FiniteElement &tr_fe2, const mfem::FiniteElement &te_fe2, 
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

void DGVectorJumpAverageIntegrator::AssembleFaceMatrix(
	const mfem::FiniteElement &tr_fe1, const mfem::FiniteElement &te_fe1,
	const mfem::FiniteElement &tr_fe2, const mfem::FiniteElement &te_fe2, 
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

	elmat.SetSize(dim*te_ndofs, tr_ndofs); 
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

		// [v.n] { phi }
		for (int d=0; d<dim; d++) {
			for (int i=0; i<te_ndof1; i++) {
				for (int j=0; j<tr_ndof1; j++) {
					elmat(i+d*te_ndof1,j) += te_shape1(i) * nor(d) * tr_shape1(j) * w; 
				}
			}
		}

		if (te_ndof2) {
			te_fe2.CalcShape(eip2, te_shape2); 
			tr_fe2.CalcShape(eip2, tr_shape2); 

			for (int d=0; d<dim; d++) {
				for (int i=0; i<te_ndof1; i++) {
					for (int j=0; j<tr_ndof2; j++) {
						elmat(i+d*te_ndof1, tr_ndof1+j) += te_shape1(i) * nor(d) * tr_shape2(j) * w; 
					}
				}
			}

			for (int d=0; d<dim; d++) {
				for (int i=0; i<te_ndof2; i++) {
					for (int j=0; j<tr_ndof1; j++) {
						elmat(i+dim*te_ndof1+d*te_ndof2, j) -= te_shape2(i) * nor(d) * tr_shape1(j) * w; 
					}
				}
			}

			for (int d=0; d<dim; d++) {
				for (int i=0; i<te_ndof2; i++) {
					for (int j=0; j<tr_ndof2; j++) {
						elmat(i+dim*te_ndof1+d*te_ndof2, j+tr_ndof1) -= te_shape2(i) * nor(d) * tr_shape2(j) * w; 
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
	elmat *= beta; 
}

void LDGTraceIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1,
	const mfem::FiniteElement &te_fe1,
	const mfem::FiniteElement &tr_fe2, 
	const mfem::FiniteElement &te_fe2,
	mfem::FaceElementTransformations &T, 
	mfem::DenseMatrix &elmat)
{
	using namespace mfem;
   int dim = tr_fe1.GetDim();
   int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2, tr_ndofs, te_ndofs;
   tr_ndof1 = tr_fe1.GetDof();
   te_ndof1 = te_fe1.GetDof();

   if (T.Elem2No >= 0)
   {
      tr_ndof2 = tr_fe2.GetDof();
      te_ndof2 = te_fe2.GetDof();
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

   Vector ortho(dim), nor(dim);
   Vector tr_s1(tr_ndof1);
   Vector tr_s2(tr_ndof2);
   Vector te_s1(te_ndof1);
   Vector te_s2(te_ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      if (tr_ndof2)
      {
         order = std::max(tr_fe1.GetOrder(), tr_fe2.GetOrder()) + std::max(te_fe1.GetOrder(),
                                                                 te_fe2.GetOrder());
      }
      else
      {
         order = tr_fe1.GetOrder() + te_fe1.GetOrder();
      }
      ir = &IntRules.Get(T.GetGeometryType(), order);
   }

   DenseMatrix A11(te_ndof1, tr_ndof1);
   DenseMatrix A12(te_ndof1, tr_ndof2);
   DenseMatrix A21(te_ndof2, tr_ndof1);
   DenseMatrix A22(te_ndof2, tr_ndof2);
   double w, sign;
   for (int n=0; n<ir->GetNPoints(); n++)
   {
      const IntegrationPoint &ip = ir->IntPoint(n);
      T.SetAllIntPoints(&ip);
      const IntegrationPoint &eip1 = T.GetElement1IntPoint();
      const IntegrationPoint &eip2 = T.GetElement2IntPoint();
      if (dim==1) {
         ortho(0) = 2*eip1.x - 1.0;
      } else {
         CalcOrtho(T.Jacobian(), ortho);         
      }
      w = ip.weight;
      if (tr_ndof2) { w /= 2; }
      ortho *= w;
      // set sign = sign(beta . normal)
      // use factor of half in weight to get sign/2
      if (beta and tr_ndof2) { sign = (*beta * ortho >= 0 ? 1.0 : -1.0); }
      else { sign = 0.0; }

      tr_fe1.CalcShape(eip1, tr_s1);
      te_fe1.CalcShape(eip1, te_s1);
      MultVWt(te_s1, tr_s1, A11);
      for (int d=0; d<dim; d++)
      {
         elmat.AddMatrix(ortho(d)*(1.0 + sign), A11, 0, d*tr_ndof1);
      }

      if (tr_ndof2)
      {
         tr_fe2.CalcShape(eip2, tr_s2);
         te_fe2.CalcShape(eip2, te_s2);
         MultVWt(te_s1, tr_s2, A12);
         MultVWt(te_s2, tr_s1, A21);
         MultVWt(te_s2, tr_s2, A22);
         for (int d=0; d<dim; d++)
         {
            elmat.AddMatrix(ortho(d)*(1.0 - sign), A12, 0, dim*tr_ndof1 + d*tr_ndof2);
            elmat.AddMatrix(-ortho(d)*(1.0 + sign), A21, te_ndof1, d*tr_ndof1);
            elmat.AddMatrix(ortho(d)*(-1.0 + sign), A22, te_ndof1,
                            dim*tr_ndof1 + d*tr_ndof2);
         }
      }
   }
}

// void LDGTraceIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &tr_fe1,
// 	const mfem::FiniteElement &te_fe1,
// 	const mfem::FiniteElement &tr_fe2, 
// 	const mfem::FiniteElement &te_fe2,
// 	mfem::FaceElementTransformations &T, 
// 	mfem::DenseMatrix &elmat)
// {
// 	int dim = tr_fe1.GetDim();
// 	nor.SetSize(dim); 
// 	int tr_ndof1, te_ndof1, tr_ndof2, te_ndof2, tr_ndofs, te_ndofs;
// 	tr_ndof1 = tr_fe1.GetDof();
// 	te_ndof1 = te_fe1.GetDof();
// 	tr_shape1.SetSize(tr_ndof1); 
// 	te_shape1.SetSize(te_ndof1); 

// 	if (T.Elem2No >= 0)
// 	{
// 		tr_ndof2 = tr_fe2.GetDof();
// 		te_ndof2 = te_fe2.GetDof();
// 		tr_shape2.SetSize(tr_ndof2); 
// 		te_shape2.SetSize(te_ndof2); 
// 	}
// 	else
// 	{
// 		tr_ndof2 = 0;
// 		te_ndof2 = 0;
// 	}

// 	tr_ndofs = tr_ndof1 + tr_ndof2;
// 	te_ndofs = te_ndof1 + te_ndof2;
// 	elmat.SetSize(te_ndofs, tr_ndofs*dim);
// 	elmat = 0.0;

// 	const auto *ir = IntRule;
// 	if (ir == NULL)
// 	{
// 		int order;
// 		if (tr_ndof2)
// 		{
// 			order = std::max(tr_fe1.GetOrder(), tr_fe2.GetOrder()) 
// 				+ std::max(te_fe1.GetOrder(), te_fe2.GetOrder());
// 		}
// 		else
// 		{
// 			order = tr_fe1.GetOrder() + te_fe1.GetOrder();
// 		}
// 		ir = &mfem::IntRules.Get(T.GetGeometryType(), order);
// 	}

// 	A11.SetSize(te_ndof1, tr_ndof1); 
// 	A12.SetSize(te_ndof1, tr_ndof2); 
// 	A21.SetSize(tr_ndof2, te_ndof1); 
// 	A22.SetSize(te_ndof2, tr_ndof2); 

// 	double w, sign;
// 	for (int n=0; n<ir->GetNPoints(); n++)
// 	{
// 		const auto &ip = ir->IntPoint(n);
// 		T.SetAllIntPoints(&ip);
// 		const auto &eip1 = T.GetElement1IntPoint();
// 		const auto &eip2 = T.GetElement2IntPoint();
// 		if (dim == 1) {
// 			nor(0) = 2*eip1.x - 1.0;
// 		} else {
// 			mfem::CalcOrtho(T.Jacobian(), nor);			
// 		}
// 		w = ip.weight;
// 		if (tr_ndof2) { w /= 2; }
// 		// set sign = sign(beta . normal)
// 		// use factor of half in weight to get sign/2
// 		if (beta and tr_ndof2) { sign = (*beta * nor >= 0 ? 1.0 : -1.0); }
// 		else { sign = 0.0; }

// 		if (limit > 0) {
// 			double c = coef->Eval(*T.Elem1, eip1); 
// 			if (tr_ndof2) {
// 				c += coef->Eval(*T.Elem2, eip2); 
// 				c /= 2; 
// 			}
// 			double k = kappa * T.Weight() / T.Elem1->Weight() * c; 
// 			if (k < limit) {
// 				sign = 0.0; 
// 			}
// 		}

// 		tr_fe1.CalcShape(eip1, tr_shape1);
// 		te_fe1.CalcShape(eip1, te_shape1);
// 		mfem::MultVWt(te_shape1, tr_shape1, A11);
// 		for (int d=0; d<dim; d++)
// 		{
// 			elmat.AddMatrix(w*nor(d)*(1.0 + sign), A11, 0, d*tr_ndof1);
// 		}

// 		if (tr_ndof2)
// 		{
// 			tr_fe2.CalcShape(eip2, tr_shape2);
// 			te_fe2.CalcShape(eip2, te_shape2);
// 			mfem::MultVWt(te_shape1, tr_shape2, A12);
// 			mfem::MultVWt(te_shape2, tr_shape1, A21);
// 			mfem::MultVWt(te_shape2, tr_shape2, A22);
// 			for (int d=0; d<dim; d++)
// 			{
// 				elmat.AddMatrix(w*nor(d)*(1.0 - sign), A12, 0, dim*tr_ndof1 + d*tr_ndof2);
// 				elmat.AddMatrix(-w*nor(d)*(1.0 + sign), A21, te_ndof1, d*tr_ndof1);
// 				elmat.AddMatrix(w*nor(d)*(-1.0 + sign), A22, te_ndof1, dim*tr_ndof1 + d*tr_ndof2);
// 			}
// 		}
// 	}
// }

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

void VectorFEBoundaryNormalFaceLFIntegrator::AssembleRHSElementVect(
	const mfem::FiniteElement &el, mfem::FaceElementTransformations &trans, mfem::Vector &elvec)
{
	const auto ndof = el.GetDof();
	const auto dim = el.GetDim();
	vshape.SetSize(ndof, dim);
	nor.SetSize(dim);
	elvec.SetSize(ndof);
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
		if (dim > 1) {
			mfem::CalcOrtho(trans.Jacobian(), nor);
		} else {
			nor(0) = 2*trans.GetElement1IntPoint().x - 1.0;
		}
		const auto val = inflow.Eval(trans, ip);

		el.CalcVShape(trans.GetElement1Transformation(), vshape);
		vshape.AddMult(nor, elvec, ip.weight * val);
	}
}