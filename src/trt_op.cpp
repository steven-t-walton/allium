#include "trt_op.hpp"

SNTimeMassMatrix::SNTimeMassMatrix(const mfem::FiniteElementSpace &fes, 
	const TransportVectorExtents &ext, bool lump)
	: fes(fes), psi_ext(ext) 
{
	mats.SetSize(fes.GetNE()); 
	mfem::Array<int> dofs; 
	mfem::MassIntegrator mi; 
	for (int e=0; e<fes.GetNE(); e++) {
		fes.GetElementDofs(e, dofs); 
		mats[e] = new mfem::DenseMatrix(dofs.Size()); 
		if (lump)
			mats[e]->Lump(); 
		mi.AssembleElementMatrix(*fes.GetFE(e), *fes.GetElementTransformation(e), *mats[e]); 
	}
}

void SNTimeMassMatrix::Mult(const mfem::Vector &psi, mfem::Vector &Mpsi) const 
{
	auto psi_view = ConstTransportVectorView(psi.GetData(), psi_ext); 
	auto Mpsi_view = TransportVectorView(Mpsi.GetData(), psi_ext); 
	mfem::Array<int> dofs; 
	mfem::Vector psi_local, Mpsi_local; 
	for (int g=0; g<psi_ext.extent(0); g++) {
		for (int a=0; a<psi_ext.extent(1); a++) {
			for (int e=0; e<fes.GetNE(); e++) {
				fes.GetElementDofs(e, dofs); 
				const auto &elmat = *mats[e]; 
				psi_local.SetSize(dofs.Size()); 
				Mpsi_local.SetSize(dofs.Size()); 					
				for (int n=0; n<dofs.Size(); n++) psi_local(n) = psi_view(g,a,dofs[n]); 
				elmat.Mult(psi_local, Mpsi_local); 
				for (int n=0; n<dofs.Size(); n++) Mpsi_view(g,a,dofs[n]) = Mpsi_local(n); 
			}
		}
	}
}

void NonlinearFormBlockInverse::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const auto &dnfi = *nform.GetDNFI(); 
	const auto &fnfi = nform.GetInteriorFaceIntegrators(); 
	const auto &bfnfi = nform.GetBdrFaceIntegrators(); 
	assert(fnfi.Size()==0 and bfnfi.Size()==0); 

	auto *fes = nform.FESpace(); 
	auto *mesh = fes->GetMesh(); 

	mfem::Vector el_data, el_x; 
	const mfem::FiniteElement *fe; 
	mfem::ElementTransformation *T; 
	mfem::DofTransformation *doftrans; 
	mfem::DenseMatrix A, elmat; 
	mfem::Array<int> vdofs; 
	for (int i = 0; i < fes->GetNE(); i++)
	{
		fe = fes->GetFE(i);
		doftrans = fes->GetElementVDofs(i, vdofs);
		T = fes->GetElementTransformation(i);
		data.GetSubVector(vdofs, el_data);
		if (doftrans) {doftrans->InvTransformPrimal(el_data); }
		A.SetSize(vdofs.Size()); 
		A = 0.0; 
		for (int k = 0; k < dnfi.Size(); k++)
		{
			dnfi[k]->AssembleElementGrad(*fe, *T, el_data, elmat);
			if (doftrans) { doftrans->TransformDual(elmat); }
			A += elmat; 
		}
		x.GetSubVector(vdofs, el_x);
		if (lump) {
		#ifndef NDBEUG 
			// ensure diagonal matrix 
			bool diag = true; 
			for (int i=0; i<A.Height(); i++) {
				double Aii = A(i,i);
				for (int j=0; j<A.Width(); j++) {
					if (i != j and std::fabs(A(i,j)/Aii) > 1e-14) diag = false; 
				}
			}
			if (!diag) MFEM_ABORT("matrix isn't diagonal");
		#endif
			for (int r=0; r<A.Height(); r++) {
				el_x(r) /= A(r,r); 
			}
		} else {
			mfem::LinearSolve(A, el_x.GetData()); 				
		}
		y.SetSubVector(vdofs, el_x); 		
	}
}