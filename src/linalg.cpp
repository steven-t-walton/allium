#include "linalg.hpp"

mfem::HypreParMatrix *ElementByElementBlockInverse(const mfem::ParFiniteElementSpace &fes, const mfem::HypreParMatrix &A) {
	mfem::SparseMatrix diag, offd; 
	A.GetDiag(diag); 
	HYPRE_BigInt *cmap; 
	A.GetOffd(offd, cmap); 
	if (offd.Width()>0) { MFEM_ABORT("block inverse only available for block diagonal matrices"); }
	mfem::Array<int> vdofs; 
	auto *inv = new mfem::SparseMatrix(diag.Height()); 
	mfem::DenseMatrix mat; 
	for (auto e=0; e<fes.GetNE(); e++) {
		fes.GetElementVDofs(e, vdofs); 
		mat.SetSize(vdofs.Size()); 
		diag.GetSubMatrix(vdofs, vdofs, mat); 
		mat.Invert(); 
		inv->AddSubMatrix(vdofs, vdofs, mat); 
	}
	inv->Finalize(); 
	auto *ptr = new mfem::HypreParMatrix(fes.GetComm(), fes.GlobalVSize(), fes.GetDofOffsets(), inv); 
	ptr->SetOwnerFlags(true, true, true); 
	return ptr; 
}