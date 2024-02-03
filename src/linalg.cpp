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

TripleProductOperator::TripleProductOperator(const mfem::Operator *a, const mfem::Operator *b, const mfem::Operator *c,
	bool owna, bool ownb, bool ownc)
	: A(a), B(b), C(c), ownA(owna), ownB(ownb), ownC(ownc), mfem::Operator(a->Height(), c->Width())
{
	MFEM_VERIFY(A->Width() == B->Height(), 
		"incompatible Operators: A->Width() = " << A->Width() << ", B->Height() = " << B->Height());
	MFEM_VERIFY(B->Width() == C->Height(), 
		"incompatible Operators: B->Width() = " << B->Width() << ", C->Height() = " << C->Height());
	t1.SetSize(C->Height()); 
	t2.SetSize(B->Height()); 

	// only difference between mfem and this implementation 
	t1 = 0.0; 
	t2 = 0.0; 
}

TripleProductOperator::~TripleProductOperator() 
{
	if (ownA) delete A; 
	if (ownB) delete B; 
	if (ownC) delete C; 
}