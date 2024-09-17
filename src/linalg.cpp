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

mfem::HypreParMatrix *BlockOperatorToMonolithic(const mfem::BlockOperator &bop) 
{
	mfem::Array2D<mfem::HypreParMatrix*> blocks(bop.NumRowBlocks(), bop.NumColBlocks()); 
	blocks = nullptr;
	for (auto row=0; row<blocks.NumRows(); row++) {
		for (auto col=0; col<blocks.NumCols(); col++) {
			if (bop.IsZeroBlock(row,col)) continue;
			const mfem::Operator *op = &bop.GetBlock(row,col); 
			const auto *hypre_op = dynamic_cast<const mfem::HypreParMatrix*>(op); 
			if (!hypre_op) MFEM_ABORT("blocks must be HypreParMatrix"); 
			blocks(row,col) = const_cast<mfem::HypreParMatrix*>(hypre_op); 
		}
	}
	return mfem::HypreParMatrixFromBlocks(blocks); 
}

SubBlockReductionOperator::SubBlockReductionOperator(mfem::BlockVector &data, const mfem::Operator &op, int comp)
	: data(data), op(op), comp(comp)
{
	const int num_blocks = data.NumBlocks();
	row_offsets.SetSize(num_blocks+1);
	col_offsets.SetSize(num_blocks+1);
	row_offsets[0] = col_offsets[0] = 0;
	for (int b=0; b<num_blocks; b++) {
		row_offsets[b+1] = data.BlockSize(b);
		col_offsets[b+1] = data.BlockSize(b);
	} 
	row_offsets[comp+1] = op.Height();
	if (op.Width() != col_offsets[comp+1]) {
		MFEM_ABORT("blocks not compatible");
	}
	row_offsets.PartialSum();
	col_offsets.PartialSum();
	height = row_offsets.Last();
	width = col_offsets.Last();
}

void SubBlockReductionOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	const mfem::BlockVector bx(const_cast<mfem::Vector&>(x), col_offsets);
	mfem::BlockVector by(y, row_offsets);
	data = bx;
	for (int b=0; b<row_offsets.Size()-1; b++) {
		if (b==comp) {
			op.Mult(bx.GetBlock(b), by.GetBlock(b));
		} else {
			by.GetBlock(b) = bx.GetBlock(b);
		}
	}
}

void SubBlockReductionOperator::MultTranspose(const mfem::Vector &x, mfem::Vector &y) const 
{
	if (y.GetData() != data.GetData()) {
		mfem::BlockVector by(y, col_offsets);
		by = data;
	}
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

void SLISolver::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	if (!iterative_mode) {
		x = 0.0; 
	}

	double norm, r0; 
	int i; 
	converged = false; 
	bool done = false; 
	for (i=1; true;) {
		// compute residual 
		oper->Mult(x, r); 
		subtract(b, r, r); // r = b - Ax 
		// apply preconditioner, compute norm of residual, update solution
		if (prec) {
			prec->Mult(r, z); // z = Br 
			norm = sqrt(Dot(z,z)); 
			add(x, 1.0, z, x); 
		} else {
			norm = sqrt(Dot(r,r)); 
			add(x, 1.0, r, x); 
		}

		if (i==1) {
			initial_norm = norm; 
			r0 = std::max(norm*rel_tol, abs_tol); 
		}

		if (norm < r0) {
			converged = true; 
			final_iter = i; 
		}
		if (i >= max_iter or converged) {
			done = true; 
		}


		if (prec) {
			Monitor(i, norm, x, z, done); 			
		} else {
			Monitor(i, norm, x, r, done); 
		}

		if (done) { break; }

		i++; 
	}
	final_iter = i; 
	final_norm = norm; 
}

void BlockLDUInverseOperator::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	mfem::BlockVector block_b(const_cast<mfem::Vector&>(b), offsets); 
	mfem::BlockVector block_x(x, offsets); 
	tmp.Update(offsets); 

	// backward solve U 
	Dinv.Mult(block_b.GetBlock(1), tmp.GetBlock(1)); // D^{-1} b_2 -> tmp_2 
	B.Mult(tmp.GetBlock(1), tmp.GetBlock(0)); // B D^{-1} b_2 -> tmp_1 
	add(block_b.GetBlock(0), -1.0, tmp.GetBlock(0), tmp.GetBlock(0)); // b_1 - B D^{-1} b_2 -> tmp_1 

	// diagonal solve 
	Sinv.Mult(tmp.GetBlock(0), block_x.GetBlock(0)); 

	// forward solve L 
	C.Mult(block_x.GetBlock(0), block_x.GetBlock(1)); // C x_1 -> x_2 
	add(block_b.GetBlock(1), -1.0, block_x.GetBlock(1), tmp.GetBlock(1)); // b_2 - C x_1 -> tmp_2 
	Dinv.Mult(tmp.GetBlock(1), block_x.GetBlock(1)); // D^{-1} (b_2 - C x_1) 
}

void SuperLUSolver::SetOperator(const mfem::Operator &op)
{
	height = op.Height(); 
	width = op.Width();
	const auto *hypre = dynamic_cast<const mfem::HypreParMatrix*>(&op);
	const auto *block = dynamic_cast<const mfem::BlockOperator*>(&op);
	if (hypre)
		slu_op.reset(new mfem::SuperLURowLocMatrix(op));
	else if (block) {
		auto *monolithic = BlockOperatorToMonolithic(*block);
		slu_op.reset(new mfem::SuperLURowLocMatrix(*monolithic));
		delete monolithic;
	}
	else MFEM_ABORT("operator type not supported");
	slu.SetOperator(*slu_op);
}

BlockDiagonalPreconditioner::BlockDiagonalPreconditioner(int nBlocks)
	: nBlocks(nBlocks), solvers(nBlocks)
{
	solvers = nullptr;
}

BlockDiagonalPreconditioner::~BlockDiagonalPreconditioner()
{
	if (owns_blocks > 0) {
		for (int b=0; b<nBlocks; b++) {
			delete solvers[b];
		}
	}
}

void BlockDiagonalPreconditioner::SetOperator(const mfem::Operator &op) 
{
	const auto *block_op = dynamic_cast<const mfem::BlockOperator*>(&op);
	if (!block_op) MFEM_ABORT("operator must be a block operator");
	for (int b=0; b<nBlocks; b++) {
		solvers[b]->SetOperator(block_op->GetBlock(b,b));
	}
	offsets = block_op->RowOffsets();
	height = width = op.Height();
}

void BlockDiagonalPreconditioner::Mult(const mfem::Vector &b, mfem::Vector &x) const 
{
	bBlock.Update(const_cast<mfem::Vector&>(b), offsets);
	xBlock.Update(x, offsets);

	for (int b=0; b<nBlocks; b++) {
		if (solvers[b])
			solvers[b]->Mult(bBlock.GetBlock(b), xBlock.GetBlock(b));
		// default to identity if null
		else
			xBlock.GetBlock(b) = bBlock.GetBlock(b);
	}
}

void BlockDiagonalPreconditioner::SetDiagonalBlock(int iBlock, mfem::Solver &solver)
{
	assert(iBlock >= 0 and iBlock < nBlocks);
	solvers[iBlock] = &solver;
}