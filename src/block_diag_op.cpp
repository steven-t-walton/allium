#include "block_diag_op.hpp"
#include "log.hpp"

DenseBlockDiagonalOperator::DenseBlockDiagonalOperator(
	const mfem::Table &_row_table, const mfem::Table &_col_table)
	: row_table(&_row_table), col_table(&_col_table)
{
	data.SetSize(row_table->Size());
	for (int block=0; block<data.Size(); block++) {
		data[block] = new mfem::DenseMatrix(row_table->RowSize(block), col_table->RowSize(block));
		(*data[block]) = 0.0;
	}

	height = row_table->Size_of_connections();
	width = col_table->Size_of_connections();
}

DenseBlockDiagonalOperator::DenseBlockDiagonalOperator(const mfem::FiniteElementSpace &fes)
{
	mfem::Array<int> vdofs;
	fes.GetElementVDofs(0, vdofs); // <-- ensure dof table built 
	row_table = col_table = &fes.GetElementToDofTable();
	data.SetSize(row_table->Size());
	for (int block=0; block<data.Size(); block++) {
		data[block] = new mfem::DenseMatrix(row_table->RowSize(block), col_table->RowSize(block));
		(*data[block]) = 0.0;
	}

	height = width = row_table->Size_of_connections();
}

DenseBlockDiagonalOperator::DenseBlockDiagonalOperator(
	const mfem::FiniteElementSpace &tr_fes, const mfem::FiniteElementSpace &te_fes)
{
	assert(tr_fes.GetNE() == te_fes.GetNE());
	mfem::Array<int> vdofs;
	tr_fes.GetElementVDofs(0, vdofs); // <-- ensure dof tables built 
	te_fes.GetElementVDofs(0, vdofs);
	row_table = &te_fes.GetElementToDofTable();
	col_table = &tr_fes.GetElementToDofTable();
	data.SetSize(row_table->Size());
	for (int block=0; block<data.Size(); block++) {
		data[block] = new mfem::DenseMatrix(row_table->RowSize(block), col_table->RowSize(block));
		(*data[block]) = 0.0;
	}

	height = row_table->Size_of_connections();
	width = col_table->Size_of_connections();
}

DenseBlockDiagonalOperator::~DenseBlockDiagonalOperator()
{
	for (int block=0; block<data.Size(); block++) {
		delete data[block];
	}
}

const mfem::DenseMatrix &DenseBlockDiagonalOperator::GetBlock(int block) const 
{
	assert(block >= 0 and block < data.Size());
	return *data[block];
}

mfem::DenseMatrix &DenseBlockDiagonalOperator::GetBlock(int block) 
{
	assert(block >= 0 and block < data.Size());
	return *data[block];
}

void DenseBlockDiagonalOperator::SetBlock(int block, const mfem::DenseMatrix &elmat)
{
	assert(block >= 0 and block < data.Size());
	assert(data[block]);
	auto &my_elmat = *data[block];
	assert(my_elmat.Height() == elmat.Height() and my_elmat.Width() == elmat.Width());
	my_elmat = elmat;
}

void DenseBlockDiagonalOperator::Invert() 
{
	assert(height == width);
	for (int b=0; b<data.Size(); b++) {
		data[b]->Invert();
	}
}

void add(double a, const DenseBlockDiagonalOperator &A, 
	double b, const DenseBlockDiagonalOperator &B, DenseBlockDiagonalOperator &C)
{
	assert(A.NumBlocks() == B.NumBlocks() and A.NumBlocks() == C.NumBlocks());
	for (int block=0; block<A.NumBlocks(); block++) {
		const auto &Amat = A.GetBlock(block);
		const auto &Bmat = B.GetBlock(block);
		auto &Cmat = C.GetBlock(block);
		mfem::Add(a, Amat, b, Bmat, Cmat);
	}
}

void Mult(const DenseBlockDiagonalOperator &A, const DenseBlockDiagonalOperator &B, 
	DenseBlockDiagonalOperator &C)
{
	assert(A.NumBlocks() == B.NumBlocks() and A.NumBlocks() == C.NumBlocks());
	for (int block=0; block<A.NumBlocks(); block++) {
		const auto &Amat = A.GetBlock(block);
		const auto &Bmat = B.GetBlock(block);
		auto &Cmat = C.GetBlock(block);
		mfem::Mult(Amat, Bmat, Cmat);
	}
}

void Mult(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &B, 
	DenseBlockDiagonalOperator &C)
{
	assert(A.NumBlocks() == C.NumBlocks());
	assert(A.Width() == B.Height());
	const auto &row_table = A.GetRowDofs();
	const auto &col_table = A.GetColDofs();
	mfem::Array<int> row_dofs, col_dofs;
	mfem::DenseMatrix Bmat;
	for (int block=0; block<A.NumBlocks(); block++) {
		col_table.GetRow(block, col_dofs);
		row_table.GetRow(block, row_dofs);
		Bmat.SetSize(row_dofs.Size(), col_dofs.Size());
		B.GetSubMatrix(row_dofs, col_dofs, Bmat);
		const auto &Amat = A.GetBlock(block);
		auto &Cmat = C.GetBlock(block);
		mfem::Mult(Amat, Bmat, Cmat);
	}
}

mfem::SparseMatrix *Mult(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &B)
{
	auto *Asp = A.AsSparseMatrix(); 
	auto *mult = mfem::Mult(*Asp, B);
	delete Asp;
	return mult;
}

mfem::SparseMatrix *TripleProduct(const DenseBlockDiagonalOperator &A, const DenseBlockDiagonalOperator &B, 
	const mfem::SparseMatrix &C)
{
	DenseBlockDiagonalOperator D(A.GetRowDofs(), A.GetColDofs());
	Mult(A, B, D);
	return Mult(D, C);
}

mfem::SparseMatrix *RAP(const DenseBlockDiagonalOperator &A, const mfem::SparseMatrix &P)
{
	auto *Asp = A.AsSparseMatrix();
	auto *rap = mfem::RAP(P, *Asp, P);
	delete Asp;
	return rap;
}

void DenseBlockDiagonalOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(x.Size() == width);
	assert(y.Size() == height);
	mfem::Array<int> x_dofs, y_dofs;
	mfem::Vector sub_x, sub_y;
	for (int b=0; b<data.Size(); b++) {
		col_table->GetRow(b, x_dofs);
		sub_x.SetSize(x_dofs.Size());
		x.GetSubVector(x_dofs, sub_x);

		row_table->GetRow(b, y_dofs);
		sub_y.SetSize(y_dofs.Size());
		data[b]->Mult(sub_x, sub_y);

		y.SetSubVector(y_dofs, sub_y);
	}
}

void DenseBlockDiagonalOperator::AddMult(const mfem::Vector &x, mfem::Vector &y, const double a) const
{
	assert(x.Size() == width);
	assert(y.Size() == height);
	mfem::Array<int> x_dofs, y_dofs;
	mfem::Vector sub_x, sub_y, product;
	for (int b=0; b<data.Size(); b++) {
		col_table->GetRow(b, x_dofs);
		row_table->GetRow(b, y_dofs);

		x.GetSubVector(x_dofs, sub_x);
		product.SetSize(y_dofs.Size());
		data[b]->Mult(sub_x, product);

		y.GetSubVector(y_dofs, sub_y);
		add(product, a, sub_y, sub_y);

		y.SetSubVector(y_dofs, sub_y);
	}
}

mfem::SparseMatrix *DenseBlockDiagonalOperator::AsSparseMatrix() const
{
	auto *A = new mfem::SparseMatrix(height, width);
	mfem::Array<int> row_dofs, col_dofs;
	for (int b=0; b<data.Size(); b++) {
		row_table->GetRow(b, row_dofs);
		col_table->GetRow(b, col_dofs);

		const auto &elmat = GetBlock(b);
		A->AddSubMatrix(row_dofs, col_dofs, elmat);
	}
	A->Finalize();
	return A;
}

void DenseBlockDiagonalSolver::SetOperator(const mfem::Operator &_op)
{
	op = dynamic_cast<const DenseBlockDiagonalOperator*>(&_op); 
	if (!op) MFEM_ABORT("operator must be a DenseBlockDiagonalOperator"); 
	height = width = op->Height(); 
}

void DenseBlockDiagonalSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(op); 
	mfem::Array<int> vdofs; 
	mfem::Vector subx, suby;
	const auto &dof_table = op->GetRowDofs();
	for (int e=0; e<op->NumBlocks(); e++) {
		const auto &elmat = op->GetBlock(e); 
		dof_table.GetRow(e, vdofs);
		x.GetSubVector(vdofs, subx); 
		suby.SetSize(subx.Size()); 
		local_solver.SetOperator(elmat); 
		local_solver.Mult(subx, suby); 
		y.SetSubVector(vdofs, suby); 
	}
}

void DiagonalDenseMatrixInverse::SetOperator(const mfem::Operator &op)
{
	mat = dynamic_cast<const mfem::DenseMatrix*>(&op); 
	if (!mat) MFEM_ABORT("must be a DenseMatrix"); 
	height = width = op.Height(); 
#ifndef NDBEUG
	bool diag = true; 
	const mfem::DenseMatrix &A = *mat; 
	for (int i=0; i<Height(); i++) {
		double Aii = A(i,i); 
		for (int j=0; j<Width(); j++) {
			if (i != j and std::fabs(A(i,j)/Aii) > 1e-14) diag = false;
		}
	}
	if (!diag) MFEM_ABORT("matrix must be diagonal");
#endif
}

void DiagonalDenseMatrixInverse::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(mat); 
	const mfem::DenseMatrix &A = *mat; 
	for (int i=0; i<Height(); i++) {
		y(i) = x(i) / A(i,i);
	}
}

DenseBlockDiagonalNonlinearForm::~DenseBlockDiagonalNonlinearForm()
{
	delete grad; 
}

void DenseBlockDiagonalNonlinearForm::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	using namespace mfem; 
	Array<int> vdofs;
	Vector el_x, el_y;
	const FiniteElement *fe;
	ElementTransformation *T;
	DofTransformation *doftrans;
	Mesh *mesh = fes->GetMesh();

	y = 0.0;

	if (dnfi.Size()) {
		// Which attributes need to be processed?
		Array<int> attr_marker(mesh->attributes.Size() ? mesh->attributes.Max() : 0);
		attr_marker = 0;
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] == NULL) {
				attr_marker = 1;
				break;
			}
			Array<int> &marker = *dnfi_marker[k];
			MFEM_ASSERT(marker.Size() == attr_marker.Size(),
				"invalid marker for domain integrator #"
				<< k << ", counting from zero");
			for (int i = 0; i < attr_marker.Size(); i++) {
				attr_marker[i] |= marker[i];
			}
		}

		for (int i = 0; i < fes->GetNE(); i++) {
			const int attr = mesh->GetAttribute(i);
			if (attr_marker[attr-1] == 0) { continue; }

			fe = fes->GetFE(i);
			doftrans = fes->GetElementVDofs(i, vdofs);
			T = fes->GetElementTransformation(i);
			x.GetSubVector(vdofs, el_x);
			if (doftrans) {doftrans->InvTransformPrimal(el_x); }
			for (int k = 0; k < dnfi.Size(); k++) {
				if (dnfi_marker[k] &&(*dnfi_marker[k])[attr-1] == 0) { continue; }

				dnfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
				if (doftrans) {doftrans->TransformDual(el_y); }
				y.AddElementVector(vdofs, el_y);
			}
		}
	}

	if (fnfi.Size()) { MFEM_ABORT("face nonlinear form integrators not supported"); }
	if (bfnfi.Size()) { MFEM_ABORT("bdr face nonlinear form integrators not supported"); }
}

DenseBlockDiagonalOperator &DenseBlockDiagonalNonlinearForm::GetGradient(const mfem::Vector &x) const 
{
	using namespace mfem; 
	if (!grad) grad = new DenseBlockDiagonalOperator(*fes); 

	Array<int> vdofs;
	Vector el_x;
	const FiniteElement *fe;
	DenseMatrix elmat; 
	ElementTransformation *T;
	DofTransformation *doftrans;
	Mesh *mesh = fes->GetMesh();

	if (dnfi.Size()) {
		// Which attributes need to be processed?
		Array<int> attr_marker(mesh->attributes.Size() ? mesh->attributes.Max() : 0);
		attr_marker = 0;
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] == NULL) {
				attr_marker = 1;
				break;
			}
			Array<int> &marker = *dnfi_marker[k];
			MFEM_ASSERT(marker.Size() == attr_marker.Size(),
				"invalid marker for domain integrator #"
				<< k << ", counting from zero");
			for (int i = 0; i < attr_marker.Size(); i++) {
				attr_marker[i] |= marker[i];
			}
		}

		for (int i = 0; i < fes->GetNE(); i++) {
			const int attr = mesh->GetAttribute(i);
			if (attr_marker[attr-1] == 0) { continue; }

			auto &grad_mat = grad->GetBlock(i); 
			grad_mat = 0.0; 
			fe = fes->GetFE(i);
			doftrans = fes->GetElementVDofs(i, vdofs);
			T = fes->GetElementTransformation(i);
			x.GetSubVector(vdofs, el_x);
			if (doftrans) {doftrans->InvTransformPrimal(el_x); }
			for (int k = 0; k < dnfi.Size(); k++) {
				if (dnfi_marker[k] &&(*dnfi_marker[k])[attr-1] == 0) { continue; }

				dnfi[k]->AssembleElementGrad(*fe, *T, el_x, elmat);
				if (doftrans) { doftrans->TransformDual(elmat); }
				// Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
				grad_mat += elmat; // <-- only change for dnfi 
			}
		}
	}

	if (fnfi.Size()) { MFEM_ABORT("face nonlinear form integrators not supported"); }
	if (bfnfi.Size()) { MFEM_ABORT("bdr face nonlinear form integrators not supported"); }

	return *grad; 
}

void DenseBlockDiagonalNonlinearForm::AssembleLocalResidual(int element, 
	const mfem::Vector &x, mfem::Vector &y) const
{
	using namespace mfem; 
	Array<int> vdofs;
	Vector el_x, el_y;
	const FiniteElement *fe;
	ElementTransformation *T;
	DofTransformation *doftrans;
	Mesh *mesh = fes->GetMesh();

	y = 0.0;

	if (dnfi.Size()) {
		// Which attributes need to be processed?
		Array<int> attr_marker(mesh->attributes.Size() ? mesh->attributes.Max() : 0);
		attr_marker = 0;
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] == NULL) {
				attr_marker = 1;
				break;
			}
			Array<int> &marker = *dnfi_marker[k];
			MFEM_ASSERT(marker.Size() == attr_marker.Size(),
				"invalid marker for domain integrator #"
				<< k << ", counting from zero");
			for (int i = 0; i < attr_marker.Size(); i++) {
				attr_marker[i] |= marker[i];
			}
		}

		const int attr = mesh->GetAttribute(element);
		if (attr_marker[attr-1] == 0) { return; }

		fe = fes->GetFE(element);
		doftrans = fes->GetElementVDofs(element, vdofs);
		T = fes->GetElementTransformation(element);
		el_x = x; 
		if (doftrans) {doftrans->InvTransformPrimal(el_x); }
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] &&(*dnfi_marker[k])[attr-1] == 0) { continue; }

			dnfi[k]->AssembleElementVector(*fe, *T, el_x, el_y);
			if (doftrans) {doftrans->TransformDual(el_y); }
			y += el_y; 
		}
	}

	if (fnfi.Size()) { MFEM_ABORT("face nonlinear form integrators not supported"); }
	if (bfnfi.Size()) { MFEM_ABORT("bdr face nonlinear form integrators not supported"); }
}

void DenseBlockDiagonalNonlinearForm::AssembleLocalGradient(
	int element, const mfem::Vector &x, mfem::DenseMatrix &A) const 
{
	using namespace mfem; 

	Array<int> vdofs;
	Vector el_x;
	const FiniteElement *fe;
	DenseMatrix elmat; 
	ElementTransformation *T;
	DofTransformation *doftrans;
	Mesh *mesh = fes->GetMesh();

	if (dnfi.Size()) {
		// Which attributes need to be processed?
		Array<int> attr_marker(mesh->attributes.Size() ? mesh->attributes.Max() : 0);
		attr_marker = 0;
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] == NULL) {
				attr_marker = 1;
				break;
			}
			Array<int> &marker = *dnfi_marker[k];
			MFEM_ASSERT(marker.Size() == attr_marker.Size(),
				"invalid marker for domain integrator #"
				<< k << ", counting from zero");
			for (int i = 0; i < attr_marker.Size(); i++) {
				attr_marker[i] |= marker[i];
			}
		}

		const int attr = mesh->GetAttribute(element);
		if (attr_marker[attr-1] == 0) { return; }

		fe = fes->GetFE(element);
		doftrans = fes->GetElementVDofs(element, vdofs);
		A.SetSize(vdofs.Size()); 
		A = 0.0; 
		T = fes->GetElementTransformation(element);
		el_x = x; 
		if (doftrans) {doftrans->InvTransformPrimal(el_x); }
		for (int k = 0; k < dnfi.Size(); k++) {
			if (dnfi_marker[k] &&(*dnfi_marker[k])[attr-1] == 0) { continue; }

			dnfi[k]->AssembleElementGrad(*fe, *T, el_x, elmat);
			if (doftrans) { doftrans->TransformDual(elmat); }
			// Grad->AddSubMatrix(vdofs, vdofs, elmat, skip_zeros);
			A += elmat; // <-- only change for dnfi 
		}
	}

	if (fnfi.Size()) { MFEM_ABORT("face nonlinear form integrators not supported"); }
	if (bfnfi.Size()) { MFEM_ABORT("bdr face nonlinear form integrators not supported"); }	
}

void EnergyBalanceNewtonSolver::SetOperator(const mfem::Operator &op) 
{
	NewtonSolver::SetOperator(op); 
	xnew.SetSize(height); 
	successive_residual.SetSize(height);
}

double EnergyBalanceNewtonSolver::Norm(const mfem::Vector &x) const
{
	return x.Norml1();
}

void EnergyBalanceNewtonSolver::Floor(mfem::Vector &x) const 
{
	for (auto &val : x) {
		if (val <= minimum_solution) {
			EventLog.Register("floored temperature");
			val = minimum_solution;
		}
	}
}

void EnergyBalanceNewtonSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
	MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

	if (!iterative_mode) {
		x = 0.0;
	}

	int it;
	double successive_iter_norm, norm, norm0; 

	// compute initial residual norm 
	oper->Mult(x, r); 
	r -= b; 
	norm0 = norm = initial_norm = Norm(r); 
	if (norm0/Norm(b) < rel_tol) {
		converged = true; 
		final_iter = 1;
		final_norm = rel_tol;
		initial_norm = 1.0;
		Floor(x);
		return;
	}

	// stopping criterion for nonlinear residual norm 
	const double norm_goal = std::max(rel_tol*norm, abs_tol); 
	// exit if solution changes by a small amount 
	// max_stagnation_count times 
	int stagnation_count = 0;
	exit_code = Undefined;
	for (it=0; true; it++) {
		if (norm <= norm_goal) {
			exit_code = Success; 
			break;
		}
		if (it >= max_iter) {
			exit_code = MaxIterationsReached; 
			break; 
		}
		grad = &oper->GetGradient(x); 
		prec->SetOperator(*grad); 
		prec->Mult(r, c); 
		add(x, -1.0, c, xnew);
		for (int i=0; i<height; i++) {
			if (xnew(i) <= minimum_solution) 
				xnew(i) = (1.0 - under_relax_param)*x(i);
		}
		add(xnew, -1.0, x, successive_residual);
		successive_iter_norm = successive_residual.Normlinf(); 
		x = xnew;

		oper->Mult(x, r); 
		r -= b;
		norm = Norm(r);
		if (successive_iter_norm <= minimum_solution * 1e-2) {
			stagnation_count++; 
			if (stagnation_count > max_stagnation_count) {
				exit_code = Stagnated; 
				break; 				
			}
		} else {
			stagnation_count = 0; 
		}
	}

	Floor(x);
	converged = exit_code > 0; 
	final_iter = it; 
	final_norm = norm;
}

void DenseBlockDiagonalNonlinearSolver::SetOperator(const mfem::Operator &op) 
{
	form = dynamic_cast<const DenseBlockDiagonalNonlinearForm*>(&op); 
	if (!form) MFEM_ABORT("must supply DenseBlockDiagonalNonlinearForm"); 
	height = width = form->Height(); 
}

void DenseBlockDiagonalNonlinearSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	assert(form); 
	const bool have_b = b.Size() > 0; 
	mfem::Array<int> vdofs; 
	mfem::Vector sub_b, sub_x, r; 
	const auto *fes = form->FESpace(); 
	converged = true; 
	final_iter = 0;
	final_norm = -1.0; 
	final_rel_norm = -1.0; 

	if (!local_solver.iterative_mode) x = 0.0; 

	for (int e=0; e<fes->GetNE(); e++) {
		fes->GetElementVDofs(e, vdofs); 
		if (have_b)
			b.GetSubVector(vdofs, sub_b); 
		x.GetSubVector(vdofs, sub_x); // get for initial guess
		DenseBlockDiagonalNonlinearForm::LocalOperator local_op(*form, e); 
		local_solver.SetOperator(local_op); 
		local_solver.Mult(sub_b, sub_x); 
		x.SetSubVector(vdofs, sub_x); 

		// conform to iterative solver interface 
		if (!local_solver.GetConverged()) {
			converged = false; 
			EventLog["local nonlinear solve failed"]++;				
		}
		final_iter = std::max(final_iter, local_solver.GetNumIterations()); 
		final_norm = std::max(local_solver.GetFinalNorm(), final_norm); 
		final_rel_norm = std::max(local_solver.GetFinalRelNorm(), final_rel_norm); 
	} 

	const auto *pfes = dynamic_cast<const mfem::ParFiniteElementSpace*>(fes); 
	if (pfes and pfes->GetNRanks()>1) {
		int iter_local = final_iter; 
		MPI_Reduce(&iter_local, &final_iter, 1, MPI_INT, MPI_MAX, 0, pfes->GetComm()); 
		bool local_converged = converged; 
		MPI_Reduce(&local_converged, &converged, 1, MPI_C_BOOL, MPI_LAND, 0, pfes->GetComm());
		double local_norms[2] = {final_rel_norm, final_norm}; 
		double norms[2] = {0.0, 0.0}; 
		MPI_Reduce(&local_norms, &norms, 2, MPI_DOUBLE, MPI_MAX, 0, pfes->GetComm()); 
		final_rel_norm = norms[0];
		final_norm = norms[1]; 
	}
}