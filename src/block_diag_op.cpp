#include "block_diag_op.hpp"
#include "log.hpp"

BlockDiagonalByElementOperator::BlockDiagonalByElementOperator(const mfem::FiniteElementSpace &f)
	: fes(f), mfem::Operator(f.GetVSize())
{
	data.SetSize(fes.GetNE()); 
	for (int e=0; e<data.Size(); e++) {
		data[e] = nullptr; 
	}
}

BlockDiagonalByElementOperator::~BlockDiagonalByElementOperator()
{
	for (auto *ptr : data) {
		delete ptr; 
	}
}

void BlockDiagonalByElementOperator::SetElementMatrix(int elem, const mfem::DenseMatrix &elmat) 
{
	if (data[elem]) delete data[elem]; 
	data[elem] = new mfem::DenseMatrix(elmat); 
}

const mfem::DenseMatrix &BlockDiagonalByElementOperator::GetElementMatrix(int elem) const 
{
	assert(data[elem]); 
	return *data[elem]; 
}

mfem::DenseMatrix &BlockDiagonalByElementOperator::GetElementMatrix(int elem)
{
	if (!data[elem]) {
		mfem::Array<int> vdofs; 
		fes.GetElementVDofs(elem, vdofs); 
		data[elem] = new mfem::DenseMatrix(vdofs.Size()); 
		(*data[elem]) = 0.0; 
	}
	return *data[elem]; 
}

void BlockDiagonalByElementOperator::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	mfem::Array<int> vdofs; 
	mfem::Vector subx, suby; 
	for (int e=0; e<fes.GetNE(); e++) {
		const mfem::DenseMatrix *elmat = data[e]; 
		if (!elmat) continue; 
		fes.GetElementVDofs(e, vdofs); 
		x.GetSubVector(vdofs, subx); 
		suby.SetSize(subx.Size()); 
		elmat->Mult(subx, suby);
		y.SetSubVector(vdofs, suby);  
	}
}

void BlockDiagonalByElementSolver::SetOperator(const mfem::Operator &_op)
{
	op = dynamic_cast<const BlockDiagonalByElementOperator*>(&_op); 
	if (!op) MFEM_ABORT("operator must be a BlockDiagonalByElementOperator"); 
	height = width = op->Height(); 
}

void BlockDiagonalByElementSolver::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
	assert(op); 
	mfem::Array<int> vdofs; 
	mfem::Vector subx, suby;
	auto *fes = op->FESpace(); 
	for (int e=0; e<fes->GetNE(); e++) {
		const auto &elmat = op->GetElementMatrix(e); 
		fes->GetElementVDofs(e, vdofs); 
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

BlockDiagonalByElementNonlinearForm::~BlockDiagonalByElementNonlinearForm()
{
	delete grad; 
}

void BlockDiagonalByElementNonlinearForm::Mult(const mfem::Vector &x, mfem::Vector &y) const
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

mfem::Operator &BlockDiagonalByElementNonlinearForm::GetGradient(const mfem::Vector &x) const 
{
	using namespace mfem; 
	if (!grad) grad = new BlockDiagonalByElementOperator(*fes); 

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

			auto &grad_mat = grad->GetElementMatrix(i); 
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

void BlockDiagonalByElementNonlinearForm::AssembleLocalResidual(int element, 
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

void BlockDiagonalByElementNonlinearForm::AssembleLocalGradient(
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
}

void EnergyBalanceNewtonSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
	MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
	MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

	const bool have_b = (b.Size() == Height());

	if (!iterative_mode) {
		x = 0.0;
	}

	int it;
	double successive_iter, norm, norm0; 

	// compute initial residual norm 
	oper->Mult(x, r); 
	if (have_b) r -= b; 
	norm0 = norm = initial_norm = Norm(r); 

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
				xnew(i) = x(i) - under_relax_param*c(i); 
		}
		x -= xnew; 
		successive_iter = Norm(x); 
		x = xnew;
		if (successive_iter <= minimum_solution * 1e-2) {
			stagnation_count++; 
			if (stagnation_count > max_stagnation_count) {
				initial_norm = Norm(b); 
				exit_code = Stagnated; 
				break; 				
			}
		} else {
			stagnation_count = 0; 
		}
		oper->Mult(x, r); 
		if (have_b) r -= b;
		norm = Norm(r); 
	}

	converged = exit_code > 0; 
	final_iter = it; 
	final_norm = norm;
}

void BlockDiagonalByElementNonlinearSolver::SetOperator(const mfem::Operator &op) 
{
	form = dynamic_cast<const BlockDiagonalByElementNonlinearForm*>(&op); 
	if (!form) MFEM_ABORT("must supply BlockDiagonalByElementNonlinearForm"); 
	height = width = form->Height(); 
}

void BlockDiagonalByElementNonlinearSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
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
		BlockDiagonalByElementNonlinearForm::LocalOperator local_op(*form, e); 
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