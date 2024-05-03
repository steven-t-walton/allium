#include "block_diag_op.hpp"

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
	mfem::Array<int> vdofs; 
	mfem::Vector subx;
	auto *fes = op->FESpace(); 
	for (int e=0; e<fes->GetNE(); e++) {
		auto elmat = op->GetElementMatrix(e); // copy since linearsolve modifies data 
		fes->GetElementVDofs(e, vdofs); 
		x.GetSubVector(vdofs, subx); 
		if (assume_diagonal) {
		#ifndef NDBEUG 
			// ensure matrix is actually diagonal 
			// performed in debug mode only 
			bool diag = true; 
			for (int i=0; i<elmat.Height(); i++) {
				double Aii = elmat(i,i);
				for (int j=0; j<elmat.Width(); j++) {
					if (i != j and std::fabs(elmat(i,j)/Aii) > 1e-14) diag = false; 
				}
			}
			if (!diag) MFEM_ABORT("matrix isn't diagonal");
		#endif
			for (int i=0; i<elmat.Height(); i++) {
				subx(i) /= elmat(i,i); 
			}
		} else {
			mfem::LinearSolve(elmat, subx.GetData()); 			
		}
		y.SetSubVector(vdofs, subx); 
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