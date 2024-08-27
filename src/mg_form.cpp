#include "mg_form.hpp"
#include "lumping.hpp"

void MGMassIntegrator::AssembleElementMatrices(const mfem::FiniteElement &fe, mfem::ElementTransformation &trans, 
	const mfem::Array2D<mfem::DenseMatrix*> &elmats)
{
	const int dof = fe.GetDof();
	shape.SetSize(dof);
	uu.SetSize(dof);

	// block diagonal by group 
	if (vec_coef) {
		for (int i=0; i<G; i++) {
			for (int j=0; j<G; j++) {
				if (i==j) {
					elmats(i,j)->SetSize(dof,dof);
					*elmats(i,j) = 0.0;
				}
				else elmats(i,j)->SetSize(0,0);
			}
		}		
	} 

	// dense coupling 
	else {
		for (int i=0; i<G; i++) {
			for (int j=0; j<G; j++) {
				elmats(i,j)->SetSize(dof,dof);
				*elmats(i,j) = 0.0;
			}
		}
	}

	const mfem::IntegrationRule *ir = IntRule; 
	if (!ir) {
		ir = &mfem::IntRules.Get(fe.GetGeomType(), fe.GetOrder()*2);
	}

	for (int n = 0; n < ir->GetNPoints(); n++)
	{
		const mfem::IntegrationPoint &ip = ir->IntPoint(n);
		trans.SetIntPoint (&ip);

		fe.CalcPhysShape(trans, shape);

		auto w = trans.Weight() * ip.weight;
		MultVVt(shape, uu);
		if (vec_coef) {
			vec_coef->Eval(vec_eval, trans, ip);
			for (int g=0; g<G; g++) {
				elmats(g,g)->Add(w*vec_eval(g), uu);
			}
		}

		else {
			mat_coef->Eval(mat_eval, trans, ip);
			for (int i=0; i<G; i++) {
				for (int j=0; j<G; j++) {
					elmats(i,j)->Add(w * mat_eval(i,j), uu);
				}
			}
		}

	}
}

MultiGroupBilinearForm::MultiGroupBilinearForm(mfem::FiniteElementSpace &fes, int G)
	: fes(fes), G(G)
{
	offsets.SetSize(G+1);
	offsets = fes.GetTrueVSize(); 
	offsets[0] = 0; 
	offsets.PartialSum();
	spmats.SetSize(G,G);
	spmats = nullptr;
	elmats.SetSize(G,G);
	elmats_all.SetSize(G,G);
	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			elmats(i,j) = new mfem::DenseMatrix;
			elmats_all(i,j) = new mfem::DenseMatrix;
		}
	}
	height = width = offsets.Last();
}

MultiGroupBilinearForm::~MultiGroupBilinearForm()
{
	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			delete spmats(i,j);
			delete elmats(i,j);
			delete elmats_all(i,j);
		}
	}
	delete block_op;
}

void MultiGroupBilinearForm::AddDomainIntegrator(MultiGroupBilinearFormIntegrator *mgbfi)
{
	domain_integs.Append(mgbfi);
}

void MultiGroupBilinearForm::Assemble(int skip_zeros)
{
	// setup sparse/dense matrices 
	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			delete spmats(i,j);
			spmats(i,j) = new mfem::SparseMatrix(fes.GetTrueVSize());
		}
	}
	mfem::Array<int> vdofs;
	auto &mesh = *fes.GetMesh();

	if (domain_integs.Size()) {
		for (int e=0; e<mesh.GetNE(); e++) {
			const auto &fe = *fes.GetFE(e);
			auto &trans = *mesh.GetElementTransformation(e);
			fes.GetElementVDofs(e, vdofs);
			for (int i=0; i<G; i++) {
				for (int j=0; j<G; j++) {
					elmats_all(i,j)->SetSize(vdofs.Size()); 
					*elmats_all(i,j) = 0.0;
				}
			}
			for (auto &domain_integ : domain_integs) {
				domain_integ->AssembleElementMatrices(fe, trans, elmats);
				for (int i=0; i<G; i++) {
					for (int j=0; j<G; j++) {
						if (elmats(i,j)->Height()==0) continue;
						*elmats_all(i,j) += *elmats(i,j);						
					}
				}
			}
			for (int i=0; i<G; i++) {
				for (int j=0; j<G; j++) {
					if (elmats_all(i,j)->Height()==0) continue;
					spmats(i,j)->AddSubMatrix(vdofs, vdofs, *elmats_all(i,j), skip_zeros);					
				}
			}
		}		
	}
}

void MultiGroupBilinearForm::Finalize(int skip_zeros)
{
	if (!block_op) {
		block_op = new mfem::BlockOperator(offsets);
	}

	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			if (spmats(i,j)->NumNonZeroElems() > 0) {
				spmats(i,j)->Finalize(skip_zeros);
				block_op->SetBlock(i,j, spmats(i,j));
			}
		}
	}		
}