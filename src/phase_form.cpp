#include "phase_form.hpp"

void MomentVectorNonlinearForm::Mult(const mfem::Vector &x, mfem::Vector &y) const 
{
	using namespace mfem; 
	Array<int> vdofs;
	Vector el_x, el_y;
	const FiniteElement *fe;
	ElementTransformation *T;
	DofTransformation *doftrans;
	Mesh *mesh = fes->GetMesh();

	const auto G = phi_ext.extent(MomentIndex::ENERGY);
	const auto M = phi_ext.extent(MomentIndex::MOMENT);

	y = 0.0;
	auto y_view = MomentVectorView(y.GetData(), phi_ext);

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
				const auto dof = fe->GetDof();
				for (int g=0; g<G; g++) {
					for (int n=0; n<dof; n++) {
						y_view(g, 0, vdofs[n]) += el_y(n + g*dof);
					}
				}
			}
		}
	}

	if (fnfi.Size()) { MFEM_ABORT("face nonlinear form integrators not supported"); }
	if (bfnfi.Size()) { MFEM_ABORT("bdr face nonlinear form integrators not supported"); }
}