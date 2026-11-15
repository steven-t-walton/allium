#include "dg_local_matrices.hpp"
#include "log.hpp"
#include "lumping.hpp"
#include "mg_form.hpp"
#include "multigroup.hpp"

DGLocalMatrices::DGLocalMatrices(mfem::ParFiniteElementSpace &fes, MultiGroupCoefficient &total, int lump)
	: fes(fes), total(total), lump(lump)
{
	// face neighbor data required for shared faces
	fes.ExchangeFaceNbrData();
	auto &mesh = *fes.GetParMesh();
	G = total.GetVDim();
	dim = mesh.Dimension();
	mass.resize(G*fes.GetNE());
	grad.resize(dim*fes.GetNE());
	face.resize(4*mesh.GetNumFaces());
	Assemble();
}

void DGLocalMatrices::Assemble()
{
	mfem::StopWatch timer;
	timer.Start();
	auto &mesh = *fes.GetParMesh();

	mfem::Array2D<mfem::DenseMatrix*> local_mass_mats(G,G);
	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			local_mass_mats(i,j) = new mfem::DenseMatrix;
		}
	}

	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e);
		auto &trans = *mesh.GetElementTransformation(e);
		LumpedIntegrationRule lumped_ir(trans.GetGeometryType());

		MGMassIntegrator mi(total);
		if (IsMassLumped(lump)) mi.SetIntegrationRule(lumped_ir);
		mi.AssembleElementMatrices(fe, trans, local_mass_mats);
		for (int g=0; g<G; g++) {
			mass[g*fes.GetNE() + e] = *local_mass_mats(g,g);
		}

		mfem::Vector Omega(dim);
		for (auto d=0; d<dim; d++) {
			Omega = 0.0;
			Omega(d) = 1.0;
			mfem::VectorConstantCoefficient Q(Omega);
			mfem::ConservativeConvectionIntegrator conv_int(Q, 1.0);
			if (IsGradientLumped(lump)) conv_int.SetIntegrationRule(lumped_ir);
			conv_int.AssembleElementMatrix(fe, trans, grad[d*fes.GetNE() + e]);
		}
	}

	FaceMassMatricesIntegrator fmi;
	mfem::DenseMatrix elmat;
	for (auto f=0; f<mesh.GetNumFaces(); f++) {
		const auto info = mesh.GetFaceInformation(f);
		mfem::FaceElementTransformations *face_trans;
		if (info.IsShared()) {
			face_trans = mesh.GetSharedFaceTransformationsByLocalIndex(f, true);
		} else {
			face_trans = mesh.GetFaceElementTransformations(f);
		}
		const auto &el1 = *fes.GetFE(face_trans->Elem1No);
		const auto *el2 = &el1;
		if (face_trans->Elem2No >= 0) {
			el2 = fes.GetFE(face_trans->Elem2No);
		}

		// setup lumped integration rule
		LumpedIntegrationRule lumped_ir(face_trans->GetGeometryType());
		if (IsFaceLumped(lump)) fmi.SetIntegrationRule(lumped_ir);

		fmi.AssembleFaceMatrix(el1, *el2, *face_trans, elmat);
		const auto dof1 = el1.GetDof();
		elmat.GetSubMatrix(0,dof1,0,dof1, face[4*f]);
		if (face_trans->Elem2No >= 0) {
			const auto dof2 = el2->GetDof();
			elmat.GetSubMatrix(0,dof1,dof1,dof1+dof2, face[4*f + 1]);
			elmat.GetSubMatrix(dof1,dof1+dof2,0,dof1, face[4*f + 2]);
			elmat.GetSubMatrix(dof1,dof1+dof2,dof1,dof1+dof2, face[4*f + 3]);
		}
	}

	for (int i=0; i<G; i++) {
		for (int j=0; j<G; j++) {
			delete local_mass_mats(i,j);
		}
	}
	TimingLog.Log("sweep assembly", timer.RealTime());
}

void DGLocalMatrices::AssembleTimeMass()
{
	if (HasTimeMass()) { return; }
	auto &mesh = *fes.GetParMesh();
	time_mass.resize(fes.GetNE());
	mfem::MassIntegrator mi;
	for (auto e=0; e<fes.GetNE(); e++) {
		const auto &fe = *fes.GetFE(e);
		auto &trans = *mesh.GetElementTransformation(e);
		LumpedIntegrationRule lumped_ir(trans.GetGeometryType());

		if (IsMassLumped(lump)) mi.SetIntegrationRule(lumped_ir);
		mi.AssembleElementMatrix(fe, trans, time_mass[e]);
	}
}

void FaceMassMatricesIntegrator::AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat)
{
	const auto dof1 = el1.GetDof();
	shape1.SetSize(dof1);
	int dof2 = 0;
	if (trans.Elem2No >= 0) {
		dof2 = el2.GetDof();
		shape2.SetSize(dof2);
	}
	const auto dof = dof1 + dof2;
	elmat.SetSize(dof);
	elmat = 0.0;

	const auto *ir = IntRule;
	if (ir == NULL) {
		int order;
		if (trans.Elem2No >= 0) {
			order = 2*std::max(el1.GetOrder(), el2.GetOrder());
		}
		else {
			order = 2*el1.GetOrder();
		}
		ir = &mfem::IntRules.Get(trans.GetGeometryType(), order);
	}

	for (auto n=0; n<ir->GetNPoints(); n++) {
		const auto &ip = ir->IntPoint(n);
		trans.SetAllIntPoints(&ip);
		double w = ip.weight * trans.Weight();

		const auto &eip1 = trans.GetElement1IntPoint();
		el1.CalcShape(eip1, shape1);
		for (auto i=0; i<dof1; i++) {
			for (auto j=0; j<dof1; j++) {
				elmat(i,j) += shape1(i) * shape1(j) * w;
			}
		}

		if (trans.Elem2No >= 0) {
			const auto &eip2 = trans.GetElement2IntPoint();
			el2.CalcShape(eip2, shape2);
			for (auto i=0; i<dof1; i++) {
				for (auto j=0; j<dof2; j++) {
					double val = shape1(i) * shape2(j) * w;
					elmat(i,j+dof1) += val;
					elmat(j+dof1,i) += val;
				}
			}

			for (auto i=0; i<dof2; i++) {
				for (auto j=0; j<dof2; j++) {
					elmat(i+dof1,j+dof1) += shape2(i) * shape2(j) * w;
				}
			}
		}
	}
}
