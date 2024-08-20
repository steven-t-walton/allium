#include "gtest/gtest.h"
#include "moment_integrators.hpp"
#include "smm_integrators.hpp"
#include "sweep.hpp"
#include "lumping.hpp"
#include "trt_integrators.hpp"

TEST(Integrators, Penalty1D) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::BilinearForm Pform(&fes); 
	Pform.AddInteriorFaceIntegrator(new PenaltyIntegrator(1.0, false)); 
	Pform.AddBdrFaceIntegrator(new PenaltyIntegrator(1.0, false)); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	auto dense = std::unique_ptr<mfem::DenseMatrix>(P.ToDenseMatrix()); 

	mfem::DenseMatrix exact(6); 
	exact = 0.0; 
	exact(0,0) = 1.0; exact(5,5) = 1.0; 
	exact(1,1) = 1.0; exact(1,2) = -1.0; 
	exact(2,1) = -1.0; exact(2,2) = 1.0; 
	exact(3,3) = 1.0; exact(3,4) = -1.0; 
	exact(4,3) = -1.0; exact(4,4) = 1.0; 
	(*dense) -= exact; 
	auto norm = dense->MaxMaxNorm(); 
	EXPECT_TRUE(norm < 1e-14); 
}

TEST(Integrators, Penalty1DScale) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian1D(3, 1.0); 
	double hmin, hmax, kmin, kmax; 
	mesh.GetCharacteristics(hmin, hmax, kmin, kmax); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::BilinearForm Pform(&fes); 
	Pform.AddInteriorFaceIntegrator(new PenaltyIntegrator(1.0, true)); 
	Pform.AddBdrFaceIntegrator(new PenaltyIntegrator(1.0, true)); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	auto dense = std::unique_ptr<mfem::DenseMatrix>(P.ToDenseMatrix()); 

	mfem::DenseMatrix exact(6); 
	exact = 0.0; 
	exact(0,0) = 1.0; exact(5,5) = 1.0; 
	exact(1,1) = 1.0; exact(1,2) = -1.0; 
	exact(2,1) = -1.0; exact(2,2) = 1.0; 
	exact(3,3) = 1.0; exact(3,4) = -1.0; 
	exact(4,3) = -1.0; exact(4,4) = 1.0; 
	exact *= 1./hmin; 
	(*dense) -= exact; 
	auto norm = dense->MaxMaxNorm(); 
	EXPECT_TRUE(norm < 1e-14); 	
}

TEST(Integrators, JumpAverage2D) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace fes(&mesh, &fec), vfes(&mesh, &fec, dim); 

	mfem::MixedBilinearForm Pform(&vfes, &fes); 
	Pform.AddInteriorFaceIntegrator(new DGJumpAverageIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&fes), x(&fes); 
	u(0) = 1; 
	u(1) = 2; 
	u(2) = 5; 
	u(3) = 10; 
	v(0) = 5; 
	v(1) = 15; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2+1)*(5-15)/2; 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, VectorJumpJump2DX) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2, 1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::BilinearForm Pform(&vfes); 
	Pform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&vfes), x(&vfes); 
	u(0) = 1; 
	u(1) = 2; 
	u(2) = 10; 
	u(3) = 20; 
	v(0) = 3; 
	v(1) = 6; 
	v(2) = -10; 
	v(3) = -100; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2-1)*(6-3); 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, VectorJumpJump2DY) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1, 2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(0, dim, mfem::BasisType::GaussLegendre); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::BilinearForm Pform(&vfes); 
	Pform.AddInteriorFaceIntegrator(new DGVectorJumpJumpIntegrator); 
	Pform.Assemble(); 
	Pform.Finalize(); 
	const auto &P = Pform.SpMat(); 

	mfem::GridFunction u(&vfes), v(&vfes), x(&vfes); 
	u(2) = 1; 
	u(3) = 2; 
	u(0) = 10; 
	u(1) = 20; 
	v(2) = 3; 
	v(3) = 6; 
	v(0) = -10; 
	v(1) = -100; 
	P.Mult(u, x); 
	double inner = v*x; 
	double answer = (2-1)*(6-3); 
	EXPECT_DOUBLE_EQ(inner, answer); 
}

TEST(Integrators, SMMWeakDiv) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::DenseMatrix T(dim); 
	T(0,0) = 1.0; T(0,1) = T(1,0) = 0.25; T(1,1) = 2.0; 
	mfem::MatrixConstantCoefficient Tc(T); 
	WeakTensorDivergenceLFIntegrator wtd(Tc); 
	mfem::Vector elvec; 
	wtd.AssembleRHSElementVect(*fes.GetFE(0), *mesh.GetElementTransformation(0), elvec); 
	mfem::Vector exact({-0.625, 0.375, -0.375, 0.625, -1.125, -0.875, 0.875, 1.125}); 
	exact -= elvec; 
	double norm = exact.Norml2(); 
	EXPECT_DOUBLE_EQ(norm, 0.0); 
}

TEST(Integrators, SMMFaceTermInt) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::DenseMatrix T(dim); 
	T(0,0) = 1.0; T(0,1) = T(1,0) = 0.25; T(1,1) = 2.0; 
	mfem::MatrixConstantCoefficient Tc(T); 

	VectorJumpTensorAverageLFIntegrator lfint(Tc); 

	mfem::FaceElementTransformations *trans; 
	for (auto f=0; f<mesh.GetNumFaces(); f++) {
		trans = mesh.GetInteriorFaceTransformations(f); 
		if (trans) break; 
	}

	const auto &fe1 = *vfes.GetFE(trans->Elem1No); 
	const auto &fe2 = *vfes.GetFE(trans->Elem2No); 
	mfem::Vector elvec; 
	lfint.AssembleRHSElementVect(fe1, fe2, *trans, elvec); 
	mfem::Vector ex({0,-.5,0,-.5,0,-1./8,0,-1./8, .5,0,.5,0,1./8,0,1./8,0}); 
	ex -= elvec; 
	EXPECT_DOUBLE_EQ(ex.Norml2(), 0.0); 
}

TEST(Integrators, SMMFaceTermBdr) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 

	mfem::DenseMatrix T(dim); 
	T(0,0) = 1.0; T(0,1) = T(1,0) = 0.25; T(1,1) = 2.0; 
	mfem::MatrixConstantCoefficient Tc(T); 

	VectorJumpTensorAverageLFIntegrator lfint(Tc); 

	auto &trans = *mesh.GetBdrFaceTransformations(0); 
	const auto &fe1 = *vfes.GetFE(trans.Elem1No); 
	mfem::Vector elvec; 
	lfint.AssembleRHSElementVect(fe1, trans, elvec); 
	mfem::Vector ex({1./16, 1./16, 0, 0, 0.5, 0.5, 0, 0}); 
	ex -= elvec; 
	EXPECT_DOUBLE_EQ(ex.Norml2(), 0.0); 
}

TEST(Integrators, BoundaryNormalFaceLFIntegrator) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 
	mfem::ConstantCoefficient inflow(2.0); 
	BoundaryNormalFaceLFIntegrator lfi(inflow); 
	auto &trans = *mesh.GetBdrFaceTransformations(0); 
	mfem::Vector elvec; 
	lfi.AssembleRHSElementVect(*vfes.GetFE(trans.Elem1No), trans, elvec); 
	mfem::Vector ex({0,0,0,0,-0.5,-0.5,0,0}); 
	ex -= elvec; 
	EXPECT_DOUBLE_EQ(ex.Norml2(), 0.0); 
}

TEST(Integrators, ProjectedBoundaryNormalLF) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 
	auto f = [](const mfem::Vector &x) {
		return x(0)*x(1) + x(0) + x(1) + 2.0; 
	};
	mfem::FunctionCoefficient inflow(f); 
	BoundaryNormalFaceLFIntegrator lfi(inflow); 
	auto &trans = *mesh.GetBdrFaceTransformations(0); 
	mfem::Vector elvec; 
	lfi.AssembleRHSElementVect(*vfes.GetFE(trans.Elem1No), trans, elvec); 

	ProjectedCoefBoundaryNormalLFIntegrator pclfi(inflow, fec); 
	mfem::Vector pcelvec; 
	pclfi.AssembleRHSElementVect(*vfes.GetFE(trans.Elem1No), trans, pcelvec); 
	pcelvec -= elvec; 
	EXPECT_DOUBLE_EQ(pcelvec.Norml2(), 0.0); 
}

TEST(Integrators, ProjectedBoundaryLF) {
	mfem::Mesh mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace vfes(&mesh, &fec, dim); 
	auto f = [](const mfem::Vector &x) {
		return x(0)*x(1) + x(0) + x(1) + 2.0; 
	};
	mfem::FunctionCoefficient inflow(f); 
	mfem::BoundaryLFIntegrator lfi(inflow); 
	auto &trans = *mesh.GetBdrFaceTransformations(0); 
	mfem::Vector elvec; 
	lfi.AssembleRHSElementVect(*vfes.GetFE(trans.Elem1No), trans, elvec); 

	ProjectedCoefBoundaryLFIntegrator pclfi(inflow, fec); 
	mfem::Vector pcelvec; 
	pclfi.AssembleRHSElementVect(*vfes.GetFE(trans.Elem1No), trans, pcelvec); 
	pcelvec -= elvec; 
	EXPECT_DOUBLE_EQ(pcelvec.Norml2(), 0.0); 
}

TEST(Integrators, SweepFaceIntegrator) {
	auto mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::FaceElementTransformations *trans; 
	for (auto f=0; f<mesh.GetNumFaces(); f++) {
		trans = mesh.GetInteriorFaceTransformations(f); 
		if (trans) break; 
	}

	FaceMassMatricesIntegrator bfi; 
	mfem::DenseMatrix M; 
	bfi.AssembleFaceMatrix(*fes.GetFE(trans->Elem1No), *fes.GetFE(trans->Elem2No), *trans, M); 
	mfem::DenseMatrix ex11({{0,0,0,0}, {0,1./3,0,1./6}, {0,0,0,0}, {0,1./6,0,1./3}}); 
	mfem::DenseMatrix M11; 
	M.GetSubMatrix(0,4,M11); 
	ex11 -= M11; 
	EXPECT_NEAR(ex11.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix ex12({{0,0,0,0}, {1./3,0,1./6,0}, {0,0,0,0}, {1./6,0,1./3,0}}); 
	mfem::DenseMatrix M12; 
	M.GetSubMatrix(0,4,4,8,M12); 
	ex12 -= M12; 
	EXPECT_NEAR(ex12.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix M21; 
	M.GetSubMatrix(4,8,0,4,M21); 
	M21.Transpose(); 
	M21 -= M12; 
	EXPECT_NEAR(M21.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix ex22({{1./3,0,1./6,0}, {0,0,0,0}, {1./6,0,1./3,0}, {0,0,0,0}}); 
	mfem::DenseMatrix M22; 
	M.GetSubMatrix(4,8, M22); 
	ex22 -= M22; 
	EXPECT_NEAR(ex22.FNorm(), 0.0, 1e-14); 
}

TEST(Integrators, SweepFaceIntegratorLumped) {
	auto mesh = mfem::Mesh::MakeCartesian2D(2,1, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	mfem::FaceElementTransformations *trans; 
	for (auto f=0; f<mesh.GetNumFaces(); f++) {
		trans = mesh.GetInteriorFaceTransformations(f); 
		if (trans) break; 
	}

	QuadratureLumpedIntegrator bfi(new FaceMassMatricesIntegrator);
	mfem::DenseMatrix M; 
	bfi.AssembleFaceMatrix(*fes.GetFE(trans->Elem1No), *fes.GetFE(trans->Elem2No), *trans, M); 
	mfem::DenseMatrix ex11({{0,0,0,0}, {0,1./2,0,0.0}, {0,0,0,0}, {0,0.0,0,1./2}}); 
	mfem::DenseMatrix M11; 
	M.GetSubMatrix(0,4,M11); 
	ex11 -= M11; 
	EXPECT_NEAR(ex11.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix ex12({{0,0,0,0}, {1./2,0,0,0}, {0,0,0,0}, {0,0,1./2,0}}); 
	mfem::DenseMatrix M12; 
	M.GetSubMatrix(0,4,4,8,M12); 
	ex12 -= M12; 
	EXPECT_NEAR(ex12.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix M21; 
	M.GetSubMatrix(4,8,0,4,M21); 
	M21.Transpose(); 
	M21 -= M12; 
	EXPECT_NEAR(M21.FNorm(), 0.0, 1e-14); 

	mfem::DenseMatrix ex22({{1./2,0,0,0}, {0,0,0,0}, {0,0,1./2,0}, {0,0,0,0}}); 
	mfem::DenseMatrix M22; 
	M.GetSubMatrix(4,8, M22); 
	ex22 -= M22; 
	EXPECT_NEAR(ex22.FNorm(), 0.0, 1e-14); 
}

TEST(Integrators, TriMassMatrix) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	const auto &fe = *fes.GetFE(0);
	auto &trans = *mesh.GetElementTransformation(0); 
	mfem::DenseMatrix elmat; 
	mfem::MassIntegrator mi; 
	mi.AssembleElementMatrix(fe, trans, elmat); 
	elmat.Lump(); 

	mfem::DenseMatrix elmat_lump;
	QuadratureLumpedIntegrator lmi(&mi, 0);
	lmi.AssembleElementMatrix(fe, trans, elmat_lump); 
	elmat -= elmat_lump; 
	EXPECT_NEAR(elmat.FNorm(), 0.0, 1e-14); 
}

TEST(Integrators, MixedVecScalMass) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 0.25, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	const auto &fe = *fes.GetFE(0);
	const auto dof = fe.GetDof();
	auto &trans = *mesh.GetElementTransformation(0); 
	mfem::DenseMatrix elmat; 
	mfem::Vector coef_data(mesh.Dimension());
	for (int d=0; d<coef_data.Size(); d++) { coef_data(d) = d+1; }
	mfem::VectorConstantCoefficient coef(coef_data);
	MixedVectorScalarMassIntegrator mi(coef);
	mi.AssembleElementMatrix2(fe, fe, trans, elmat);

	mfem::MassIntegrator mass;
	mfem::DenseMatrix M;
	mass.AssembleElementMatrix(fe, trans, M);
	for (int d=0; d<mesh.Dimension(); d++) {
		mfem::DenseMatrix sub(dof,dof); 
		elmat.GetSubMatrix(d*dof, (d+1)*dof, 0, dof, sub);
		sub *= 1.0/coef_data(d);
		sub -= M; 
		EXPECT_NEAR(sub.FNorm(), 0.0, 1e-14);
	}
}

TEST(Integrators, LumpedMixedVecScalMass) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1,1, mfem::Element::QUADRILATERAL, true, 1.0, 0.25, false); 
	const auto dim = mesh.Dimension(); 
	mfem::L2_FECollection fec(1, dim, mfem::BasisType::GaussLobatto); 
	mfem::FiniteElementSpace fes(&mesh, &fec); 

	const auto &fe = *fes.GetFE(0);
	const auto dof = fe.GetDof();
	auto &trans = *mesh.GetElementTransformation(0); 
	LumpedIntegrationRule rule(trans.GetGeometryType());
	mfem::DenseMatrix elmat; 
	mfem::Vector coef_data(mesh.Dimension());
	for (int d=0; d<coef_data.Size(); d++) { coef_data(d) = d+1; }
	mfem::VectorConstantCoefficient coef(coef_data);
	MixedVectorScalarMassIntegrator mi(coef);
	mi.SetIntegrationRule(rule);
	mi.AssembleElementMatrix2(fe, fe, trans, elmat);

	mfem::MassIntegrator mass;
	mass.SetIntegrationRule(rule);
	mfem::DenseMatrix M;
	mass.AssembleElementMatrix(fe, trans, M);
	for (int d=0; d<mesh.Dimension(); d++) {
		mfem::DenseMatrix sub(dof,dof); 
		elmat.GetSubMatrix(d*dof, (d+1)*dof, 0, dof, sub);
		sub *= 1.0/coef_data(d);
		sub -= M; 
		EXPECT_NEAR(sub.FNorm(), 0.0, 1e-14);
	}
}