#include "gtest/gtest.h"
#include "block_diag_op.hpp"
#include "lumping.hpp"

TEST(BlockDiag, BlockMult) {
	auto mesh = mfem::Mesh::MakeCartesian2D(1, 1, mfem::Element::QUADRILATERAL, false, 1.0, 1.0); 
	auto fec = mfem::L2_FECollection(1, mesh.Dimension(), mfem::BasisType::GaussLobatto); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	DenseBlockDiagonalOperator A(fes), B(fes), C(fes);
	for (int e=0; e<mesh.GetNE(); e++) {
		A.GetBlock(e) = e;
		B.GetBlock(e) = e;
	}
	Mult(A,B,C);
	double sum = 0.0;
	for (int e=0; e<mesh.GetNE(); e++) {
		const auto &c = C.GetBlock(e);
		mfem::DenseMatrix ex(c.Height());
		ex = e*e;
		ex -= c;
		sum += ex.FNorm();
	}
	EXPECT_NEAR(sum, 0.0, 1e-12);
}

TEST(BlockDiag, SparseMult) {
	auto mesh = mfem::Mesh::MakeCartesian1D(10, 1.0);
	auto fec = mfem::L2_FECollection(1, mesh.Dimension());
	auto fes = mfem::FiniteElementSpace(&mesh, &fec);

	DenseBlockDiagonalOperator A(fes), B(fes), AB(fes), C(fes);
	mfem::MassIntegrator mi;
	mfem::DenseMatrix elmat;
	for (int e=0; e<fes.GetNE(); e++) {
		mi.AssembleElementMatrix(*fes.GetFE(e), *mesh.GetElementTransformation(e), elmat);
		A.SetBlock(e, elmat);
		B.SetBlock(e, elmat);
	}

	auto sB = std::unique_ptr<mfem::SparseMatrix>(B.AsSparseMatrix());
	Mult(A, *sB, C);
	Mult(A, B, AB);

	double norm = 0.0;
	for (int e=0; e<fes.GetNE(); e++) {
		auto &c = C.GetBlock(e);
		const auto &ab = AB.GetBlock(e);
		c -= ab; 
		norm += c.MaxMaxNorm();
	}
	EXPECT_DOUBLE_EQ(norm, 0.0);
}

TEST(BlockDiag, SparseMultLumped) {
	auto mesh = mfem::Mesh::MakeCartesian1D(10, 1.0);
	auto fec = mfem::L2_FECollection(1, mesh.Dimension());
	auto fes = mfem::FiniteElementSpace(&mesh, &fec);

	DenseBlockDiagonalOperator A(fes), B(fes), AB(fes), C(fes);
	mfem::DenseMatrix elmat;
	QuadratureLumpedIntegrator mi(new mfem::MassIntegrator);
	for (int e=0; e<fes.GetNE(); e++) {
		mi.AssembleElementMatrix(*fes.GetFE(e), *mesh.GetElementTransformation(e), elmat);
		A.SetBlock(e, elmat);
		B.SetBlock(e, elmat);
	}

	auto sB = std::unique_ptr<mfem::SparseMatrix>(B.AsSparseMatrix());
	Mult(A, *sB, C);
	Mult(A, B, AB);

	double norm = 0.0;
	for (int e=0; e<fes.GetNE(); e++) {
		auto &c = C.GetBlock(e);
		const auto &ab = AB.GetBlock(e);
		c -= ab; 
		norm += c.MaxMaxNorm();
	}
	EXPECT_DOUBLE_EQ(norm, 0.0);
}