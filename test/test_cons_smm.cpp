#include "gtest/gtest.h"
#include "cons_smm_op.hpp"
#include "smm_integrators.hpp"
#include "dg_trace_coll.hpp"
#include "transport_op.hpp"
#include "sweep.hpp"
#include "p1diffusion.hpp"
#include "block_smm_op.hpp"

TEST(DGTraceColl, Quad) {
	auto mesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::QUADRILATERAL, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	auto fec = DGTrace_FECollection(1, dim); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	const auto *fe = fes.GetTraceElement(0, mesh.GetFaceGeometry(0)); 
	EXPECT_TRUE(fe); 
	EXPECT_EQ(fe->GetDof(), 2); 
	EXPECT_EQ(fes.GetVSize(), fes.GetNE()*8); 
}

TEST(DGTraceColl, Tri) {
	auto mesh = mfem::Mesh::MakeCartesian2D(2,2, mfem::Element::TRIANGLE, true, 1.0, 1.0, false); 
	const auto dim = mesh.Dimension(); 
	auto fec = DGTrace_FECollection(1, dim); 
	auto fes = mfem::FiniteElementSpace(&mesh, &fec); 
	const auto *fe = fes.GetTraceElement(0, mesh.GetFaceGeometry(0)); 
	EXPECT_TRUE(fe); 
	EXPECT_EQ(fe->GetDof(), 2); 
	EXPECT_EQ(fes.GetVSize(), fes.GetNE() * 6); 
}
