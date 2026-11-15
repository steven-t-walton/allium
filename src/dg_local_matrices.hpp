#pragma once

#include "mfem.hpp"
#include <vector>

class MultiGroupCoefficient;

// Element- and face-local matrices of the upwind DG transport discretization,
// shared by the sweep kernels and AdvectionOperator.
//
// For ordinate Omega and group g the element system on element e is
// 	A = sum_d Omega_d Grad(d,e) + sum_{outflow f} (Omega.n) Face(f,i,i) + Mass(g,e) [+ tau TimeMass(e)]
// where i is the side of face f that e is on, n is the unit normal out of e,
// and tau is the time absorption (e.g. 1/(c dt)). Inflow from the upwind
// element on side j enters the right hand side as -(Omega.n) Face(f,i,j) psi_j.
// The matrices are independent of Omega, so a kernel can form A for any
// ordinate (or batch of ordinates) without re-integration.
//
// Grad(d,e) is mfem's ConservativeConvectionIntegrator with velocity e_d, i.e.
// -(psi, e_d . grad v)_e. Face(f,i,j) is the face mass matrix (v_i, u_j)_f.
// Lumping follows the flags of lumping.hpp (mass, gradient and face lumping).
class DGLocalMatrices
{
private:
	mfem::ParFiniteElementSpace &fes;
	MultiGroupCoefficient &total;
	int lump;
	int G, dim;
	// mass weighted by the total opacity, indexed (g,e)
	std::vector<mfem::DenseMatrix> mass;
	// convection matrix for unit direction e_d, indexed (d,e)
	std::vector<mfem::DenseMatrix> grad;
	// face mass blocks indexed (f,i,j) where i,j in {0,1}
	// index the sides of the face (0 = element 1, 1 = element 2)
	std::vector<mfem::DenseMatrix> face;
	// unweighted mass used for time absorption, indexed e
	// empty until AssembleTimeMass is called
	std::vector<mfem::DenseMatrix> time_mass;
public:
	// fes: L2 space (face neighbor data is exchanged here)
	// total: total opacity, one component per group; evaluated in Assemble
	// lump: lumping type of lumping.hpp, 0 = none
	DGLocalMatrices(mfem::ParFiniteElementSpace &fes, MultiGroupCoefficient &total, int lump=0);

	// (re-)assemble mass, gradient, and face matrices
	// call after the total opacity changes
	void Assemble();
	// assemble time mass matrices once
	// no coefficient so the time step can change without re-assembly
	void AssembleTimeMass();
	bool HasTimeMass() const { return !time_mass.empty(); }

	int NumGroups() const { return G; }
	int GetLumpingType() const { return lump; }
	// sigma_{t,g} weighted mass matrix of element e
	const mfem::DenseMatrix &Mass(int g, int e) const { return mass[g*fes.GetNE() + e]; }
	// convection matrix of element e for the unit direction e_d
	const mfem::DenseMatrix &Grad(int d, int e) const { return grad[d*fes.GetNE() + e]; }
	// face mass block (test functions of side i, trial functions of side j) on face f
	// side 0 is element 1 of mfem's face information, side 1 element 2 (local or face neighbor)
	// only Face(f,0,0) exists on boundary faces
	const mfem::DenseMatrix &Face(int f, int i, int j) const { return face[4*f + 2*i + j]; }
	// unweighted mass matrix of element e, requires AssembleTimeMass
	const mfem::DenseMatrix &TimeMass(int e) const { return time_mass[e]; }
};

// assemble face mass matrices using the FaceElementTransformations object
// used in sweep by multiplying the face matrices by Omega.normal
// and using upwinding
class FaceMassMatricesIntegrator : public mfem::BilinearFormIntegrator
{
private:
	mfem::Vector shape1, shape2;
public:
	void AssembleElementMatrix(const mfem::FiniteElement&, mfem::ElementTransformation&, mfem::DenseMatrix&) {
		MFEM_ABORT("only defined for faces");
	}
	void AssembleFaceMatrix(const mfem::FiniteElement &el1, const mfem::FiniteElement &el2,
		mfem::FaceElementTransformations &trans, mfem::DenseMatrix &elmat);
};
