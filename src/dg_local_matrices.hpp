#pragma once

#include "mfem.hpp"
#include <vector>

class MultiGroupCoefficient;

// element- and face-local DG upwind transport matrices
// shared by the sweep kernels and the advection operator
// the angle-dependent local operator is assembled from these as
// 	sum_d Omega_d Grad(d,e) + sum_{outflow f} (Omega.n) Face(f,i,i) + Mass(g,e) [+ sigma_t TimeMass(e)]
// with inflow coupling -(Omega.n) Face(f,i,j)
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
	const mfem::DenseMatrix &Mass(int g, int e) const { return mass[g*fes.GetNE() + e]; }
	const mfem::DenseMatrix &Grad(int d, int e) const { return grad[d*fes.GetNE() + e]; }
	const mfem::DenseMatrix &Face(int f, int i, int j) const { return face[4*f + 2*i + j]; }
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
