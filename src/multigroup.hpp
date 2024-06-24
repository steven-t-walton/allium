#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"

// data structure class that stores information about 
// energy discretization 
class MultiGroupEnergyGrid {
private:
	mfem::Array<double> bounds; // G+1, defines energy grid 
	mfem::Array<double> midpoints; // midpoint of each group (could be log mid point) 
	mfem::Array<double> widths; // bin width 
public:
	// allow constrution by calling named constructor Make{Gray,LogSpaced,EqualSpaced} 
	// by move only, copy is deleted 
	// or by manually building the bounds and midpoints arrays 
	// models design of mfem::Mesh 
	MultiGroupEnergyGrid() = default;
	MultiGroupEnergyGrid(const mfem::Array<double> &grid, const mfem::Array<double> &midpoints);
	MultiGroupEnergyGrid(const MultiGroupEnergyGrid &other) = delete;
	MultiGroupEnergyGrid(MultiGroupEnergyGrid &&other) = default;
	MultiGroupEnergyGrid &operator=(MultiGroupEnergyGrid &&other) = default;
	MultiGroupEnergyGrid &operator=(MultiGroupEnergyGrid &other) = delete;

	double MeanEnergy(int group) const { return midpoints[group]; }
	double Width(int group) const { return widths[group]; }
	double LowerBound(int group) const { return bounds[group]; }
	double UpperBound(int group) const { return bounds[group+1]; }
	int Size() const { return midpoints.Size(); }
	double MinEnergy() const { return bounds[0]; }
	double MaxEnergy() const { return bounds.Last(); }

	const auto &Bounds() const { return bounds; }
	const auto &Midpoints() const { return midpoints; }
	const auto &Widths() const { return widths; }

	// 1 group 
	// Emin and Emax should be chosen to ensure the tails of the planck are captured 
	// so that multigroup planck -> gray planck 
	static MultiGroupEnergyGrid MakeGray(double Emin, double Emax);
	// G log spaced groups [Emin, Emax] (includes both end points)
	// if extend_to_zero = true: 1 group from 0.0 to Emin, G-1 log spaced to [Emin, Emax]
	static MultiGroupEnergyGrid MakeLogSpaced(double Emin, double Emax, int G, bool extend_to_zero=true);
	// G equally spaced groups [Emin, Emax]
	// if extend_to_zero = true: 1 group from 0.0 to Emin, G-1 equal spaced to [Emin, Emax]
	static MultiGroupEnergyGrid MakeEqualSpaced(double Emin, double Emax, int G, bool extend_to_zero=true);
};

class OpacityGroupCollapseOperator : public mfem::Operator {
private:
	mfem::FiniteElementSpace &fes; 
	const mfem::Array<double> &energy_grid; 
	mfem::VectorCoefficient *f; 
public:
	OpacityGroupCollapseOperator(mfem::FiniteElementSpace &sigma_space, const mfem::Array<double> &grid, 
		mfem::VectorCoefficient *weight_func=nullptr) 
	: fes(sigma_space), energy_grid(grid), f(weight_func) 
	{
		width = fes.GetVSize(); // G x space dofs 
		height = fes.GetNDofs(); // 1 x space dofs 
		assert(grid.Size() - 1 == fes.GetVDim()); 
	}

	void Mult(const mfem::Vector &sigma_mf, mfem::Vector &sigma_gray) const; 
};

class GroupCollapseOperator : public mfem::Operator {
private:
	const MomentVectorExtents &mg_ext;
	MomentVectorExtents gr_ext;
public:
	GroupCollapseOperator(const MomentVectorExtents &mg_ext);
	void Mult(const mfem::Vector &mg, mfem::Vector &gray) const override; 
	void MultTranspose(const mfem::Vector &gray, mfem::Vector &mg) const override;
};

// makes a moment vector look like a multigroup vector coefficient 
// for use in OpacityGroupCollapseOperator 
class MomentVectorMultiGroupCoefficient : public mfem::VectorCoefficient {
private:
	const mfem::FiniteElementSpace &fes;
	const MomentVectorExtents &phi_ext;
	const mfem::Vector &data;

	int moment_id = 0;
	mfem::Vector shape, local_data;
public:
	MomentVectorMultiGroupCoefficient(
		const mfem::FiniteElementSpace &fes, const MomentVectorExtents &phi_ext, const mfem::Vector &data);
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};