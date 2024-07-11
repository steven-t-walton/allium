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

class WeightedGroupCollapseOperator : public mfem::Operator {
private:
	const mfem::FiniteElementSpace &fes;
	const MomentVectorExtents &phi_ext;
	mfem::VectorCoefficient &f;

	mutable mfem::Vector spectrum;
public:
	WeightedGroupCollapseOperator(
		const mfem::FiniteElementSpace &fes, 
		const MomentVectorExtents &phi_ext,
		mfem::VectorCoefficient &f);
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

// a coefficient type intended for multigroup data such as opacities 
// this class differs from mfem::VectorCoefficient by implementing 
// a system to create a scalar coefficient that represents a single group 
// the vector dimension is the number of groups 
// the base class requires implementing the scalar evaluation in 
// group g and has a naive method to compute all groups at once 
// derived classes have the option to use this naive evaluation 
// or implement an optimized vector evaluation that reuses 
// computation common to all groups 
class MultiGroupCoefficient : public mfem::VectorCoefficient
{
public:
	MultiGroupCoefficient(int G) : mfem::VectorCoefficient(G)
	{ }

	// required scalar evaluation in group g 
	virtual double Eval(int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
	{
		MFEM_ABORT("scalar eval not implemented");
	}
	// naive vector eval by calling Eval(g, trans, ip) G times 
	virtual void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override
	{
		v.SetSize(vdim);
		for (int g=0; g<vdim; g++) {
			v(g) = Eval(g, trans, ip);
		}
	}
	// subclass that allows creating a coefficient that 
	// calls Eval(g, trans, ip) given group index g 
	class GroupCoefficient : public mfem::Coefficient 
	{
	private:
		MultiGroupCoefficient &mg_coef; 
		int g; 
	public:
		GroupCoefficient(MultiGroupCoefficient &mg_coef, int g)
			: mg_coef(mg_coef), g(g)
		{ }
		double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override 
		{
			return mg_coef.Eval(g, trans, ip);
		}
	};

	// get a scalar mfem::Coefficient in group g 
	// caller has ownership of returned coefficient 
	// and must call delete 
	virtual mfem::Coefficient *GetGroupCoefficient(int g) 
	{
		return new GroupCoefficient(*this, g);
	}
};

class GridFunctionMGCoefficient : public MultiGroupCoefficient
{
private:
	const mfem::GridFunction &gf;
	mfem::VectorGridFunctionCoefficient vec_coef;
public:
	GridFunctionMGCoefficient(const mfem::GridFunction &gf)
		: gf(gf), vec_coef(&gf), MultiGroupCoefficient(gf.VectorDim())
	{ }
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override 
	{
		vec_coef.Eval(v, trans, ip);
	}
	mfem::GridFunctionCoefficient *GetGroupCoefficient(int g) override
	{
		return new mfem::GridFunctionCoefficient(&gf, g+1); 
	}
	const mfem::GridFunction &GetGridFunction() const { return gf; }
};

class PlanckSpectrumMGCoefficient : public MultiGroupCoefficient {
private:
	const mfem::Array<double> &energy_grid; 
	mfem::Coefficient &T;
public:
	PlanckSpectrumMGCoefficient(const mfem::Array<double> &energy_grid, mfem::Coefficient &T);

	// scalar evaluation 
	double Eval(int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override; 
	// efficient vector evaluation 
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

class RosselandSpectrumMGCoefficient : public MultiGroupCoefficient {
private:
	const mfem::Array<double> &energy_grid; 
	mfem::Coefficient &T;
public:
	RosselandSpectrumMGCoefficient(const mfem::Array<double> &energy_grid, mfem::Coefficient &T);
	
	// scalar evaluation 
	double Eval(int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override; 
	// efficient vector evaluation 
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

class GrayMGCoefficient : public MultiGroupCoefficient {
private:
	mfem::Coefficient &coef;
	int G;
public:
	GrayMGCoefficient(mfem::Coefficient &coef, int G)
		: coef(coef), G(G), MultiGroupCoefficient(G)
	{ }
	double Eval(int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override 
	{
		return coef.Eval(trans, ip);
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override
	{
		const auto val = coef.Eval(trans, ip);
		v = val;
	}
};

class ConstantMGCoefficient : public MultiGroupCoefficient {
private:
	const mfem::Vector &constants; 
public:
	ConstantMGCoefficient(const mfem::Vector &constants)
		: constants(constants), MultiGroupCoefficient(constants.Size())
	{ }
	double Eval(int g, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override
	{
		return constants(g);
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override
	{
		v = constants;
	}
};