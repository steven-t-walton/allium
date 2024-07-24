#pragma once 

#include "mfem.hpp"
#include "tvector.hpp"
#include "angular_quadrature.hpp"
#include "multigroup.hpp"

class SMMCorrectionTensorCoefficient : public mfem::MatrixArrayCoefficient
{
private:
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	ConstTransportVectorView psi; 
public:
	mfem::Array<mfem::ParGridFunction*> gfs; 

	SMMCorrectionTensorCoefficient(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi);
	~SMMCorrectionTensorCoefficient(); 
};

class SecondMomentTensorCoefficient : public mfem::MatrixArrayCoefficient
{
private:
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature	&quad; 
	ConstTransportVectorView psi; 
	mfem::Array<mfem::ParGridFunction*> gfs; 
public:
	SecondMomentTensorCoefficient(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, ConstTransportVectorView _psi); 
	~SecondMomentTensorCoefficient(); 
};

class MatrixDivergenceGridFunctionCoefficient : public mfem::VectorCoefficient
{
private:
	mfem::MatrixArrayCoefficient &T; 
	mfem::DenseMatrix grad; 
public:
	MatrixDivergenceGridFunctionCoefficient(mfem::MatrixArrayCoefficient &t) 
		: T(t), mfem::VectorCoefficient(t.GetHeight()) 
	{
		grad.SetSize(vdim, vdim*vdim); 
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &T, const mfem::IntegrationPoint &ip); 
};

class SMMBdrCorrectionFactorCoefficient : public mfem::Coefficient 
{
private:
	mfem::ParFiniteElementSpace &fes; 
	const AngularQuadrature &quad; 
	ConstTransportVectorView psi; 
	double alpha; 

	int dim; 
	mfem::Vector nor, shape; 
public:
	SMMBdrCorrectionFactorCoefficient(mfem::ParFiniteElementSpace &_fes, const AngularQuadrature &_quad, 
		ConstTransportVectorView _psi, double _alpha=0.5); 
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip); 
};

class GrayInflowPartialCurrentCoefficient : public mfem::Coefficient 
{
private:
	PhaseSpaceCoefficient &phase_coef; 
	const AngularQuadrature &quad; 
	double scale; 

	mfem::Vector nor; 
public:
	GrayInflowPartialCurrentCoefficient(PhaseSpaceCoefficient &pc, const AngularQuadrature &q, double s=1.0) 
		: phase_coef(pc), quad(q), scale(s)
	{ }

	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override; 
};

class InflowPartialCurrentCoefficient : public mfem::VectorCoefficient
{
private:
	PhaseSpaceCoefficient &phase_coef;
	const AngularQuadrature &quad; 
	const MultiGroupEnergyGrid &energy_grid;
	double scale; 

	mfem::Vector nor, spectrum;
public:
	InflowPartialCurrentCoefficient(PhaseSpaceCoefficient &pc, const AngularQuadrature &quad, 
		const MultiGroupEnergyGrid &grid, double s=1.0)
		: phase_coef(pc), quad(quad), energy_grid(grid), scale(s), mfem::VectorCoefficient(grid.Size())
	{
		spectrum.SetSize(vdim);
		nor.SetSize(quad.Dimension());
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

class VectorComponentSumCoefficient : public mfem::Coefficient
{
private:
	mfem::VectorCoefficient &coef;

	mfem::Vector eval;
public:
	VectorComponentSumCoefficient(mfem::VectorCoefficient &coef)
		: coef(coef)
	{
		eval.SetSize(coef.GetVDim());
	}
	double Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override
	{
		coef.Eval(eval, trans, ip);
		return eval.Sum();
	}
};