#pragma once 

#include "mfem.hpp"

class MatrixTransposeVectorProductCoefficient : public mfem::VectorCoefficient
{
private:
	mfem::MatrixCoefficient &mat; 
	mfem::VectorCoefficient &vec;

	mfem::Vector vec_eval; 
	mfem::DenseMatrix mat_eval; 
public:
	MatrixTransposeVectorProductCoefficient(mfem::MatrixCoefficient &mat, mfem::VectorCoefficient &vec)
		: mat(mat), vec(vec), mfem::VectorCoefficient(mat.GetWidth())
	{
		if (mat.GetHeight() != vec.GetVDim()) 
			MFEM_ABORT("mismatch");
		vec_eval.SetSize(vec.GetVDim());
		mat_eval.SetSize(mat.GetHeight(), mat.GetWidth());
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
	{
		v.SetSize(vdim);
		vec.Eval(vec_eval, trans, ip);
		mat.Eval(mat_eval, trans, ip);
		mat_eval.MultTranspose(vec_eval, v);
	}
};

class RowL2NormVectorCoefficient : public mfem::VectorCoefficient
{
private:
	mfem::MatrixCoefficient &coef;

	mfem::DenseMatrix eval;
public:
	RowL2NormVectorCoefficient(mfem::MatrixCoefficient &coef)
		: coef(coef), mfem::VectorCoefficient(coef.GetHeight())
	{
		eval.SetSize(coef.GetHeight(), coef.GetWidth());
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
	{
		v.SetSize(vdim);
		coef.Eval(eval, trans, ip);
		for (int i=0; i<eval.Height(); i++) {
			v(i) = 0.0;
			for (int j=0; j<eval.Width(); j++) {
				v(i) += eval(i,j)*eval(i,j);
			}
			v(i) = std::sqrt(v(i));
		}
	}
};

class RowL1NormVectorCoefficient : public mfem::VectorCoefficient
{
private:
	mfem::MatrixCoefficient &coef;

	mfem::DenseMatrix eval;
public:
	RowL1NormVectorCoefficient(mfem::MatrixCoefficient &coef)
		: coef(coef), mfem::VectorCoefficient(coef.GetHeight())
	{
		eval.SetSize(coef.GetHeight(), coef.GetWidth());
	}
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
	{
		v.SetSize(vdim);
		coef.Eval(eval, trans, ip);
		for (int i=0; i<eval.Height(); i++) {
			v(i) = 0.0;
			for (int j=0; j<eval.Width(); j++) {
				v(i) += std::fabs(eval(i,j));
			}
		}
	}
};