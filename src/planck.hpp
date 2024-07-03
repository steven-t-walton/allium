#pragma once 

#include "constants.hpp"
#include "mfem.hpp"

// "Computing multigroup radiation integrals using polylogarithm-based methods" Bradley Clark, JCP 1987
// https://doi.org/10.1016/0021-9991(87)90185-9 

// draco: https://github.com/lanl/Draco/blob/develop/src/cdi/CDI.hh 

namespace internal {

// generated in mathematica: HornerForm[Integrate[Series[x^3/(Exp[x] - 1), {x, 0, 21}], {x, 0, y}], y] 
constexpr double coef_3 = 1.0/3;
constexpr double coef_4 = -1.0/8; 
constexpr double coef_5 = 1.0/60;
constexpr double coef_7 = -1.0/5040; 
constexpr double coef_9 = 1.0/272160; 
constexpr double coef_11 = -1.0/13305600; 
constexpr double coef_13 = 1.0/622702080; 
constexpr double coef_15 = -6.91 / 196151155200;
constexpr double coef_17 = 1.0/1270312243200;
constexpr double coef_19 = -3.617/202741834014720;
constexpr double coef_21 = 43.867/107290978560589824;
constexpr double coef = 15.0/pow(constants::pi,4);
constexpr double rosseland_max = pow(std::numeric_limits<double>::max(), 0.25);

inline double PlanckTaylorSeries9(double x) {
	const double x2 = x*x;
	double taylor = x2 * coef_9 + coef_7;
	taylor = taylor * x2 + coef_5; 
	taylor = taylor * x + coef_4; 
	taylor = taylor * x + coef_3; 
	taylor *= x * x2 * coef;
	return taylor;
}

inline double PlanckTaylorSeries13(double x) {
	const double x2 = x*x;
	double taylor = x2 * coef_13 + coef_11;
	taylor = taylor * x2 + coef_9;
	taylor = taylor * x2 + coef_5; 
	taylor = taylor * x + coef_4; 
	taylor = taylor * x + coef_3; 
	taylor *= x * x2 * coef;
	return taylor;
}

inline double PlanckTaylorSeries17(double x) {
	const double x2 = x*x;
	double taylor = x2 * coef_17 + coef_15;
	taylor = taylor * x2 + coef_13; 
	taylor = taylor * x2 + coef_11; 
	taylor = taylor * x2 + coef_9;
	taylor = taylor * x2 + coef_7; 
	taylor = taylor * x2 + coef_5; 
	taylor = taylor * x + coef_4; 
	taylor = taylor * x + coef_3;
	taylor *= x * x2 * coef;
	return taylor;
}

inline double PlanckTaylorSeries21(double x) {
	const double x2 = x*x;
	double taylor = x2 * coef_21 + coef_19;
	taylor = taylor * x2 + coef_17; 
	taylor = taylor * x2 + coef_15; 
	taylor = taylor * x2 + coef_13; 
	taylor = taylor * x2 + coef_11; 
	taylor = taylor * x2 + coef_9;
	taylor = taylor * x2 + coef_7; 
	taylor = taylor * x2 + coef_5; 
	taylor = taylor * x + coef_4; 
	taylor = taylor * x + coef_3;
	taylor *= x * x2 * coef;
	return taylor;
}

constexpr int planck_taylor_degree = 21;
inline double PlanckTaylorSeries(double x) {
	if constexpr (planck_taylor_degree == 9) {
		return internal::PlanckTaylorSeries9(x);
	} else if constexpr (planck_taylor_degree == 13) {
		return internal::PlanckTaylorSeries13(x);
	} else if constexpr (planck_taylor_degree == 17) {
		return internal::PlanckTaylorSeries17(x);
	} else if constexpr (planck_taylor_degree == 21) {
		return internal::PlanckTaylorSeries21(x);
	} else {
		MFEM_ABORT("taylor series degree not defined");
		return -1.0;
	}
}

constexpr int planck_polylog_degree = 9;
inline double PlanckPolyLogarithmic(double x) {
	const double x2 = x*x;
	const double x3 = x2*x;
	const double d = -6.0;
	const double c = -6.0 * x; 
	const double b = -3 * x2;
	const double a = -x3;

	double val = 0.0;
	for (int l=1; l<=planck_polylog_degree; l++) {
		const double exp = std::exp(-x*l);
		const int l2 = l*l;
		const int l3 = l2*l;
		const int l4 = l3*l;
		val += (a/l + b/l2 + c/l3 + d/l4) * exp;
	}
	return 1.0 + internal::coef * val;
}

} // end namespace internal 

inline double IntegrateNormalizedPlanck(double x) {
	if (x > 1e100) return 1.0;
	if (x==0.0) return 0.0;
	const double taylor = internal::PlanckTaylorSeries(x);
	const double poly = internal::PlanckPolyLogarithmic(x);
	const double integral = std::min(taylor, poly);
	return integral;
}

inline double IntegrateNormalizedPlanck(double E, double T) {
	const double x = E/T;
	return IntegrateNormalizedPlanck(x);
}

inline double PlanckToRosseland(double x, double planck) {
	if (x > internal::rosseland_max) return planck;
	return planck - internal::coef/4.0 * pow(x,4) / (std::exp(x) - 1.0);
}

inline double IntegrateNormalizedRosseland(double x) {
	const double planck = IntegrateNormalizedPlanck(x);
	return PlanckToRosseland(x, planck);
}

inline double IntegrateNormalizedRosseland(double E, double T) {
	const double x = E/T;
	return IntegrateNormalizedRosseland(x);
}

void EvalPlanckSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum);
void EvalRosselandSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum);

class MultiGroupPlanckCoefficient : public mfem::VectorCoefficient {
private:
	const mfem::Array<double> &energy_grid; 
	mfem::Coefficient &T;
public:
	MultiGroupPlanckCoefficient(const mfem::Array<double> &energy_grid, mfem::Coefficient &T);
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};

class MultiGroupRosselandCoefficient : public mfem::VectorCoefficient {
private:
	const mfem::Array<double> &energy_grid; 
	mfem::Coefficient &T;
public:
	MultiGroupRosselandCoefficient(const mfem::Array<double> &energy_grid, mfem::Coefficient &T);
	void Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) override;
};