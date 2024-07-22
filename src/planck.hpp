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

constexpr std::array<double,9> inv_int1 = {
	1.0/2, 
	1.0/3, 
	1.0/4, 
	1.0/5, 
	1.0/6, 
	1.0/7, 
	1.0/8, 
	1.0/9
};

constexpr std::array<double,9> inv_int2 = {
	1.0/4, 
	1.0/9, 
	1.0/16, 
	1.0/25, 
	1.0/36, 
	1.0/49, 
	1.0/64, 
	1.0/81
};

constexpr std::array<double,9> inv_int3 = {
	1.0/8, 
	1.0/27, 
	1.0/64, 
	1.0/125, 
	1.0/216, 
	1.0/343, 
	1.0/512, 
	1.0/749
};

constexpr std::array<double,9> inv_int4 = {
	1.0/16, 
	1.0/81, 
	1.0/256, 
	1.0/625, 
	1.0/1296, 
	1.0/2401, 
	1.0/4096, 
	1.0/6561
};

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
	const double eix = std::exp(-x);
	double eixp = eix;
	double li1 = eix;
	double li2 = eix;
	double li3 = eix; 
	double li4 = eix;

	for (int l=2; l<=planck_polylog_degree; l++) {
		eixp *= eix;
		li1 += eixp * inv_int1[l-2];
		li2 += eixp * inv_int2[l-2];
		li3 += eixp * inv_int3[l-2]; 
		li4 += eixp * inv_int4[l-2];
	}
	return 1.0 - internal::coef * (6.0*li4 + x * (6.0*li3 + x*(3.0*li2 + x*li1)));
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