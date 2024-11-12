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
constexpr double second_deriv_max = std::log(std::numeric_limits<double>::max())-1.0;

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

// precompute coefficients used in polylogarithm calculations
// divisions done at compile time => only need multiplications at runtime 
constexpr std::array<double,9> inv_int = {
	1.0/2, 
	1.0/3, 
	1.0/4, 
	1.0/5, 
	1.0/6, 
	1.0/7, 
	1.0/8, 
	1.0/9
};

constexpr int planck_polylog_degree = 9;
// computes polylogarithm approximation to normalized 
// planck integral from [0,x]
// approximation is a combination of the first 4 
// polylogarithms evaluated using the series 
// L_a(x) = sum_{l=1}^{L} e^{-l*x}/l^a 
inline double PlanckPolyLogarithmic(double x) {
	static_assert(planck_polylog_degree <= inv_int.size());
	const double eix = std::exp(-x);

	// evaluate l=1 case 
	double eixp = eix;
	double li1 = eix;
	double li2 = eix;
	double li3 = eix; 
	double li4 = eix;

	// l >= 2 
	for (int l=2; l<=planck_polylog_degree; l++) {
		double inv = inv_int[l-2]; // 1.0/l 
		// e^{-l*x} in numerator
		// evaluate using product to avoid 
		// more expensive std::exp evals 
		eixp *= eix; 
		// use inv_int array to avoid divisions 
		double t = eixp * inv; // e^{-l*x}/l 

		// Li1 
		li1 += t; 

		// Li2 
		t *= inv; // e^{-l*x}/l^2 
		li2 += t; 

		// Li3 
		t *= inv; // e^{-l*x}/l^3 
		li3 += t; 

		// Li4 
		t *= inv; // e^{-l*x}/l^4 
		li4 += t;
	}
	// combine Li's into polylog approximation of b(x) 
	// use Horner-type evaluation to avoid pow calls 
	return 1.0 - internal::coef * (6.0*li4 + x * (6.0*li3 + x*(3.0*li2 + x*li1)));
}

} // end namespace internal 

inline double IntegrateNormalizedPlanck(double x) {
	assert(x >= 0.0);
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
	return planck - 0.25*internal::coef * pow(x,4) / std::expm1(x);
}

inline double IntegrateNormalizedRosseland(double x) {
	const double planck = IntegrateNormalizedPlanck(x);
	return PlanckToRosseland(x, planck);
}

inline double IntegrateNormalizedRosseland(double E, double T) {
	const double x = E/T;
	return IntegrateNormalizedRosseland(x);
}

inline double PlanckToSecondDerivative(double x, double planck) {
	if (x > internal::second_deriv_max) return planck;
	const auto ex = std::exp(x);
	const auto val = planck - internal::coef/12*std::pow(x,4)*(ex*(x+3.0) - 3.0)/std::pow(std::expm1(x), 2);
	if (std::isnan(val)) return planck; 
	else return val;
}

inline double IntegrateNormalizedPlanckSecondDerivative(double x) {
	const double planck = IntegrateNormalizedPlanck(x);
	return PlanckToSecondDerivative(x, planck);
}

inline double IntegrateNormalizedPlanckSecondDerivative(double E, double T) {
	const double x = E/T;
	return IntegrateNormalizedPlanckSecondDerivative(x);
}

inline double PlanckFunction(double E, double T)
{
	const auto x = E/T;
	if (x < 1e-30) return 0.0; 
	else return internal::coef * x * x * x / std::expm1(x);
}

namespace internal {

// evaluate spectrum over all groups at a given temperature 
// templated on function F(E,T) that returns int_0^E f dE 
// so that F_g = F(E_{g+1},T) - F(E_g,T) 
// this function minimizes calls to F by re-using evaluations 
// from the previous group 
template<double (*F)(double,double)>
inline void EvalSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	const auto G = energy_grid.Size() - 1;
	spectrum.SetSize(G);
	double prev = 0.0; 
	for (int g=0; g<G; g++) {
		double next = F(energy_grid[g+1], T);
		spectrum(g) = next - prev;
		prev = next;
	}
}

} // end namespace internal 

// evaluate planck spectrum in all groups
inline void EvalPlanckSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	internal::EvalSpectrum<IntegrateNormalizedPlanck>(energy_grid, T, spectrum);
}
// evaluate rosseland spectrum in all groups
inline void EvalRosselandSpectrum(const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	internal::EvalSpectrum<IntegrateNormalizedRosseland>(energy_grid, T, spectrum);
}
inline void EvalPlanckSecondDerivativeSpectrum(
	const mfem::Array<double> &energy_grid, double T, mfem::Vector &spectrum)
{
	internal::EvalSpectrum<IntegrateNormalizedPlanckSecondDerivative>(energy_grid, T, spectrum);
}
// verify group structure produces integral of 
// normalized planck from Emin to Emax = 1.0 
// for max/min temperatures in the domain
// outputs warning if not 
void CheckPlanckSpectrumCovered(MPI_Comm comm, double Emin, double Emax, 
	const mfem::Vector &temperature, double tol);