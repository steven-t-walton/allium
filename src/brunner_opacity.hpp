#pragma once 

#include <vector>
#include <cmath>

namespace BrunnerOpac
{

// A simplified frequency-dependent analytic opacity with edge- and line-like features.
// The form and background of this opacity can be found in
//    T. A. Brunner, "A Family of Multi-Dimensional Thermal Radiative Transfer Test Problems",
//    2023, LLNL-TR-858450
// This class stores the parameters and evaluates the frequency dependent opacity
// in Eq. 24 of that document.
class AnalyticEdgeOpacity
{
  public:
	// epsilonMin: below this photon energy, the opacity is a constant
	// epsilonEdge: the photon energy of the shell edge
	// C0: The overall strength of the opacity
	// C1: The strength of the edge
	// C2: The strength of the lines
	// lineWidth: if numLines>0, the width of line-like features in photon energy
	// lineSep: if numLines>0, the separation line-like features in photon energy
	// numLines: the number of line-like features.
	AnalyticEdgeOpacity(double epsilonMin_,
							  double epsilonEdge_,
							  double C0_,
							  double C1_,
							  double C2_,
							  double lineWidth_,
							  double lineSep_,
							  int numLines_)
		: epsilonMin{epsilonMin_},
		  epsilonEdge{epsilonEdge_},
		  C0{C0_},
		  C1{C1_},
		  C2{C2_},
		  lineWidth{lineWidth_},
		  lineSep{lineSep_},
		  numLines{numLines_}
	{
	}

	// Computes the frequency dependent specific opacity with units of area per mass,
	// $\kappa =\sigma/\rho$, where $\sigma$ is from Eq. 24.
	// epsilon: photon energy $\epsilon = h\nu$ in the same units as T
	// T: material temperature, really $k T$, in energy units.
	// rho: material density, mass per volume
	double computeKappa(const double epsilon, const double T, const double rho) const;

	// Computes the frequency dependent opacity with units of inverse length in Eq. 24
	// Same inputs as on computeKappa
	double computeSigma(const double epsilon, const double T, const double rho) const
	{
		return rho * computeKappa(epsilon, T, rho);
	}

	// Compute locations to break numerical integrations into subregions near features
	std::vector<double> computeBreaks() const;

  private:
	// The lowest photon energy used to compute an opacity,
	// mimicking a plasma frequency cut-off, $\epsilon_\text{min}$
	double epsilonMin;
	// The photon energy of the shell edge feature, $\epsilon_\text{edge}$
	double epsilonEdge;
	// A scaling factor to fit the opacity, $C_0$
	double C0;
	// The strength of the edge in the opacity, $C_1$
	double C1;
	// The strength of the lines in the opacity, $C_2$
	double C2;
	// The width of the line-like features, in photon energy, $\delta_w$
	double lineWidth;
	// The distance between the line-like features, in photon energy, $\delta_s$
	double lineSep;
	// The number of line-like features, $N_l$.  Can be zero.
	int numLines;
};

// Combine an AnalyticEdgeOpacity for a particular material and some group bounds to
// make it easy to compute multigroup integrals.  The group bounds are adjusted so that
// the lowest bound is zero and the upper bound is effectively infinity.
class MultiGroupIntegrator
{
  public:
	MultiGroupIntegrator(const AnalyticEdgeOpacity opac_, std::vector<double> groupBounds_)
		: opac{opac_},
		  opacBreaks{opac.computeBreaks()},
		  groupBounds{groupBounds_}
	{
		// We want to integrate from zero to infinity; upper bound is dealt with in the integration.
		groupBounds[0] = 0.0;
	}

	// Compute specific multigroup opacities with units of area per mass for a given
	// temperature $T$ and density $\rho$:
	// planckAverage = $\kappa_{P,g}=\sigma_{P,g}/\rho$, Eq. 9
	// rosselandAverage $\kappa_{R,g} = \sigma_{R,g}/\rho$, Eq. 12
	// b\_g = $b_g$, Eq. 19
	// dbdT\_g = $\partial b_g(T)/\partial T$, Eq. 20
	// planckMean is the grey (one group) Planck mean $\kappa_P$ specific opacity
	// rosselandMean is the grey (one group) Rosseland mean $\kappa_R$ specific opacity.
	void computeGroupAverages(const double T,
									  const double rho,
									  std::vector<double> &planckAverage,
									  std::vector<double> &rosselandAverage,
									  std::vector<double> &b_g,
									  std::vector<double> &dbdT_g,
									  double &planckMean,
									  double &rosselandMean) const;

	// An internal work routine to compute each integrand.
	void computeIntegrand(double epsilon, double rho, double T, double shift, double *results) const;

	// Merges the opacity breaks and ones needed for reasonable integration of the Planck function
	std::vector<double> computeAllSubRanges(double T) const;

	// filters the breaks from computeAllSubranges for ones in one group only defined
	// by lowBound and highBound.  safetyFactor is a relative tolerance for making sure
	// no subrange is too small.
	std::vector<double> filterRanges(double lowBound,
												double highBound,
												const std::vector<double> &allRanges,
												const double safetyFactor) const;

  private:
	// The detailed opacity we'll integrate.
	AnalyticEdgeOpacity opac;
	// integration sub-ranges proposed by opacity around edges and lines, etc.
	std::vector<double> opacBreaks;
	// The modified group bounds
	std::vector<double> groupBounds;
};

// Normalized photon energy where the total normalized Planck function
// will equal 1.0 using double precision numbers.
constexpr double cumulative_planck_max()
{
	return 46.435;
}

// $15/\pi^4$, a normalization in Eqs.~\ref{eq:normPlanck}-\ref{eq:normRoss}
constexpr double fifteen_over_pi_4()
{
	return 0.153989733820265027837291749007;
}

// Lowest normalized photon energy where we will consider shifting
constexpr double planck_shift_limit()
{
	return 36.1;
}

// Compute the integrand of Eq. 19, namely a normalized Eq. 3, where $x = h \nu/k T = \epsilon/T$
// is the normalized photon energy.  For values of $x>36.1$, the exponent can overflow.  We apply a
// transform to make it better behaved.  The optional shift can extend the range of $x$ where we can
// retain shape for weighting opacities even if magnitude is wrong.
inline double safePlanck(double x, double shift = 0.0)
{
	if (x < 1.0e-30)
	{
		return 0.0;
	}
	else if (x < planck_shift_limit())
	{
		return fifteen_over_pi_4() * x * x * x / std::expm1(x);
	}
	else
	{
		// For large $x$, we multiply and divide by $e^{-x}$ so $1-e^{-x}=1$
		// We also allow shifting to avoid denormalized numbers.
		double z = std::exp(-x + shift);
		return fifteen_over_pi_4() * z * x * x * x;
	}
}

// Compute the integrand of Eq 20, where $x = h \nu/k T = \epsilon/T$ is
// the normalized photon energy to safePlanck
inline double safeRoss(double x, double shift = 0.0)
{
	if (x < 1.0e-30)
	{
		return 0.0;
	}
	else if (x < planck_shift_limit())
	{
		// Multiply and divide by $e^{-2x}$ so everything stays well behaved.
		double expm1x = -std::expm1(-x);
		return 0.25 * fifteen_over_pi_4() * x * x * x * x * std::exp(-x) / (expm1x * expm1x);
	}
	else
	{
		// For large $x$, the denominator vanishes and we can shift.
		return 0.25 * fifteen_over_pi_4() * x * x * x * x * std::exp(-x + shift);
	}
}

} // namespace BrunnerOpac