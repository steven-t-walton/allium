#include "brunner_opacity.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace BrunnerOpac
{

// Compute locations to break numerical integrations into subregions near features
std::vector<double> AnalyticEdgeOpacity::computeBreaks() const
{
	// These parameters specify how detailed to integrate the line-like features.
	// These were chosen to get accurate results, not speed.
	constexpr int numSigmas = 5;
	constexpr int breaksPerSigma = 3;

	std::vector<double> breakPoints;
	breakPoints.reserve(3 + 2 * numSigmas * breaksPerSigma * numLines);

	// Add the discontinuities in the opacity formula
	breakPoints.push_back(epsilonMin);
	breakPoints.push_back(epsilonEdge);

	// Capture the general shape of each Gaussian line
	for (int l = 0; l < numLines; ++l)
	{
		const double lineCenter = epsilonEdge - (l + 1) * lineSep;
		for (int s = -numSigmas * breaksPerSigma; s <= numSigmas * breaksPerSigma; ++s)
		{
			breakPoints.push_back(lineCenter + s * lineWidth / breaksPerSigma);
		}
	}

	std::sort(breakPoints.begin(), breakPoints.end());

	return breakPoints;
}

// Compute the opacity for a given frequency, temperature, and density
double AnalyticEdgeOpacity::computeKappa(const double epsilon, const double T, const double rho) const
{
	const double epsilonHat = std::max(epsilon, epsilonMin);

	const double term1 = C0 * rho / (std::sqrt(T) * epsilonHat * epsilonHat * epsilonHat);
	const double term2 = -std::expm1(-epsilonHat / T);
	const double term3 = 1.0 + (epsilonHat > epsilonEdge ? C1 : 0.0);

	double kappa = term1 * term2 * term3;

	for (int l = 0; l < numLines; ++l)
	{
		const double lineCenter = epsilonEdge - (l + 1) * lineSep;
		const double normPos = (epsilonHat - lineCenter) / lineWidth;
		kappa += term1 * term2 * C2 * std::exp(-0.5 * normPos * normPos) / (numLines - l);
	}

	return kappa;
}

// We have internal fixed-sized work arrays of this length.
constexpr int maxVars()
{
	return 16;
}

// Standard Gauss integration using 16 points over the range [a,b].  The function $f$
// evaluates numVars integrands and stores them in result.  numVars must be less than maxVars().
// result is assumed to be zeroed on entry.
inline void gauss16Integrate(std::function<void(double, double *, int)> f,
									  const double a,
									  const double b,
									  const int numVars,
									  double *result)
{
	constexpr int glOrder{16};
	constexpr std::array<double, glOrder> muGauss{
		-0.9894009349916499,
		-0.9445750230732326,
		-0.8656312023878318,
		-0.755404408355003,
		-0.6178762444026438,
		-0.45801677765722737,
		-0.2816035507792589,
		-0.09501250983763743,
		0.09501250983763754,
		0.2816035507792589,
		0.4580167776572275,
		0.6178762444026438,
		0.7554044083550031,
		0.8656312023878316,
		0.9445750230732326,
		0.9894009349916499,
	};

	constexpr std::array<double, glOrder> wtGauss{
		0.02715245941175411,
		0.0622535239386479,
		0.09515851168249277,
		0.12462897125553386,
		0.1495959888165767,
		0.16915651939500254,
		0.18260341504492358,
		0.1894506104550685,
		0.1894506104550685,
		0.18260341504492358,
		0.16915651939500254,
		0.1495959888165767,
		0.12462897125553386,
		0.09515851168249277,
		0.0622535239386479,
		0.02715245941175411,
	};

	double mean = 0.5 * (b + a);
	double scale = 0.5 * (b - a);
	std::array<double, maxVars()> eval;

	for (int i = 0; i < glOrder; ++i)
	{
		f(scale * muGauss[i] + mean, eval.data(), numVars);
		for (int j = 0; j < numVars; ++j)
		{
			result[j] += scale * wtGauss[i] * eval[j];
		}
	}
}

// Main user access to compute opacities
void MultiGroupIntegrator::computeGroupAverages(const double T,
																const double rho,
																std::vector<double> &planckAverage,
																std::vector<double> &rosselandAverage,
																std::vector<double> &b_g,
																std::vector<double> &dbdT_g,
																double &planckMean,
																double &rosselandMean) const
{
	int numGroups = groupBounds.size() - 1;

	planckAverage.resize(numGroups);
	rosselandAverage.resize(numGroups);
	b_g.resize(numGroups);
	dbdT_g.resize(numGroups);

	const std::vector<double> allSubRanges = computeAllSubRanges(T);

	// Work array for grey opacities.
	std::array<double, 4> globalResults{0.0, 0.0, 0.0, 0.0};
	for (int g = 0; g < numGroups; ++g)
	{
		const double lowBound = groupBounds[g];
		// Make sure upper bound, if we're on the last group, is at least "infinity" as far
		// as the cumulative integral of the double precision Planck function is concerned
		const double highBound = g + 1 < numGroups ? groupBounds[g + 1]
																 : std::max(groupBounds[numGroups], cumulative_planck_max() * T);
		// If the normalized lower bound of the group is above the range where the Planck
		// function behaves exponentially, set the shift to be the lower group bound.
		// One shift needs to be applied to all evaluations within a group for it to
		// cancel in the numerator and denominator of Eq. 9 and Eq. 12.
		const double shift = lowBound / T > planck_shift_limit() ? lowBound / T : 0.0;

		// This is very conservative, but works.   Making it bigger might speed the code up.
		constexpr double safetyFactor = 1.0e-6;
		std::vector<double> subGroupRanges = filterRanges(lowBound, highBound, allSubRanges, safetyFactor);

		// Work array for group averages.
		std::array<double, 8> groupResults{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (size_t sr = 0; sr < subGroupRanges.size() - 1; ++sr)
		{
			std::array<double, 8> localResults{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
			const int numVars = localResults.size();

			auto opacFunc = [&](double epsilon, double *results, int)
			{ computeIntegrand(epsilon, rho, T, shift, results); };
			gauss16Integrate(opacFunc, subGroupRanges[sr], subGroupRanges[sr + 1], numVars, localResults.data());

			for (int i = 0; i < numVars; ++i)
			{
				groupResults[i] += localResults[i];
			}
			for (int i = 0; i < 4; ++i)
			{
				globalResults[i] += localResults[i + 4];
			}
		}

		if (groupResults[1] > 0.0)
		{
			planckAverage[g] = groupResults[0] / groupResults[1];
		}
		else
		{
			// For very large epsilon = groupBounds/T, the denominator is zero
			// The weight is also a rapidly decaying exponetial, so the average
			// will essentially be at the left boundary of the group.
			planckAverage[g] = opac.computeKappa(groupBounds[g] / T, T, rho);
		}

		if (groupResults[3] > 0.0)
		{
			rosselandAverage[g] = groupResults[2] / groupResults[3];
		}
		else
		{
			// For very large epsilon = groupBounds/T, the denominator is zero
			// The weight is also a rapidly decaying exponetial, so the average
			// will essentially be at the left boundary of the group.
			rosselandAverage[g] = opac.computeKappa(groupBounds[g] / T, T, rho);
		}

		// We need unshifted values the for total emission calculation
		b_g[g] = groupResults[5];
		dbdT_g[g] = groupResults[6];
	}
	// Compute grey results.
	if (globalResults[0] > 0.0 and globalResults[1] > 0.0)
	{
		planckMean = globalResults[0] / globalResults[1];
	}
	else
	{
		planckMean = 0.0;
	}
	if (globalResults[0] > 0.0 and globalResults[3] > 0.0)
	{
		rosselandMean = globalResults[2] / globalResults[3];
	}
	else
	{
		rosselandMean = 0.0;
	}
}

// We compute eight different integrals simultaneously for different needs
void MultiGroupIntegrator::computeIntegrand(double epsilon, double rho, double T, double shift, double *results) const
{
	const double kappa = opac.computeKappa(epsilon, T, rho);

	// To get accurate group evaluations at large photon energy,
	// we shift the weight functions to keep the shape be remain nonzero.
	const double planck = safePlanck(epsilon / T, shift);
	const double rosseland = safeRoss(epsilon / T, shift);
	results[0] = kappa * planck;
	results[1] = planck;
	results[2] = rosseland;
	results[3] = rosseland / kappa;

	// But for the grey opacities and the evaluation of the Planck
	// integrals, we don't want shifting.
	const double planckUS = safePlanck(epsilon / T, 0.0);
	const double rosselandUS = safeRoss(epsilon / T, 0.0);
	results[4] = kappa * planckUS;
	results[5] = planckUS;
	results[6] = rosselandUS;
	results[7] = rosselandUS / kappa;
}

// Merges the opacity breaks and ones needed for reasonable integration of the Planck function.
std::vector<double> MultiGroupIntegrator::computeAllSubRanges(double T) const
{
	std::vector<double> proposedBreaks(opacBreaks);

	// Normalized photon energy breaks to accurately integrate the Planck function
	std::array<double, 28> planckBreaks{
		0.0, 0.25,   0.5,  1.0,  1.69, 2.0,   2.34, 2.82,  2.92, 3.5,  3.83, 4.45, 4.97, 6.19,
		9.0, 12.965, 14.0, 16.0, 18.0, 21.26, 25.0, 29.07, 31.0, 34.0, 38.0, 42.0, 46.0, 50.0,
	};
	for (double ab : planckBreaks)
	{
		proposedBreaks.push_back(ab * T);
	}
	std::sort(proposedBreaks.begin(), proposedBreaks.end());
	return proposedBreaks;
}

// returns a set of "mini" group bounds breaking the current group into sub ranges
std::vector<double> MultiGroupIntegrator::filterRanges(double lowBound,
																		 double highBound,
																		 const std::vector<double> &allRanges,
																		 const double safetyFactor) const
{
	std::vector<double> subRanges(allRanges.size());
	subRanges.resize(0);

	subRanges.push_back(lowBound);
	double lastBreak = lowBound;

	for (double b : allRanges)
	{
		if (lastBreak * (1.0 + safetyFactor) < b and b < highBound * (1.0 - safetyFactor))
		{
			subRanges.push_back(b);
			lastBreak = b;
		}
	}
	subRanges.push_back(highBound);
	return subRanges;
}

} // namespace BrunnerOpac
