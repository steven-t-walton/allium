#include "opacity.hpp"

PWOpacityCoefficient::PWOpacityCoefficient(const mfem::Array<int> &attrs, const mfem::Array<OpacityCoefficient*> &coefs)
	: OpacityCoefficient(coefs[0]->GetVDim())
{
	assert(attrs.Size() == coefs.Size()); 
	for (auto i=0; i<attrs.Size(); i++) {
		map[attrs[i]] = coefs[i]; 
	}
}

void PWOpacityCoefficient::SetTemperature(mfem::Coefficient &T) 
{
	for (auto &it : map) {
		if (it.second) it.second->SetTemperature(T); 
	}
}

void PWOpacityCoefficient::SetDensity(mfem::Coefficient &rho) 
{
	for (auto &it : map) {
		if (it.second) it.second->SetDensity(rho); 
	}
}

void PWOpacityCoefficient::Eval(mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip)
{
	const auto attr = trans.Attribute; 
	auto *ptr = map.at(attr); 
	ptr->Eval(v, trans, ip); 
}

double EdgeLineOpacityFunction::operator()(double rho, double T, double E) const
{
	const double Ehat = std::max(E, Emin);
	const double coef = c0 * rho*rho * std::pow(T, nT) / std::pow(Ehat, 3) * (1.0 - std::exp(-Ehat/T));
	double edge = 1.0 + (Ehat > Eedge ? c1 : 0.0);

	double lines = 0.0;
	for (int l=0; l<Nlines; l++) {
		const auto center = Eedge - (l+1)*delta_s;
		const auto norm_pos = (Ehat - center) / delta_w;
		lines += std::exp(-0.5*norm_pos*norm_pos) / (Nlines - l);
	}
	lines *= c2;
	return coef * (edge + lines);
}

void MultiGroupFunctionOpacityCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
{
	const auto T = temperature->Eval(trans, ip);
	const auto rho = density->Eval(trans, ip);
	v.SetSize(vdim);
	if (!rule) {
		for (int g=0; g<vdim; g++) {
			double ip; 
			if (bounds[g] > 0.0) {
				ip = std::sqrt(bounds[g]*bounds[g+1]);
			} else {
				ip = (bounds[g] + bounds[g+1])/2;
			}
			v(g) = opacity_func(rho, T, ip);
		}
	}

	else {
		if (weight_func) {
			for (int g=0; g<vdim; g++) {
				const auto dE = bounds[g+1] - bounds[g];
				v(g) = 0.0;
				double weight_sum = 0.0;
				for (int n=0; n<rule->GetNPoints(); n++) {
					const auto &ip = rule->IntPoint(n);
					const double E = bounds[g] + dE * ip.x;
					const double weight = weight_func(E,T);
					v(g) += ip.weight * opacity_func(rho, T, E) * weight;
					weight_sum += weight * ip.weight; 
				}
				v(g) /= weight_sum;
			}			
		}

		else {
			for (int g=0; g<vdim; g++) {
				const auto dE = bounds[g+1] - bounds[g];
				v(g) = 0.0;
				for (int n=0; n<rule->GetNPoints(); n++) {
					const auto &ip = rule->IntPoint(n);
					const double E = bounds[g] + dE * ip.x;
					v(g) += ip.weight * opacity_func(rho, T, E); 
				}
			}
		}
	}
}

BrunnerOpacityCoefficient::BrunnerOpacityCoefficient(
	const mfem::Array<double> &bounds, 
	double c0, double c1, double c2, double Emin, double Eedge, 
	double delta_s, double delta_w, int lines, bool planck_weight)
	: planck_weight(planck_weight), OpacityCoefficient(bounds.Size()-1)
{
	opac = new BrunnerOpac::AnalyticEdgeOpacity(Emin, Eedge, c0, c1, c2, delta_w, delta_s, lines);
	std::vector<double> bounds_std(bounds.Size());
	for (int i=0; i<bounds.Size(); i++) {
		bounds_std[i] = bounds[i];
	}
	integrator = new BrunnerOpac::MultiGroupIntegrator(*opac, bounds_std);

	planckAvg.resize(vdim); 
	rossAvg.resize(vdim);
	Bg.resize(vdim);
	Rg.resize(vdim);
}

void BrunnerOpacityCoefficient::Eval(
	mfem::Vector &v, mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) 
{
	const auto T = temperature->Eval(trans, ip);
	const auto rho = density->Eval(trans, ip);	
	double planckMean, rossMean;
	integrator->computeGroupAverages(T, rho, planckAvg, rossAvg, Bg, Rg, planckMean, rossMean);
	v.SetSize(vdim);
	for (int i=0; i<vdim; i++) {
		if (planck_weight) v(i) = planckAvg[i] * rho; 
		else v(i) = rossAvg[i] * rho;
	}
}
