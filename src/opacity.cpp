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

double AnalyticEdgeOpacityCoefficient::ComputeOpacity(
	const double T, const double rho, const double E)
{
	const double Ehat = std::max(E, Emin);
	const double coef = c0 * rho*rho / (std::sqrt(T) * std::pow(Ehat, 3)) * (1.0 - std::exp(-Ehat/T));
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

double AnalyticEdgeOpacityCoefficient::IntegrateOpacity(
	const double Elow, const double Ehigh, const double T, const double rho)
{
	if (int_order==1) {
		return ComputeOpacity(T, rho, std::sqrt(Elow*Ehigh));
	}
	
	const auto &rule = mfem::IntRules.Get(mfem::Geometry::SEGMENT, int_order);
	double val = 0.0;
	const auto dE = Ehigh - Elow;
	for (int n=0; n<rule.GetNPoints(); n++) {
		const auto &ip = rule.IntPoint(n);
		const double E = Elow + dE * ip.x;
		const auto opac = ComputeOpacity(T, rho, E);
		val += opac * ip.weight;
	}
	return val;
}
