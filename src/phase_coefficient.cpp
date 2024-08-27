#include "phase_coefficient.hpp"
#include "planck.hpp"

void PhaseSpaceCoefficient::SetAngle(const mfem::Vector &Omega_in)
{
	// copy Omega preserving Omega.Size() = 3 
	for (int d=0; d<Omega_in.Size(); d++) {
		Omega(d) = Omega_in(d); 
	}		
}

void PhaseSpaceCoefficient::SetEnergy(double low, double high, double mid)
{
	assert(high > low);
	assert(mid > low and mid < high);
	energy_low = low; 
	energy_high = high; 
	mean_energy = mid;
	energy_width = high - low;
}

PWPhaseSpaceCoefficient::PWPhaseSpaceCoefficient(const mfem::Array<int> &attrs, const mfem::Array<PhaseSpaceCoefficient*> &coefs) {
	assert(attrs.Size() == coefs.Size()); 
	for (auto i=0; i<attrs.Size(); i++) {
		map[attrs[i]] = coefs[i]; 
	}
}

void PWPhaseSpaceCoefficient::SetAngle(const mfem::Vector &Omega_in) {
	for (auto &it : map) {
		if (it.second) 
			it.second->SetAngle(Omega_in); 
	}
}

void PWPhaseSpaceCoefficient::SetEnergy(double low, double high, double mid) {
	for (auto &it : map) {
		if (it.second) 
			it.second->SetEnergy(low, high, mid); 
	}
}

double PWPhaseSpaceCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	const auto attr = trans.Attribute; 
	auto *ptr = map.at(attr); 
	if (ptr) return ptr->Eval(trans, ip); 
	else return 0.0; 
}

double FunctionGrayCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	mfem::Vector x(3); 
	mfem::Vector transip(x, 0, 3); 
	trans.Transform(ip, transip); 	
	return f(x, Omega); 
}

double FunctionPhaseSpaceCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	mfem::Vector x(3); 
	mfem::Vector transip(x, 0, 3); 
	trans.Transform(ip, transip); 	
	if (f) return f(x,Omega,mean_energy) * energy_width;
	else return ftd(x,Omega,mean_energy,GetTime()) * energy_width;  
}

double PlanckEmissionPSCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	const auto T = temperature.Eval(trans, ip);
	const auto b = IntegrateNormalizedPlanck(energy_high, T) - IntegrateNormalizedPlanck(energy_low, T);
	return b * constants::StefanBoltzmann * std::pow(T, 4) / 4 / constants::pi;
}