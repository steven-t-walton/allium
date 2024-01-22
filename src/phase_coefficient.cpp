#include "phase_coefficient.hpp"

PWPhaseSpaceCoefficient::PWPhaseSpaceCoefficient(const mfem::Array<int> &attrs, const mfem::Array<PhaseSpaceCoefficient*> &coefs) {
	assert(attrs.Size() == coefs.Size()); 
	for (auto i=0; i<attrs.Size(); i++) {
		map[attrs[i]] = coefs[i]; 
	}
}

void PWPhaseSpaceCoefficient::SetState(const mfem::Vector &Omega_in, int g) {
	for (auto &it : map) {
		it.second->SetState(Omega_in, g); 
	}
}

double PWPhaseSpaceCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	const auto attr = trans.Attribute; 
	return map.at(attr)->Eval(trans, ip); 
}

double FunctionGrayCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	double x[3]; 
	mfem::Vector transip(x, 3); 
	trans.Transform(ip, transip); 	
	return f(transip, Omega); 
}