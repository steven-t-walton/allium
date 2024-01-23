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

double InflowPartialCurrentCoefficient::Eval(mfem::ElementTransformation &trans, const mfem::IntegrationPoint &ip) {
	assert(trans.ElementType == mfem::ElementTransformation::BDR_FACE); 
	auto *ftrans = dynamic_cast<mfem::FaceElementTransformations*>(&trans); 
	assert(ftrans); 
	ftrans->SetAllIntPoints(&ip); 
	const auto dim = trans.GetSpaceDim(); 
	nor.SetSize(dim); 
	if (dim==1) {
		nor(0) = 2*ftrans->GetElement1IntPoint().x - 1.0;
	} else {
		CalcOrtho(ftrans->Jacobian(), nor); 				
	}
	nor.Set(1./nor.Norml2(), nor); 

	double Jin = 0.0; 
	for (auto a=0; a<quad.Size(); a++) {
		const auto &Omega = quad.GetOmega(a); 
		phase_coef.SetState(Omega); 
		double psi_in = phase_coef.Eval(trans, ip); 
		double dot = Omega*nor; 
		if (dot <= 0) {
			Jin += dot * psi_in * quad.GetWeight(a); 
		}
	}
	return Jin * scale; 
}