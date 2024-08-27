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