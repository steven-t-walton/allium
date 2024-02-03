#include "utils.hpp"
#include <cmath> 
#include <sstream>
#include <iomanip>

std::string FormatTimeString(double time) {
	std::stringstream ss; 
	if (time < 60) {
		ss << std::setprecision(3) << time; 
		return ss.str(); 
	}
	double remainder = std::fmod(time, 3600*24); 
	int days = time / (3600*24); 
	int hours = remainder / 3600; 
	remainder = std::fmod(time, 3600); 
	int minutes = remainder / 60; 
	auto seconds = std::fmod(remainder, 60); 
	if (days > 0) {
		ss << days << "-"; 
	}
	if (hours > 0 or days > 0) {
		ss << std::setfill('0') << std::setw(2) << hours << ":"; 
	} 
	if (minutes > 0 or hours > 0 or days > 0) {
		ss << std::setfill('0') << std::setw(2) << minutes << ":";
	}
	ss << std::setfill('0') << std::fixed << std::setprecision(2) << std::setw(5) << seconds; 		
	return ss.str(); 
}