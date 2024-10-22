#include "utils.hpp"

namespace utils
{

unsigned int Floor(mfem::Vector &x, double min)
{
	unsigned int count = 0;
	for (auto &val : x) {
		if (val <= min) {
			val = min; 
			count++;
		}
	}
	return count;
}

}