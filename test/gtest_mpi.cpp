#include "mfem.hpp"
#include "gtest/gtest.h"

int main(int argc, char *argv[]) {
	mfem::Mpi::Init(); 
	mfem::Hypre::Init(); 
	auto size = mfem::Mpi::WorldSize(); 
	auto rank = mfem::Mpi::WorldRank(); 
	::testing::InitGoogleTest(&argc, argv); 
	::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
	if (rank != 0) {
		delete listeners.Release(listeners.default_result_printer());
	}
	if (rank==0) 
		printf("Running main() from %s with %d ranks\n", __FILE__, size);
	return RUN_ALL_TESTS(); 
}