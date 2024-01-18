#include "gtest/gtest.h"
#include "mfem.hpp"

TEST(MFEM, DenseMatrixOrdering) {
	mfem::DenseMatrix A(2,2); 
	for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
			A(i,j) = 2*i + j; 
		}
	}
	const double *data = A.Data(); 
	for (int i=0; i<2; i++) {
		for (int j=0; j<2; j++) {
			EXPECT_DOUBLE_EQ(A(i,j), data[2*j+i]); 
		}
	}

	mfem::Vector div(4); 
	A.GradToDiv(div); 

	for (int i=0; i<4; i++) {
		EXPECT_DOUBLE_EQ(div(i), data[i]); 
	}
}