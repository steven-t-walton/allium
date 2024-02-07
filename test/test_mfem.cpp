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

TEST(MFEM, ArrayResize) {
	mfem::Array<int> array; 
	array.Reserve(8); 
	EXPECT_EQ(array.Size(), 0); 
	EXPECT_EQ(array.Capacity(), 8); 

	array.SetSize(4); 
	EXPECT_EQ(array.Size(), 4); 
	EXPECT_EQ(array.Capacity(), 8); 
}

TEST(MFEM, PartialSum) {
	mfem::Array<int> I(4); 
	I[0] = 0; 
	I[1] = 2; 
	I[2] = 4; 
	I[3] = 3; 
	I.PartialSum(); 
	mfem::Array<int> ex({0,2,6,9}); 
	for (auto i=0; i<4; i++) {
		EXPECT_EQ(ex[i], I[i]); 
	}
}