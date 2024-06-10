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

TEST(MFEM, BlockVector) {
	mfem::Array<int> offsets(4);
	offsets[0] = 0; 
	offsets[1] = 2; 
	offsets[2] = 2; 
	offsets[3] = 4;
	offsets.PartialSum();

	mfem::Array<int> red_offsets(3);
	red_offsets[0] = 0;
	red_offsets[1] = 2;
	red_offsets[2] = 4;
	red_offsets.PartialSum();
	mfem::BlockVector x(offsets);
	x = 0.0;
	mfem::BlockVector y(x, offsets[1], red_offsets);
	y.GetBlock(0) = 1.0;
	y.GetBlock(1) = 2.0;
	
	mfem::Vector ex({0,0,1,1,2,2,2,2});
	x -= ex;
	EXPECT_NEAR(x.Norml2(), 0.0, 1e-12);	
}