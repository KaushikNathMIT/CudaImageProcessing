//
// Created by kaushik on 30/3/17.
//

#ifndef CUDAIMAGEARCHITECTURE_MATRIXMULTIPLICATION_H
#define CUDAIMAGEARCHITECTURE_MATRIXMULTIPLICATION_H
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*

	Created on 23/3/17

	Author: Ayush Soni and Rishabh Agarwal

	Title: Cannon's Algorithm CUDA (using NxN PEs)

	Description: Algorithm for matrix multiplication using processing elements in a Mesh topology. The project implements Compute Unified Device Architecture (CUDA) by NVIDIA, so as to perform operations in parallel.

*/


//importing CUDA libraries
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

//importing standard namespace
using namespace std;

//Defining dimensions as constant, for ease in shared memory declaration, as the dynamic shared memory in CUDA gets slightly complicated
#define DIM 3

/*
Cannons Algorithm is as follows:
	row i of matrix a is circularly shifted by i elements to the left.
	col j of matrix b is circularly shifted by j elements up.
	Repeat n times:
		p[i][j] multiplies its two entries and adds to running total.
		circular shift each row of a 1 element left
		circular shift each col of b 1 element up

*/

class MatMulCuda {
public:
    MatMulCuda();
};

#endif //CUDAIMAGEARCHITECTURE_MATRIXMULTIPLICATION_H
