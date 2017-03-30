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
#define DIM 4

/*
Cannons Algorithm is as follows:
	row i of matrix a is circularly shifted by i elements to the left.
	col j of matrix b is circularly shifted by j elements up.
	Repeat n times:
		p[i][j] multiplies its two entries and adds to running total.
		circular shift each row of a 1 element left
		circular shift each col of b 1 element up

*/
__global__ void cannonsAlgorithm(double* matrixACuda, double* matrixBCuda, double* matrixCCuda) {

    int myId = threadIdx.x;
    int myRow = floor((double)myId/DIM);
    int myCol = myId%DIM;

    //Declaring shared memory block of required size
    __shared__ double sharedMemoryForA[DIM*DIM];
    __shared__ double sharedMemoryForB[DIM*DIM];

    /* ------------------------ INITAL SETUP ------------------------ */

    //Loading each processing element's data into corresponding location
    sharedMemoryForA[myId] = matrixACuda[myId];
    sharedMemoryForB[myId] = matrixBCuda[myId];

    //synchronizing threads so as to make sure all have loaded, marks end of loading phase
    __syncthreads();

    //Calculating where to take value from
    int sourceRankA = floor((double)((myId + myRow) / DIM))>myRow ? (myId + myRow - DIM) : (myId + myRow);
    int sourceRankB = (myId + myCol*DIM)>=DIM*DIM ? (myId + myCol*DIM - DIM*DIM) : (myId + myCol*DIM);

    //Collecting the data from shared memories
    matrixACuda[myId] = sharedMemoryForA[sourceRankA];
    matrixBCuda[myId] = sharedMemoryForB[sourceRankB];

    //synchronizing threads, marks end of intial setup
    __syncthreads();

    /* ------------------------ Algorithm ------------------------ */
    sourceRankA = floor((double)((myId + 1) / DIM))>floor((double)(myId / DIM)) ? (myId + 1 - DIM) : (myId + 1);
    sourceRankB = (myId + DIM)>=DIM*DIM ? (myId + DIM - DIM*DIM) : (myId + DIM);
    double resPij=0.0;

    for(int i=0; i<DIM; i++) {
        resPij+=(matrixACuda[myId]*matrixBCuda[myId]);

        //Store current values
        sharedMemoryForA[myId] = matrixACuda[myId];
        sharedMemoryForB[myId] = matrixBCuda[myId];
        __syncthreads();

        //Collecting the data from shared memories
        matrixACuda[myId] = sharedMemoryForA[sourceRankA];
        matrixBCuda[myId] = sharedMemoryForB[sourceRankB];
        __syncthreads();

    }

    matrixCCuda[myId] = resPij;

}

int main() {

    //Initializing matrices of DIM*DIM size
    double *matrixA = (double*)calloc(DIM*DIM,sizeof(double));
    double *matrixB = (double*)calloc(DIM*DIM,sizeof(double));
    double *matrixC = (double*)calloc(DIM*DIM,sizeof(double));

    //taking the matrices A and B from user
    cout<<"\nEnter the matrix A["<<DIM<<"]["<<DIM<<"]:";
    for(int i=0;i<DIM*DIM;i++) cin>>matrixA[i];
    cout<<"\nEnter the matrix B["<<DIM<<"]["<<DIM<<"]:";
    for(int i=0;i<DIM*DIM;i++) cin>>matrixB[i];

    //Allocating device memory for the input matrices
    double *matrixACuda, *matrixBCuda, *matrixCCuda;
    cudaMalloc((void**)&matrixACuda,DIM*DIM*sizeof(double));
    cudaMalloc((void**)&matrixBCuda,DIM*DIM*sizeof(double));
    cudaMalloc((void**)&matrixCCuda,DIM*DIM*sizeof(double));

    //Copying input matrices to device memory
    cudaMemcpy(matrixACuda, matrixA, DIM*DIM*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixBCuda, matrixB, DIM*DIM*sizeof(double), cudaMemcpyHostToDevice);

    //Calling the CUDA kernel for Cannon's Algorithm with NxN processing elements
    cannonsAlgorithm<<<1, DIM*DIM>>>(matrixACuda, matrixBCuda, matrixCCuda);


    //Copying the resultant matrix back to host memory
    cudaMemcpy(matrixC, matrixCCuda, DIM*DIM*sizeof(double), cudaMemcpyDeviceToHost);


    //Displaying final result
    cout<<"\nA["<<DIM<<"]["<<DIM<<"]*B["<<DIM<<"]["<<DIM<<"] :";
    for(int i=0;i<DIM*DIM;i++) {
        if(i%DIM==0) cout<<endl;
        cout<<matrixC[i]<< " ";
    }


    //Freeing allocated buffers
    cudaFree(matrixACuda);
    cudaFree(matrixBCuda);
    cudaFree(matrixCCuda);

    return 0;

}