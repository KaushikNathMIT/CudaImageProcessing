//
// Created by kaushik on 30/3/17.
//

#include "MatrixMultiplication.h"

__global__ void cannonsAlgorithm(double *matrixACuda, double *matrixBCuda, double *matrixCCuda) {

    int myId = threadIdx.x;
    int myRow = floor((double) myId / DIM);
    int myCol = myId % DIM;

    //Declaring shared memory block of required size
    __shared__ double sharedMemoryForA[DIM * DIM];
    __shared__ double sharedMemoryForB[DIM * DIM];

    /* ------------------------ INITAL SETUP ------------------------ */

    //Loading each processing element's data into corresponding location
    sharedMemoryForA[myId] = matrixACuda[myId];
    sharedMemoryForB[myId] = matrixBCuda[myId];

    //synchronizing threads so as to make sure all have loaded, marks end of loading phase
    __syncthreads();

    //Calculating where to take value from
    int sourceRankA = floor((double) ((myId + myRow) / DIM)) > myRow ? (myId + myRow - DIM) : (myId + myRow);
    int sourceRankB = (myId + myCol * DIM) >= DIM * DIM ? (myId + myCol * DIM - DIM * DIM) : (myId + myCol * DIM);

    //Collecting the data from shared memories
    matrixACuda[myId] = sharedMemoryForA[sourceRankA];
    matrixBCuda[myId] = sharedMemoryForB[sourceRankB];

    //synchronizing threads, marks end of intial setup
    __syncthreads();

    /* ------------------------ Algorithm ------------------------ */
    sourceRankA = floor((double) ((myId + 1) / DIM)) > floor((double) (myId / DIM)) ? (myId + 1 - DIM) : (myId + 1);
    sourceRankB = (myId + DIM) >= DIM * DIM ? (myId + DIM - DIM * DIM) : (myId + DIM);
    double resPij = 0.0;

    for (int i = 0; i < DIM; i++) {
        resPij += (matrixACuda[myId] * matrixBCuda[myId]);

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

MatMulCuda::MatMulCuda() {

    //Initializing matrices of DIM*DIM size
    double *matrixA = (double *) calloc(DIM * DIM, sizeof(double));
    double *matrixB = (double *) calloc(DIM * DIM, sizeof(double));
    double *matrixC = (double *) calloc(DIM * DIM, sizeof(double));

    //taking the matrices A and B from user
    cout << "\nEnter the matrix A[" << DIM << "][" << DIM << "]:";
    for (int i = 0; i < DIM * DIM; i++) cin >> matrixA[i];
    cout << "\nEnter the matrix B[" << DIM << "][" << DIM << "]:";
    for (int i = 0; i < DIM * DIM; i++) cin >> matrixB[i];

    //Allocating device memory for the input matrices
    double *matrixACuda, *matrixBCuda, *matrixCCuda;
    cudaMalloc((void **) &matrixACuda, DIM * DIM * sizeof(double));
    cudaMalloc((void **) &matrixBCuda, DIM * DIM * sizeof(double));
    cudaMalloc((void **) &matrixCCuda, DIM * DIM * sizeof(double));

    //Copying input matrices to device memory
    cudaMemcpy(matrixACuda, matrixA, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixBCuda, matrixB, DIM * DIM * sizeof(double), cudaMemcpyHostToDevice);

    //Calling the CUDA kernel for Cannon's Algorithm with NxN processing elements
    cannonsAlgorithm << < 1, DIM * DIM >> > (matrixACuda, matrixBCuda, matrixCCuda);


    //Copying the resultant matrix back to host memory
    cudaMemcpy(matrixC, matrixCCuda, DIM * DIM * sizeof(double), cudaMemcpyDeviceToHost);


    //Displaying final result
    cout << "\nA[" << DIM << "][" << DIM << "]*B[" << DIM << "][" << DIM << "] :";
    for (int i = 0; i < DIM * DIM; i++) {
        if (i % DIM == 0) cout << endl;
        cout << matrixC[i] << " ";
    }


    //Freeing allocated buffers
    cudaFree(matrixACuda);
    cudaFree(matrixBCuda);
    cudaFree(matrixCCuda);


}