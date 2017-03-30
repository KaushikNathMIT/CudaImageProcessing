//
// Created by kaushik on 30/3/17.
//

#include "MatrixSearch.h"

__global__ void matrixSplit(int m, int n, int* a, int searchValue, int N) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k,l;
    int y[10];

    for(k=0;k<m;k++)
        for(l=0;l<n;l++)
            if(searchValue == a[(i*m +k)*N+ j*n +l]) printf("\nOccurence found at index %d, %d", (i*m +k)+1, j*n +l+1);
}


MatrixSearch::MatrixSearch(int searchValue, int* arr, int M, int N, int m, int n) {
    int *dx, *x;
    x = (int*)malloc(M*N*sizeof(int));
    for(int i=0; i<M*N; i++) x[i] = arr[i];
    cudaMalloc(&dx, M*N*sizeof(int));
    cudaMemcpy(dx, x, M*N*sizeof(int), cudaMemcpyHostToDevice);
    matrixSplit<<<M/m, N/n>>>(m, n, dx, searchValue, N);
    cudaFree(dx);
    free(x);
}