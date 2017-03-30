#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

using namespace std;

__global__ void matrixSplit(int m, int n, int* a, int M, int N) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k,l;
    int y[10];

    for(k=0;k<m;k++)
        for(l=0;l<n;l++)
            y[k*m + l] = a[(i*m +k)*N+ j*n +l];
    for(k=0;k<m;k++)
        for(l=0;l<n;l++) {
            printf("\nThread %d Block %d: %d", i, j, y[k*m + l]);
        }
}


int main()
{
    int *x,*dx, M=3, N=3;
    x = (int*)malloc(M*N*sizeof(int));
    for(int i=0; i<M*N;i++) cin>>x[i];
    //y = (int*)malloc(N*sizeof(int));
    //z = (int*)malloc(M*N*sizeof(int));

    cudaMalloc(&dx, M*N*sizeof(int));
    //cudaMalloc(&dy, N*sizeof(int));

    cudaMemcpy(dx, x, M*N*sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(dy, y, N*sizeof(int), cudaMemcpyHostToDevice);

    int m=1, n=1;
    matrixSplit<<<M/m, N/n>>>(m,n,dx, M, N);

    cudaFree(dx);
    free(x);

    getchar();

    return 0;
}
