#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include "mpi.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SIZ 20
#define num_inp 4

using namespace std;

#include <stdio.h>
#include <stdlib.h>


typedef struct edge {
    int first, second;
} edges;


__global__ void initialize_vertices(int *vertices, int starting_vertex) {
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v == starting_vertex) vertices[v] = 0; else vertices[v] = -1;
}

__global__ void bfs(const edge *edges, int *vertices, int current_depth) {
    int a = blockDim.x * blockIdx.x + threadIdx.x;
    int vfirst = edges[a].first;
    int dfirst = vertices[vfirst];
    int vsecond = edges[a].second;
    int dsecond = vertices[vsecond];
    if ((dfirst == current_depth) && (dsecond == -1)) {
        vertices[vsecond] = dfirst + 1;
    }
    if ((dfirst == -1) && (dsecond == current_depth)) {
        vertices[vfirst] = dsecond + 1;
    }
}


int main() {
    int vertices[100];                      //Scanf these values if u want to
    int no_of_vertices = 4, no_ofedges = 3;
    int *d_noofvertices;
    edges e[100];
    edges *d_edges;
    int *d_vertices;
    e[0].first = 0;
    e[0].second = 1;                           //edges 0,1 ; 1,2 ;
    e[1].first = 1;
    e[1].second = 2;
    e[2].first =2;
    e[2].second =3;
    int start = 0;
    int i;
    int *d_i, *d_start;
    cudaMalloc((void **) &d_i, sizeof(int));
    cudaMalloc((void **) &d_start, sizeof(int));
    cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start, &start, sizeof(int), cudaMemcpyHostToDevice);

    int max_depth = no_of_vertices;
    cudaMalloc((void **) &d_edges, sizeof(edges) * no_ofedges);
    cudaMalloc((void **) &d_vertices, sizeof(int) * no_of_vertices);
    cudaMemcpy(d_edges, e, sizeof(edges) * no_ofedges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices, vertices, sizeof(int) * no_of_vertices, cudaMemcpyHostToDevice);

    initialize_vertices << < no_of_vertices, 1 >> > (d_vertices, start);

    for (i = 0; i < max_depth; i++)                // i = current depth
    {
        bfs << <no_ofedges, 1 >> > (d_edges, d_vertices, i);//call kernel2
    }
    cudaMemcpy(vertices, d_vertices, sizeof(int) * no_of_vertices, cudaMemcpyDeviceToHost);
    printf("depth of each vertex from start = %d\n", start);
    for (i = 0; i < no_of_vertices; i++) {
        printf(" %d ", vertices[i]);
    }
    return 0;
}


/*

int input_size = 2;
int hidden_size = 20;
int output_size = 2;
int num_inputs = num_inp;
double W1[SIZ][SIZ], b1[SIZ] = {0}, W2[SIZ][SIZ], b2[SIZ];

std::default_random_engine generator(10);
std::normal_distribution<double> distribution(0.0, 1.0);

*/
/****************************************************
    THE FOLLOWING ARE THE UTILITY MPI PARALLEL FUNCTIONS
*****************************************************//*


void relu(double arr[][SIZ], int m1, int n1, double rel_mat[][SIZ], int rank) {
    double rel[SIZ];
    MPI_Scatter(arr, SIZ, MPI_DOUBLE, &rel, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank<m1) {
        for(int i = 0;i<n1;i++)
            if(rel[i]<0)
                rel[i] = 0;
    }
    MPI_Gather(rel, SIZ, MPI_DOUBLE, rel_mat, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void softmax(double arr[][SIZ], int m1, int n1, double softm[][SIZ], int rank) {
    int i;
    double soft[SIZ], sm[SIZ];

    MPI_Scatter(arr, SIZ, MPI_DOUBLE, soft, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double sum = 0;
    // printf("RANK : %d arr[0] : %lf n1: %d\n", rank,soft[0],n1);
    for (i = 0; i<n1; i++) {
        soft[i] = exp(soft[i]);
        sum += soft[i];
    }
    // printf("RANK : %d sum : %lf\n", rank,sum);
    MPI_Gather(&sum, 1, MPI_DOUBLE, sm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(sm, m1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (i = 0; i<n1; i++) {
        soft[i] = soft[i] / sm[rank];
    }
    MPI_Gather(soft, SIZ, MPI_DOUBLE, softm, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void mat_element_wise_multiply(double mat1[][SIZ], double mat2[][SIZ], int m, int n, int x, int y, double ans[][SIZ], int rank) {
    double rb1[SIZ];
    double rb2[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat2, SIZ, MPI_DOUBLE, rb2, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mul[SIZ];
    if (rank < m)
    {
        for (int i = 0; i < y; i++)
        {
            mul[i] = rb1[i] * rb2[i];
        }
    }
    MPI_Gather(mul, SIZ, MPI_DOUBLE, ans, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void printarr(double mul[][SIZ],int m,int n,char *name)
{
    printf("%s\n",name);
    for(int i = 0;i<m;i++)
    {
        for(int j = 0;j< n;j++)
            printf("%lf ",mul[i][j]);
        printf("\n");
    }
}

void mat_addition(double mat1[][SIZ], double mat2[SIZ], int m, int n, int x, double matadd[][SIZ], int rank) {
    // double ans[SIZ][SIZ];
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat2, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double sum[SIZ];
    if (rank < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum[i] = rb1[i] + mat2[i];
        }
    }
    MPI_Gather(sum, SIZ, MPI_DOUBLE, matadd, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void mat_multiply(double mat1[][SIZ], double mat2[][SIZ], int m, int n, int x, int y, double ans[][SIZ],int rank) {
    if (n != x)
    {
        printf("Error matrix size mismatch\n");
        return;
    }
    int i, j, k;
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double rb2[SIZ][SIZ];
    MPI_Bcast(mat2, SIZ * SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mat1, SIZ * SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double mul[SIZ];
    if (rank < m)
    {
        for (i = 0; i < y; i++)
        {
            mul[i] = 0;
            for (j = 0; j < n; j++)
            {
                mul[i] += (rb1[j] * mat2[j][i]);
            }
        }
    }
    // int ans[SIZ][SIZ];
    MPI_Gather(mul, SIZ, MPI_DOUBLE, ans, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

double mat_sum(double mat1[][SIZ], int m, int n,int rank) {
    double ans;
    double rb[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank<m)
    {
        for(int i = 1;i<n;i++)
        {
            rb[0] += rb[i];
        }
    }
    MPI_Reduce(&rb[0], &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0)
        return ans;
    else
        return 0;
}

double sum(double arr[], int n) {
    double sum = 0;
    int i;
    for (i = 0; i<n; i++) {
        sum += arr[i];
    }
    return sum;
}

void transpose(double arr1[][SIZ], int m, int n, double trans[][SIZ]) {
    int i, j;
    for (i = 0; i<n; i++) {
        for (j = 0; j<m; j++) {
            trans[i][j] = arr1[j][i];
        }
    }
}


void mat_col_wise_add(double arr1[][SIZ], int m, int n, double *sum) {

    int i, j;
    for (int i = 0; i<n; i++) {
        sum[i] = 0;
        for (j = 0; j<m; j++) {
            sum[i] += arr1[j][i];
        }
    }

}

void copy_matrix(double arr[][SIZ], int m, int n, double copy_mat[][SIZ], int rank) {
    double cpy[SIZ][SIZ];
    MPI_Scatter(arr, SIZ, MPI_DOUBLE, cpy, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(cpy, SIZ, MPI_DOUBLE, copy_mat, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


void argmax(double mat1[][SIZ], int m, int n, int* ans,int rank) {
    int i, j, k;
    double rb1[SIZ];
    MPI_Scatter(mat1, SIZ, MPI_DOUBLE, rb1, SIZ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int max_i;
    if (rank < m)
    {
        max_i = 0;
        for (i = 0; i < n; i++)
        {
            if (rb1[i] > rb1[max_i])
            {
                max_i = i;
            }
        }
    }
    MPI_Gather(&max_i, 1, MPI_INT, ans, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
*/
/***************************************************
    END OF MPI UTILITY PARALLEL FUNCTIONS
***************************************************//*


*/
/***************************************************
				CUDA KERNEL CODES
***************************************************//*

__global__ void logprobs_kernel(double * corect_logprobs, double * probs, int* y, int size)
{
    int i = blockIdx.x;
    corect_logprobs[i] = -log(probs[i*size + y[i]]);
}
__global__ void dscores_kernel_init(int * y, double * dscores, int size)
{
    int i = blockIdx.x;
    dscores[i*size + y[i]] -= 1;
}

__global__ void dscore_cal_kernel(double * dscores, int num_inputs, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    dscores[i*size + j] /= num_inputs;
}
__global__ void dhidden_cal_kernel(double * a1,double * dhidden,int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (a1[i*size + j] <= 0)
    {
        dhidden[i*size + j] = 0;
    }
}
__global__ void grads_w2_kernel(double * grads_W2,double * W2,double reg, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    grads_W2[i*size + j] += W2[i*size + j] * reg;
}
__global__ void grads_w1_kernel(double * grads_W1,double * W1,double reg, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    grads_W1[i*size + j] += W1[i*size + j] * reg;
}
__global__ void x_batch_kernel(double* X_batch, double * X, int * sample_indices, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    X_batch[i*size + j] = X[sample_indices[i] * size + j];
}
__global__ void y_batch_kernel(double* y_batch, double * y, int * sample_indices, int size)
{
    int i = blockIdx.x;
    y_batch[i] = y[sample_indices[i]];
}
__global__ void w1_kernel(double * grads_W1, double * W1, double learning_rate, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    W1[i*size + j] += (-learning_rate * grads_W1[i*size + j]);
}

__global__ void w2_kernel(double * grads_W2, double * W2, double learning_rate, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    W2[i*size + j] += (-learning_rate * grads_W2[i*size + j]);
}

*/
/**************************************************
	END OF ALL THE CUDA SPECIFIC FUNCTIONS
**************************************************//*


*/
/***************************************************
    START OF TWO LAYER NEURAL NETWORK FUNCTIONS
***************************************************//*


// Initializing toy data

void init_toy_data(double X[][SIZ], int *y)
{
    int i, j;
    for (i = 0; i<num_inputs; i++) {
        y[i] = 0;
        int t = i;
        for (j = input_size-1; j>=0; j--) {
            X[i][j] = t%2;
            t/=2;
            y[i] ^= (int)X[i][j];
        }
    }
}

// Loss function to calculate the loss

double loss(double reg, double grads_W1[][SIZ], double grads_W2[][SIZ], double *grads_b1, double *grads_b2, double X[][SIZ], int output_size, int *y, int hidden_size,int rank)
{
    if(rank == 0)printf("In loss \n");
    fflush(stdout);
    int i, j;
    fflush(stdout);
    double mul[SIZ][SIZ];
    mat_multiply(X, W1, num_inputs, input_size, input_size, hidden_size, mul, rank);
    double z1[SIZ][SIZ];
    mat_addition(mul, b1, num_inputs, hidden_size, hidden_size, z1,rank);
    double a1[SIZ][SIZ];
    relu(z1, num_inputs, hidden_size, a1, rank);
    double mul2[SIZ][SIZ];
    mat_multiply(a1, W2, num_inputs, hidden_size, hidden_size, output_size, mul2,rank);
    double score[SIZ][SIZ];
    mat_addition(mul2, b2, num_inputs, output_size, output_size, score,rank);
    double probs[SIZ][SIZ];
    softmax(score, num_inputs, output_size, probs, rank);
    // if(rank == 0)
    // {
    //     printarr(score,num_inputs, output_size,"score");
    //     printarr(probs,num_inputs, output_size,"probs");
    //     getchar();
    // }

    // DONE :: CUDA PARALLEL THE LOGPROBS

    cudaError_t cudaStatus;
    double *dev_a = 0;
    int *dev_b = 0;
    double *dev_c = 0;
    cudaStatus = cudaSetDevice(0);

    double corect_logprobs[num_inp];
    if (rank == 0)
    {
        // printf("corect_logprobs:\n");
        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_c, num_inputs * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_b, num_inputs * sizeof(int));

        cudaStatus = cudaMemcpy(dev_a, probs, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_b, y, num_inputs * sizeof(int), cudaMemcpyHostToDevice);

        logprobs_kernel<<<num_inputs, 1>>>(dev_c, dev_a, dev_b, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(corect_logprobs, dev_c, num_inputs * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }

    double data_loss;
    double elemul1[SIZ][SIZ];
    double elemul2[SIZ][SIZ];
    mat_element_wise_multiply(W1, W1, input_size, hidden_size, input_size, hidden_size, elemul1,rank);
    mat_element_wise_multiply(W2, W2, hidden_size, output_size, hidden_size, output_size, elemul2,rank);
    double reg_loss;
    double loss_f;
    if (rank == 0)
    {
        data_loss = sum(corect_logprobs, num_inputs) / num_inputs;
    }
    reg_loss = 0.5*reg*mat_sum(elemul1, input_size, hidden_size, rank) + 0.5*reg*mat_sum(elemul2, hidden_size, output_size, rank);
    if(rank == 0)
    {
        loss_f = data_loss + reg_loss;
        cout << "data_loss=" << data_loss << " reg_loss=" << reg_loss << " loss_f=" << loss_f << endl;
    }
    double dscores[SIZ][SIZ];
    copy_matrix(probs, num_inputs, output_size, dscores,rank);
    if (rank == 0)
    {
        // DONE :: CUDA PARALLEL DSCORES
        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_b, num_inputs * sizeof(int));

        cudaStatus = cudaMemcpy(dev_a, dscores, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_b, y, num_inputs * sizeof(int), cudaMemcpyHostToDevice);

        dscores_kernel_init<<<num_inputs, 1>>>(dev_b, dev_a, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(dscores, dev_a, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);

        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));

        cudaStatus = cudaMemcpy(dev_a, dscores, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

        dscore_cal_kernel<<<num_inputs, output_size>>>(dev_a, num_inputs, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(dscores, dev_a, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
    }
    double transa1[SIZ][SIZ];
    transpose(a1, num_inputs, hidden_size, transa1);
    mat_multiply(transa1, dscores, hidden_size, num_inputs, num_inputs, output_size, grads_W2,rank);
    mat_col_wise_add(dscores, num_inputs, output_size, grads_b2);
    double transW2[SIZ][SIZ];
    double dhidden[SIZ][SIZ];
    transpose(W2, hidden_size, output_size, transW2);
    mat_multiply(dscores, transW2, num_inputs, output_size, output_size, hidden_size, dhidden,rank);
    if (rank == 0)
    {
        // DONE :: PARALLELIZE THE DHIDDEN IN CUDA

        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));

        cudaStatus = cudaMemcpy(dev_a, a1, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_c, dhidden, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

        dhidden_cal_kernel<<<num_inputs, hidden_size>>>(dev_a, dev_c, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(dhidden, dev_c, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_c);
    }
    double transX[SIZ][SIZ];

    transpose(X, num_inputs, input_size, transX);
    mat_multiply(transX, dhidden, input_size, num_inputs, num_inputs, hidden_size, grads_W1,rank);
    mat_col_wise_add(dhidden, num_inputs, hidden_size, grads_b1);
    if (rank == 0)
    {
        // DONE :: PARALLELIZE GRADS_W1 AND GRADS_W2
        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));

        cudaStatus = cudaMemcpy(dev_a, grads_W2, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_c, W2, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

        grads_w2_kernel<<<hidden_size, output_size>>>(dev_a, dev_c, reg, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(grads_W2, dev_a, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_c);

        cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
        cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));

        cudaStatus = cudaMemcpy(dev_a, grads_W1, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
        cudaStatus = cudaMemcpy(dev_c, W1, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

        grads_w1_kernel<<<input_size, hidden_size>>>(dev_a, dev_c, reg, SIZ);
        cudaStatus = cudaGetLastError();

        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(grads_W1, dev_a, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_c);
    }
    if(rank == 0)
    {
        return loss_f;
    }
    return 0;
}

// The predict function will the predict the output from the given input

void predict(double X[][SIZ], int num_inputs, int input_size, int* y_pred,int rank)
{
    int i, j;
    double mul[SIZ][SIZ];
    mat_multiply(X, W1, num_inputs, input_size, input_size, hidden_size, mul,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double z1[SIZ][SIZ];
    mat_addition(mul, b1, num_inputs, hidden_size, hidden_size, z1,rank);
    double a1[SIZ][SIZ];
    relu(z1, num_inputs, hidden_size, a1,rank);
    double mul2[SIZ][SIZ];
    mat_multiply(a1, W2, num_inputs, hidden_size, hidden_size, output_size, mul2,rank);
    double score[SIZ][SIZ];
    mat_addition(mul2, b2, num_inputs, output_size, output_size, score,rank);
    argmax(score, num_inputs, output_size, y_pred,rank);
    if(rank == 0)
    {
        printarr(score,num_inputs,output_size,"score");
        for(i = 0;i < num_inputs;i++)
            printf("%d ",y_pred[i]);
        printf("\n");
    }
    // if(rank == 0)
    // {
    //     for(i = 0;i < num_inputs;i++)
    //         printf("%d ",y_pred[i]);
    //     printf("\n");
    //     printarr(score,num_inputs,output_size,"score");
    //     // printarr(W1,input_size, hidden_size,"W1");
    //     // printarr(mul,num_inputs, hidden_size,"mul");
    // }
}

//The train function will train the neural network on the data-set provided

void train(double X[][SIZ], int* y, double X_val[][SIZ], int* y_val, double learning_rate, double learning_rate_decay, double reg, int num_iters, int batch_size, int verbose, int num_train, int x_col, double grads_W1[][SIZ], double grads_W2[][SIZ], double *grads_b1, double* grads_b2,int rank)
{

    cudaError_t cudaStatus;
    double *dev_a = 0;
    int *dev_b = 0;
    double *dev_c = 0;
    cudaStatus = cudaSetDevice(0);

    if(rank == 0)
        printf("INSIDE TRAIN !!! Successfully!!!\n");
    srand(time(NULL));
    int iterations_per_epoch;
    if (rank == 0)
    {
        if (num_train / batch_size > 1)
        {
            iterations_per_epoch = num_train / batch_size;
        }
        else
        {
            iterations_per_epoch = 1;
        }
        printf("num_train : %d batch_size : %d iterations_per_epoch : %d\n",num_train,batch_size,iterations_per_epoch);
    }
    MPI_Bcast(&iterations_per_epoch,1,MPI_INT,0,MPI_COMM_WORLD);
    double X_batch[SIZ][SIZ];
    int sample_indices[SIZ], y_batch[SIZ];
    int t1;

    for (int it = 0; it<num_iters; it++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0)
            printf("rank : %d it : %d iterations_per_epoch : %d\n", rank, it,iterations_per_epoch);
        int pred[5];
        int *temp = (int*)calloc(sizeof(int),num_train);
        if (rank == 0)
        {
            printf("SAMPLE INDICES :: \n");
            for (int i = 0; i < batch_size; i++)
            {
                t1 = rand() % num_train;
                while (temp[t1] == 1)
                {
                    t1 = rand() % num_train;
                }
                temp[t1] = 1;
                sample_indices[i] = t1;
            }
            for(int i = 0;i<batch_size;i++)
                printf("%d ",sample_indices[i]);
            printf("\nbatch_size : %d num_train : %d\n",batch_size,num_train);
            fflush(stdout);
            // DONE :: X_batch parallelize with cuda
            cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_b, SIZ * sizeof(int));

            cudaStatus = cudaMemcpy(dev_a, X, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_b, sample_indices, SIZ * sizeof(int), cudaMemcpyHostToDevice);

            x_batch_kernel<<<batch_size, x_col>>>(dev_c, dev_a, dev_b, SIZ);
            cudaStatus = cudaGetLastError();

            cudaStatus = cudaDeviceSynchronize();

            cudaStatus = cudaMemcpy(X_batch, dev_c, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);

            printarr(X_batch,batch_size,x_col,"X_batch");

            // DONE :: Y_batch parallelize with cuda

            cudaStatus = cudaMalloc((void**)&dev_a, SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_c, SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_b, SIZ * sizeof(int));

            cudaStatus = cudaMemcpy(dev_a, y, SIZ * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_b, sample_indices, SIZ * sizeof(int), cudaMemcpyHostToDevice);

            y_batch_kernel<<<batch_size, 1>>>(dev_c, dev_a, dev_b, SIZ);
            cudaStatus = cudaGetLastError();

            cudaStatus = cudaDeviceSynchronize();

            cudaStatus = cudaMemcpy(y_batch, dev_c, SIZ * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_b);
            cudaFree(dev_c);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("rank : %dBefore loss\n",rank);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
        double loss_val = loss(reg, grads_W1, grads_W2, grads_b1, grads_b2, X, output_size, y, hidden_size, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        // printf("After loss\n");
        if (rank == 0)
        {

            // DONE :: W1,W2 PARALLELIZE WITH CUDA
            // W1:: W1,grads_W1,input_size,hidden_size,learning_rate :: W1[i][j] += (-learning_rate * grads_W1[i][j]);
            cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));

            cudaStatus = cudaMemcpy(dev_a, grads_W1, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_c, W1, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

            w1_kernel<<<input_size, hidden_size>>>(dev_a, dev_c, learning_rate, SIZ);
            cudaStatus = cudaGetLastError();

            cudaStatus = cudaDeviceSynchronize();

            cudaStatus = cudaMemcpy(W1, dev_c, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_c);

            // W2:: W2,grads_W2,hidden_size,output_size, learning_rate :: W2[i][j] += (-learning_rate * grads_W2[i][j]);
            cudaStatus = cudaMalloc((void**)&dev_a, SIZ * SIZ * sizeof(double));
            cudaStatus = cudaMalloc((void**)&dev_c, SIZ * SIZ * sizeof(double));

            cudaStatus = cudaMemcpy(dev_a, grads_W2, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);
            cudaStatus = cudaMemcpy(dev_c, W2, SIZ * SIZ * sizeof(double), cudaMemcpyHostToDevice);

            w2_kernel<<<hidden_size, output_size>>>(dev_a, dev_c, learning_rate, SIZ);
            cudaStatus = cudaGetLastError();

            cudaStatus = cudaDeviceSynchronize();

            cudaStatus = cudaMemcpy(W2, dev_c, SIZ * SIZ * sizeof(double), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_c);

            // printarr(W1,input_size,hidden_size,"W1");
            // printarr(W2,hidden_size,output_size,"W2");
            for (int i = 0; i < hidden_size; i++)
            {
                b1[i] += (-learning_rate * grads_b1[i]);
            }
            for (int i = 0; i < output_size; i++)
            {
                b2[i] += (-learning_rate * grads_b2[i]);
            }
            if (verbose == 1) {
                printf("\nIteration %d / %d: loss %f\n\n", it, num_iters, loss_val);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (it % iterations_per_epoch == 0)
        {
            // if(rank == 0)printf("Before predict\n");
            // return;
            predict(X_batch, batch_size, x_col, pred, rank);
            // MPI_Barrier(MPI_COMM_WORLD);
            // if(rank == 0)printf("Reached after predict\n");
            // return;
            int count = 0;
            if (rank == 0)
            {
                for (int i = 0; i < batch_size; i++)
                {
                    if (pred[i] == y_batch[i])
                    {
                        count++;
                    }
                }
                count = 0;
                double train_acc = count / batch_size;
            }

            predict(X_val, num_inputs, input_size, pred, rank);
            if (rank == 0)
            {
                for (int i = 0; i < num_inputs; i++)
                {
                    if (pred[i] == y_val[i])
                    {
                        count++;
                    }
                }
            }

            double val_acc = (double)count / (double)num_inputs;
            if (rank == 0)
            {
                printf("VALIDATION ACCURACY :: %f \n", val_acc);
            }
            learning_rate += learning_rate_decay;
        }
        // return;
    }
}




int main(int argc,char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                double temp = fabs(distribution(generator));
                while (temp > 1)
                    temp = temp - 1;
                W1[i][j] = temp;
            }
        }
        for (int i = 0; i < hidden_size; i++) {
            b1[i] = 0;
        }
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                double temp = fabs(distribution(generator));
                while (temp > 1)
                    temp = temp - 1;
                W2[i][j] = temp;
            }
        }
        for (int i = 0; i < output_size; i++) {
            b2[i] = 0;
        }
    }
    int pred[5];

    double grads_W1[SIZ][SIZ], grads_W2[SIZ][SIZ], grads_b1[SIZ], grads_b2[SIZ];
    double X[SIZ][SIZ];
    int y[SIZ];
    if(rank == 0)
    {
        init_toy_data(X,y);
        printarr(X,num_inputs,input_size,"X");
        for(int i = 0 ;i < num_inputs;i++)
            printf("%d ",y[i]);
        printf("\n");
    }
    double los = loss(1e-5, grads_W1, grads_W2, grads_b1, grads_b2, X, output_size, y, hidden_size, rank);

    // if (rank == 0)printf("LOSS :: %lf\n",los);
    fflush(stdout);
    double correct_loss = 1.30378789133;
    if (rank == 0) {
        printf("\n\n\n\nstarting TRAIN!!!\n\n\n");
    }
    train(X, y, X, y, 0.01, 0.1, 1e-5, 100, 1, 1, 1, input_size, grads_W1, grads_W2, grads_b1, grads_b2, rank);
    // int pred[5];
    predict(X, num_inputs, input_size, pred, rank);
    if (rank == 0)
    {
        printf("predictions::");
        for (int i = 0; i < num_inputs; i++) {
            printf("pred[%d]=%d\n", i, pred[i]);
        }
        printf("\n\n");
    }
    MPI_Finalize();
    return 0;
}
*/
