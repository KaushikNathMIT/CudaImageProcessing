//
// Created by kaushik on 30/3/17.
//

#include "ImageProcessingLibrary/ObjectDetectCuda.h"
#include "ImageProcessingLibrary/BasicFilterCuda.h"
#include "DataProcessingLibrary/MatrixMultiplication.h"
#include "DataProcessingLibrary/MatrixSearch.h"

void objectDetect();


void matSearch();

void serialMatrixSearch(int value, int pInt[], int m, int n);

using namespace std;
using namespace cv;
using namespace cv::cuda;


int main() {
    int ch;
    //cin>>ch;
    //objectDetect();
    //BasicFilterCuda();
    matSearch();
    //MatMulCuda();

}

void matSearch() {
    int M, N, m, n, searchValue;
    cout<<"Enter search value";
    cin>>searchValue;
    M=1000,N=1000,m=10,n=10;
    int arr[M*N];
    for(int i=0; i<M*N; i++) {
        arr[i] = i+1;
    }

    new MatrixSearch(searchValue, arr ,M, N, 10, 10);
    serialMatrixSearch(searchValue, arr ,M, N);
}

void serialMatrixSearch(int value, int pInt[], int m, int n) {
    double time = getTickCount();
    for(int i=0;i<m*n;i++) {
        if(value==pInt[i]) printf("\nOccurence found at index %d, %d", i/n+1,i%n+1);
    }
    time = (getTickCount()-time)/getTickFrequency();
    cout<<"\nTime "<<time;
}


void objectDetect() {
    string cascadefilename = "/home/kaushik/Downloads/frontalFace10/haarcascade_frontalface_alt.xml";
    new ObjectDetectCuda(cascadefilename);
}
