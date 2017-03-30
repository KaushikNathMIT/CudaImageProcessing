//
// Created by kaushik on 30/3/17.
//

#include "ImageProcessingLibrary/ObjectDetectCuda.h"
#include "ImageProcessingLibrary/BasicFilterCuda.h"
#include "DataProcessingLibrary/MatrixMultiplication.h"
#include "DataProcessingLibrary/MatrixSearch.h"

void objectDetect();


void matSearch();

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
    double time = getTickCount();
    //new MatrixSearch(searchValue, arr ,M, N, m, n);
    new MatrixSearch(searchValue, arr ,M, N, M, N);
    time = (getTickCount()-time)/getTickFrequency();
    cout<<"\nTime"<<time;
}


void objectDetect() {
    string cascadefilename = "/home/kaushik/Downloads/frontalFace10/haarcascade_frontalface_alt.xml";
    new ObjectDetectCuda(cascadefilename);
}
