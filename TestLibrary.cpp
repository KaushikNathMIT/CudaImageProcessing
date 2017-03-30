//
// Created by kaushik on 30/3/17.
//

#include "ImageProcessingLibrary/ObjectDetectCuda.h"
#include "ImageProcessingLibrary/BasicFilterCuda.h"
#include "DataProcessingLibrary/MatrixMultiplication.h"

void objectDetect();


using namespace std;
using namespace cv;
using namespace cv::cuda;


int main() {
    //objectDetect();
    BasicFilterCuda();
    /*int arr1[3][3], arr2[3][3];
    for(int i=0; i<3; i++) {
        for(int j=0; j<3; j++) {
            arr1[i][j] = i+1;
            arr2[i][j] = 9-(i+1);
        }
    }*/
    //new MatMulCuda();

}


void objectDetect() {
    string cascadefilename = "/home/kaushik/Downloads/frontalFace10/haarcascade_frontalface_alt.xml";
    new ObjectDetectCuda(cascadefilename);
}
