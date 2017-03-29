//
// Created by kaushik on 30/3/17.
//

#include "MainLibrary/ObjectDetectCuda.h"

void objectDetect();

void basicFilter();

using namespace std;
using namespace cv;
using namespace cv::cuda;


int main() {
    //objectDetect();
    basicFilter();
}

void basicFilter() {

}

void objectDetect() {
    string cascadefilename = "/home/kaushik/Downloads/frontalFace10/haarcascade_frontalface_alt.xml";
    new ObjectDetectCuda(cascadefilename);
}
