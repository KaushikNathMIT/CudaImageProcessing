//
// Created by kaushik on 27/3/17.
//

#include "FaceDetectionCuda.h"
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argc, char *argv[]) {
    try {
        Mat src_host;
        VideoCapture cap(0);
        //cv::Mat src_host = cv::imread("/home/kaushik/CLionProjects/MPIImageArchitecture/res/a1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        while(1) {
            cap>>src_host;
            cuda::GpuMat dst, src;
            src.upload(src_host);

            //cuda::threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY);
            cuda::CannyEdgeDetector(src, dst, 0, 30, 3);
            cv::Mat result_host;
            dst.download(result_host);

            cv::imshow("Result", result_host);
            if(waitKey(10)==27) break;
        }
    }
    catch (const cv::Exception &ex) {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}