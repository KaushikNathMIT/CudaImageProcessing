//
// Created by kaushik on 30/3/17.
//

#ifndef CUDA2_OBJECTDETECTCUDA_H
#define CUDA2_OBJECTDETECTCUDA_H


#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/cudaobjdetect.hpp"

#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;
class ObjectDetectCuda {
public: ObjectDetectCuda(string cascadeFileName);
};



#endif //CUDA2_OBJECTDETECTCUDA_H
