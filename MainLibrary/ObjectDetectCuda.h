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
#include <vector>
#include <opencv2/cudaimgproc.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

class ObjectDetectCuda {
public: ObjectDetectCuda(string cascadeFileName);
};

ObjectDetectCuda::ObjectDetectCuda(string cascadeName) {
    Ptr<cuda::CascadeClassifier> cascadeGPU = cuda::CascadeClassifier::create(cascadeName);
    VideoCapture capture(0);
    if(!capture.isOpened()) exit(0);

    int gpuCnt = getCudaEnabledDeviceCount();   // gpuCnt >0 if CUDA device detected
    if(gpuCnt==0) exit(0);  // no CUDA device found, quit


    Mat frame;
    long frmCnt = 0;
    double totalT = 0.0;

    while(true)
    {
        capture >> frame;   // grab current frame from the camera
        double t = (double)getTickCount();
        Mat frame_gray;
        cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);  // convert to gray image as face detection do NOT use color info
        GpuMat gray_gpu;  // copy the gray image to GPU memory
        GpuMat faces;
        gray_gpu.upload(frame);
        cuda::cvtColor(gray_gpu, gray_gpu,CV_BGR2GRAY);
        //cv::equalizeHist(frame_gray,frame_gray);
        cascadeGPU->setMinNeighbors(3);

        cascadeGPU->setScaleFactor(1.01);
        cascadeGPU->detectMultiScale(gray_gpu, faces);  // call face detection routine
        int detectNum = faces.cols;
        Mat obj_host;
        faces.colRange(0, detectNum).download(obj_host);  // retrieve results from GPU

        Rect* cfaces = obj_host.ptr<Rect>();  // results are now in "obj_host"
        t=((double)getTickCount()-t)/getTickFrequency();  // check how long did it take to detect face
        totalT += t;
        frmCnt++;

        for(int i=0;i<detectNum;++i)
        {
            Point pt1 = cfaces[i].tl();
            Size sz = cfaces[i].size();
            Point pt2(pt1.x+sz.width, pt1.y+sz.height);
            rectangle(frame, pt1, pt2, Scalar(255));
        }  // retrieve all detected faces and draw rectangles for visualization
        imshow("faces", frame);
        if(waitKey(10)==27) break;
    }

    cout << "fps: " << 1.0/(totalT/(double)frmCnt) <<"\n";
}

#endif //CUDA2_OBJECTDETECTCUDA_H
