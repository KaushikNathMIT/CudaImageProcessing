//
// Created by kaushik on 29/3/17.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

int main() {
    double totalT = 0.0;
    long frameCount = 0;
    Mat image;
    VideoCapture cap(0);
    while (1) {
        // Load Face cascade (.xml file)
        cap >> image;
        double t = (double) getTickCount();
        CascadeClassifier face_cascade;
        face_cascade.load("/home/kaushik/Downloads/frontalFace10/haarcascade_frontalface_alt.xml");

        // Detect faces
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(image, faces, 1.01, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        t = ((double) getTickCount() - t) / getTickFrequency();
        totalT += t;
        frameCount++;
        // Draw circles on the detected faces
        for (int i = 0; i < faces.size(); i++) {
            Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
            ellipse(image, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4,
                    8, 0);
        }

        imshow("Detected Face", image);

        if (waitKey(10) == 27) break;
    }
    cout << "fps: " << 1.0 / (totalT / (double) frameCount) << "\n";
    return 0;
}