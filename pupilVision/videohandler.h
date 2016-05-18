#ifndef VIDEOHANDLER_H
#define VIDEOHANDLER_H

#include <QtGui/QImage>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

using namespace cv;

class VideoHandler
{
public:
    VideoHandler();

    void start(int cameraIdx);
    void stop();

    const Mat &getFrame();

private:
    VideoCapture cap;
    Mat frame;
};

#endif // VIDEOHANDLER_H
