#include "videohandler.h"

#include <QDebug>

VideoHandler::VideoHandler()
{
}

void VideoHandler::start(int cameraIdx)
{
    //cap.open(cameraIdx);
    // 用于测试瞳孔检测
    string videoName = "/Users/willard/data/eyeVideo/1.avi";
    cv::VideoCapture cap1(videoName);
    this->cap = cap1;

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    if(cap.isOpened())
    {
        qDebug() << "Video information" <<
                ": width=" << cap.get(CV_CAP_PROP_FRAME_WIDTH) <<
                ", height=" << cap.get(CV_CAP_PROP_FRAME_HEIGHT) <<
                ", nframes=" << cap.get(CV_CAP_PROP_FRAME_COUNT) << endl;
    }
    else
    {
        qDebug() << "Could not initialize capturing...\n";
        return;
    }
}

void VideoHandler::stop()
{
    //cap.release();
}

const Mat& VideoHandler::getFrame()
{
    cap >> frame;
    return frame;
}
