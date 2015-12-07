
//
//  main.cpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/15/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include "PupilTracker.h"
#include "cvx.h"


using namespace std;
using namespace cv;

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
        auto k = cvWaitKey(0);
        if(k == 's'){
            return;
        }
    }else if ( event == EVENT_LBUTTONUP ){
    }
}

int main(int argc, char* argv[])
{
    
    //读取视频
	cv::VideoCapture cap("/Users/willard/data/eyeVideo/pupillab.mp4");
    if(!cap.isOpened()) {
        std::cout << "Unable to open the camera\n";
        std::exit(-1);
    }
    
    cv::Mat frame;
    int i = 0;
    while(true) {
        cap >> frame;
        if(frame.empty()) {
            std::cout << "Can't read frames from your camera\n";
            break;
        }
        
        //cout << "frame:  " << ++i << endl;
        
        pupiltracker::findPupilEllipse_out out;
        cv::Mat m;
        
        double fScale = 1;
        
        pupiltracker::TrackerParams params;
        //params.Radius_Min = 3;
        //params.Radius_Max = 8;
        params.Radius_Min = (int)20*fScale;
        params.Radius_Max = (int)40*fScale;
        //params.Radius_Min = 10;
        //params.Radius_Max = 20;
        //params.Radius_Min = 0;
        //params.Radius_Max = 50;
        
        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.StarburstPoints = 0;
        
        params.PercentageInliers = 30;
        //params.InlierIterations = 2;
        params.InlierIterations = 1;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;
        
        m = frame;
        
        // 缩放图片
        //cv::imshow("Haar Pupil", m);
        //cv::Size2f dsize = Size(m.cols*fScale, m.rows*fScale);
        //cv::resize(m, m, dsize);
        //imshow("Haar Pupil1", m);
        //cvWaitKey();
        
        pupiltracker::findPupilEllipse(params, m, out);
        
        // 显示Haar响应得到的眼睛区域
        namedWindow("Haar Pupil",CV_WINDOW_NORMAL);
        cv::moveWindow("Haar Pupil", 1020, 10);
        imshowscale("Haar Pupil", out.mHaarPupil, 1);
        cv::resizeWindow("Haar Pupil", 300, 300);
        
        namedWindow("Pupil",CV_WINDOW_NORMAL);
        cv::moveWindow("Pupil", 1020, 450);
        imshowscale("Pupil", out.mPupil, 1);
        cv::resizeWindow("Pupil", 300, 300);
        
        // 显示经过阈值化后的瞳孔区域
        namedWindow("Thresh Pupil",CV_WINDOW_NORMAL);
        cv::moveWindow("Thresh Pupil", 50, 10);
        imshowscale("Thresh Pupil", out.mPupilThresh, 1);
        cv::resizeWindow("Thresh Pupil", 300, 300);
        
        //显示轮廓
        namedWindow("Edges",CV_WINDOW_NORMAL);
        cv::moveWindow("Edges", 50, 450);
        imshowscale("Edges", out.mPupilEdges, 1);
        cv::resizeWindow("Edges", 300, 300);
        
        
        // 画椭圆中心
        pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255, 0, 255));
        
        // 画椭圆
        cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255, 0, 0), 2);
        namedWindow("Pupil Tracking", CV_WINDOW_NORMAL);
        cv::moveWindow("Pupil Tracking", 360, 200);
        cv::imshow("Pupil Tracking", m);
        
        // 设置鼠标暂停
        setMouseCallback("Pupil Tracking", CallBackFunc, NULL);
        
        if (cv::waitKey(10) != -1)
            break;
    }
    return 0;
}

