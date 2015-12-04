
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

int main(int argc, char* argv[])
{
    
    //读取视频
	cv::VideoCapture cap("F:\\ETrack\\1\\e1.avi");
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
        
        cout << "frame:  " << ++i << endl;
        
        pupiltracker::findPupilEllipse_out out;
        cv::Mat m;
        
        pupiltracker::TrackerParams params;
        //params.Radius_Min = 3;
        //params.Radius_Max = 8;
        params.Radius_Min = 30;
        params.Radius_Max = 70;
        //params.Radius_Min = 10;
        //params.Radius_Max = 20;
        //params.Radius_Min = 0;
        //params.Radius_Max = 50;
        
        params.CannyBlur = 1;
        params.CannyThreshold1 = 20;
        params.CannyThreshold2 = 40;
        params.StarburstPoints = 0;
        
        params.PercentageInliers = 30;
        params.InlierIterations = 2;
        params.ImageAwareSupport = true;
        params.EarlyTerminationPercentage = 95;
        params.EarlyRejection = true;
        params.Seed = -1;
        
        
        m = frame;
        pupiltracker::findPupilEllipse(params, m, out);
        
        // draw pupil center
        pupiltracker::cvx::cross(frame, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
        // draw the ellipse of pupil
        cv::ellipse(frame, out.elPupil, pupiltracker::cvx::rgb(255,0,255), 2);
        cv::imshow("Result", frame);
            
    
        //pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
        //cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255,0,255));
        //imshowscale("Haar Pupil", out.mHaarPupil, 3);
        //imshowscale("Pupil", out.mPupil, 3);
        //imshowscale("Thresh Pupil", out.mPupilThresh, 3);
        //imshowscale("Edges", out.mPupilEdges, 3);
        //cv::imshow("Result", m);
        if (cv::waitKey(10) != -1)
            break;
        //cout << "frame:  " << ++i << endl;
    }
    return 0;
}

