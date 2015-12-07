
//
//  main.cpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/15/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
	CvCapture* capture;
	capture = cvCaptureFromCAM( 0 ); //切换摄像头，本机为0，外置的以1逐步增加
    cvWaitKey(300);
    
    cv::Mat frame;
    int i = 0;
    if( capture ) {
		while(true) {
			frame = cvQueryFrame( capture );
			//cvtColor(frame, frame, CV_GRAY2RGB);
			// mirror it

			cv::flip(frame, frame, 1);

			if(!frame.empty()) {			
			cout << "frame:  " << ++i << endl;
			
			pupiltracker::findPupilEllipse_out out;
			cv::Mat m;
        
			pupiltracker::TrackerParams params;
			//params.Radius_Min = 3;
			//params.Radius_Max = 8;
			//params.Radius_Min = 10;
			//params.Radius_Max = 25;
			//params.Radius_Min = 10;
			//params.Radius_Max = 20;
			params.Radius_Min = 0;
			params.Radius_Max = 65;
        
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
			}else{
				std::cout << "--(!)No capture frame --Break!";
				break;
			}
			cv::imshow("Result", frame);
            
    
			//pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
			//cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255,0,255));
			//imshowscale("Haar Pupil", out.mHaarPupil, 3);
			//imshowscale("Pupil", out.mPupil, 3);
			//imshowscale("Thresh Pupil", out.mPupilThresh, 3);
			//imshowscale("Edges", out.mPupilEdges, 3);
			//cv::imshow("Result", m);
			int c = cv::waitKey(10);
			if( (char)c == 'c'){break;}
			if( (char)c == 'f'){
				imwrite("frame.png", frame);
			}
		}
	}
	return 0;
}

