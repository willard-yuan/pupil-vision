//
//  main.cpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/15/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include <iostream>

#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PupilTracker.h"
#include "cvx.h"

#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

#include <dlib/opencv.h>

using namespace std;
using namespace cv;
using namespace dlib;

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

void getEyeRegion(std::vector<dlib::rectangle> &dets, cv::Mat &frame, cv::Rect &rightEyeRegion){
    if(dets[0].left() < 0){
        rightEyeRegion = cv::Rect(0.0, int(dets[0].top()), int(dets[0].right()) - int(dets[0].left()), int(dets[0].bottom()) - int(dets[0].top()));
    }else if(dets[0].bottom() > frame.rows){
        rightEyeRegion = cv::Rect(int(dets[0].left()), int(dets[0].top()), int(dets[0].right()) - int(dets[0].left()), int(frame.rows) - int(dets[0].top()));
    }else{
        rightEyeRegion = cv::Rect(int(dets[0].left()), int(dets[0].top()), int(dets[0].right()) - int(dets[0].left()), int(dets[0].bottom()) - int(dets[0].top()));
    }
    return;
}

int main(int argc, char* argv[])
{ 
    //cv::Mat m = cv::imread(argv[1]);*/
    //cv::Mat frame = cv::imread("//Users/willard/data/eyeData/6_4.bmp");
    
    //读取视频
	CvCapture* capture;
	capture = cvCaptureFromCAM( 0 ); //切换摄像头，本机为0，外置的以1逐步增加
    cvWaitKey(300);
    
    //进行眼睛区域控制
    typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
    ifstream fin("E:\\PupilDetection\\eTrackerWinFitGaze2012\\model\\object_detector_eye.svm", ios::binary);
    if (!fin) {
        cout << "Can't find a trained object detector file object_detector.svm. " << endl;
        exit(EXIT_FAILURE);
    }
    object_detector<image_scanner_type> detector;
    deserialize(detector, fin);
    
	cv::Mat frame;
	cv::Mat mm;
    int i = 0;
    if( capture ) {
		while(true) {
			mm = cvQueryFrame( capture );
			//cvtColor(frame, frame, CV_GRAY2RGB);
			// mirror it

			cv::flip(mm, frame, 1);

			if(!frame.empty()) {			
				cout << "frame:  " << ++i << endl;

				cv_image<bgr_pixel> img(frame);
				std::vector<dlib::rectangle> dets = detector(img);
        
				pupiltracker::findPupilEllipse_out out;
				pupiltracker::tracker_log log;
				cv::Mat m;
        
				pupiltracker::TrackerParams params;
				//params.Radius_Min = 3;
				//params.Radius_Max = 8;
				//params.Radius_Min = 10;
				//params.Radius_Max = 25;
				params.Radius_Min = 10;
				params.Radius_Max = 20;
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
        
				if(dets.size() != 0 && dets[0].left() > 0){
            
					cv::Rect rightEyeRegion;
					getEyeRegion(dets, frame, rightEyeRegion);
					//cv::Rect rightEyeRegion(int32(dets[0].left()), int32(dets[0].top()), int32(dets[0].right()) - int32(dets[0].left()), int32(dets[0].bottom()) - int32(dets[0].top()));
        
					m = frame(rightEyeRegion);
            
					pupiltracker::findPupilEllipse(params, m, out, log);
            
					//瞳孔坐标变换到原图坐标下
					cv::Point2f det2f(dets[0].left(), dets[0].top());
					cv::Point2f pupilCenter;
					pupilCenter = out.pPupil;
					pupilCenter.x += det2f.x;
					pupilCenter.y += det2f.y;
					//拟合的椭圆变换到原图坐标下
					//cv::RotatedRect pupilEllipse;
					//RotatedRect pupilEllipse = RotatedRect(Point2f((int)out.elPupil.center.x,100), Size2f(100, 50), 30);
					RotatedRect pupilEllipse = RotatedRect(Point2f((int)out.elPupil.center.x, (int)out.elPupil.center.y),Size2f((int)out.elPupil.size.width, (int)out.elPupil.size.height), out.elPupil.angle);
					pupilEllipse.center.x += det2f.x;
					pupilEllipse.center.y += det2f.y;
            
					pupiltracker::cvx::cross(frame, pupilCenter, 5, pupiltracker::cvx::rgb(255,255,0));
					cv::ellipse(frame, pupilEllipse, pupiltracker::cvx::rgb(255,0,255), 2);
					cv::rectangle(frame, cvPoint(int(dets[0].left()), int(dets[0].top())), cvPoint(int(dets[0].right()), int(dets[0].bottom())), cv::Scalar(0,0,200), 1, 4);
            
    
					//pupiltracker::cvx::cross(m, out.pPupil, 5, pupiltracker::cvx::rgb(255,255,0));
					//cv::ellipse(m, out.elPupil, pupiltracker::cvx::rgb(255,0,255));
					//imshowscale("Haar Pupil", out.mHaarPupil, 3);
					//imshowscale("Pupil", out.mPupil, 3);
					//imshowscale("Thresh Pupil", out.mPupilThresh, 3);
					//imshowscale("Edges", out.mPupilEdges, 3);
					//cv::imshow("Result", m);
				}
				cv::imshow("Result", mm);
				if (cv::waitKey(10) != -1) break;
			}
		}
	}
	return 0;
	}