//
//  pupilDetect.cpp
//  pupilTrack_Gravity
//
//  Created by willard on 4/24/16.
//  Copyright © 2016 wilard. All rights reserved.
//

#include "pupilDetect.hpp"
#include "constants.h"
#include "findEyeCenter.h"


cv::Point pupilDetect(cv::Mat &mEye){
    //梯度检测区域
    cv::Mat debugEye = mEye;
    cv::Rect rightEyeRegion(cv::Point2f(0, 0), cv::Point2f(debugEye.cols, debugEye.rows));
    float eyeImageWidth = debugEye.cols;
    if (kSmoothFaceImage) {
        double sigma = kSmoothFaceFactor * eyeImageWidth;
        GaussianBlur(debugEye, debugEye, cv::Size(0, 0), sigma);
    }
    cv::Point rightPupil = findEyeCenter(debugEye, rightEyeRegion, "Right Eye");
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    float w = 180.0;
    float h = 150.0;
    float ltStartX = rightPupil.x-w/2.;
    float ltStartY = rightPupil.y-h/2.;
    if(ltStartX < 0) ltStartX = 0.0;
    if(ltStartY < 0) ltStartY = 0.0;
    if(rightPupil.x+w/2. > debugEye.cols) w = debugEye.cols - rightPupil.x;
    if(rightPupil.y+h/2. > debugEye.rows) h = debugEye.rows - rightPupil.y;
    cv::Rect region_of_interest = cv::Rect(ltStartX, ltStartY, w, h);
    cv::Mat image_roi = mEye(region_of_interest);

    //处理
    cv::Mat threshold_frame;
    cv::inRange(image_roi, 0, 90, threshold_frame);

    cv::Mat locations;   // output, locations of non-zero pixels
    cv::findNonZero(threshold_frame, locations);
    long int sumX = 0;
    long int sumY = 0;
    for(int i = 0; i < locations.rows; i++){
        cv::Point  tt = locations.at<cv::Point>(i);
        sumX = sumX + tt.x;
        sumY = sumY + tt.y;
    }

    cv::Point testCenter;
    testCenter.x = (float)sumX/locations.rows + ltStartX;
    testCenter.y = (float)sumY/locations.rows + ltStartY;
    return testCenter;
}
