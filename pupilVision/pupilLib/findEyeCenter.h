//
//  findEyeCenter.hpp
//  eTrackerV0.1
//
//  Created by willard on 11/11/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#ifndef findEyeCenter_H
#define findEyeCenter_H

#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"

cv::Point unscalePoint(cv::Point p, cv::Rect origSize);
void scaleToFastSize(const cv::Mat &src,cv::Mat &dst);
cv::Mat computeMatXGradient(const cv::Mat &mat);
void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out);
cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow);
bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat);
cv::Mat floodKillEdges(cv::Mat &mat);


#endif /* findEyeCenter_hpp */
