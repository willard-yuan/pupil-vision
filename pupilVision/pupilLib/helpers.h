//
//  helpers.hpp
//  eTrackerV0.1
//
//  Created by willard on 11/11/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#ifndef HELPERS_H
#define HELPERS_H

bool inMat(cv::Point p,int rows,int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);

#endif /* helpers_hpp */
