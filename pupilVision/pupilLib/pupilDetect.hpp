//
//  pupilDetect.hpp
//  pupilTrack_Gravity
//
//  Created by willard on 4/24/16.
//  Copyright © 2016 wilard. All rights reserved.
//

#ifndef pupilDetect_hpp
#define pupilDetect_hpp

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>

cv::Point pupilDetect(cv::Mat &mEye);

#endif /* pupilDetect_hpp */
