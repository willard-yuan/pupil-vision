//
//  constants.h
//  eTrackerV0.1
//
//  Created by willard on 11/11/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#ifndef constants_h
#define constants_h

// Debugging
const bool kPlotVectorField = false;

// Preprocessing
const bool kSmoothFaceImage = false;  // 用于设定对人脸是否做高斯平滑
const float kSmoothFaceFactor = 0.005; //平滑系数

// 控制眼睛区域
/*const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;*/

const int kEyePercentTop = 35;
const int kEyePercentSide = 23;
const int kEyePercentHeight = 25;
const int kEyePercentWidth = 35;

// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

// Eye Corner
const bool kEnableEyeCorner = false;


#endif /* constants_h */
