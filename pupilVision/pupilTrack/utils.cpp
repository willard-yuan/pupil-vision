//
//  utils.cpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/16/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include "utils.h"

/*static std::mt19937 static_gen;
int pupiltracker::random(int min, int max)
{
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(static_gen);
}*/

/*int pupiltracker::random(int min, int max)
{
    init();
    //srand((unsigned)time(NULL));
    //static int seed = 0;
    //srand(seed++);
    
    double u = (((double) rand() / RAND_MAX) + 1)/2;
    return max+(max-min)*u;
}*/

/*int pupiltracker::random(int min, int max, unsigned int seed)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distribution(min, max);
    return distribution(gen);
}*/

int pupiltracker::random(int min, int max){
	return (int) rand() / (RAND_MAX + 1.0) * (max-min+1) + min;
}

int pupiltracker::random(int min, int max, unsigned int seed)
{
    srand(seed);
	return (int) rand() / (RAND_MAX + 1.0) * (max-min+1) + min;
}

void imshowscale(const std::string& name, cv::Mat& m, double scale)
{
    cv::Mat res;
    cv::resize(m, res, cv::Size(), scale, scale, cv::INTER_NEAREST);
    cv::imshow(name, res);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if ( event == cv::EVENT_LBUTTONDOWN )
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
        auto k = cvWaitKey(0);
        if(k == 's'){
            return;
        }
    }else if ( event == cv::EVENT_LBUTTONUP ){
    }
}