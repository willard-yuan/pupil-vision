//
//  utils.hpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/16/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#ifndef utils_h
#define utils_h

#include <string>
#include <vector>
#include <set>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <random>
#include <stdlib.h>

void imshowscale(const std::string& name, cv::Mat& m, double scale);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);

namespace pupiltracker {
    
    class MakeString
    {
    public:
        std::stringstream stream;
        operator std::string() const { return stream.str(); }
        
        template<class T>
        MakeString& operator<<(T const& VAR) { stream << VAR; return *this; }
    };
    
    inline int pow2(int n)
    {
        return 1 << n;
    }
    
    template<typename T>
    inline T sq(T n)
    {
        return n * n;
    }
    
    template<typename T>
    inline T lerp(const T& val1, const T& val2, double alpha)
    {
        return val1*(1-alpha) + val2*alpha;
    }
    
    int random(int min, int max);
    int random(int min, int max, unsigned int seed);
    
    template<typename T>
    std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size)
    {
        if (size > src.size())
            throw std::range_error("Subset size out of range");
        
        std::vector<T> ret;
        std::set<size_t> vals;
        
        for (size_t j = src.size() - size; j < src.size(); ++j)
        {
            size_t idx = random(0, (int)j); // generate a random integer in range [0, j]
            
            if (vals.find(idx) != vals.end())
                idx = j;
            
            ret.push_back(src[idx]);
            vals.insert(idx);
        }
        
        return ret;
    }
    
    template<typename T>
    std::vector<T> randomSubset(const std::vector<T>& src, typename std::vector<T>::size_type size, unsigned int seed)
    {
        if (size > src.size())
            throw std::range_error("Subset size out of range");
        
        std::vector<T> ret;
        std::set<size_t> vals;
        
        for (size_t j = src.size() - size; j < src.size(); ++j)
        {
            size_t idx = random(0, j, seed+j); // generate a random integer in range [0, j]
            
            if (vals.find(idx) != vals.end())
                idx = j;
            
            ret.push_back(src[idx]);
            vals.insert(idx);
        }
        
        return ret;
    }
    
} //namespace pupiltracker


#endif /* utils_hpp */
