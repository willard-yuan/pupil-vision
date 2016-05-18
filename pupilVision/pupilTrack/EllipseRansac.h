//
//  EllipseRansac.h
//  pupilTrack
//
//  Created by willard on 4/6/16.
//  Copyright © 2016 wilard. All rights reserved.
//

#ifndef EllipseRansac_h
#define EllipseRansac_h

// 定义结构体，保存拟合的结果
struct EllipseRansac_out {
    std::vector<cv::Point2f> bestInliers;
    cv::RotatedRect bestEllipse;
    double bestEllipseGoodness;
    int earlyRejections;
    bool earlyTermination;
    //列表初始化
    EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
};

struct EllipseRansac {
    const pupiltracker::TrackerParams& params;
    const std::vector<cv::Point2f>& edgePoints;
    int n;
    const cv::Rect& bb;
    const cv::Mat_<float>& mDX;
    const cv::Mat_<float>& mDY;
    int earlyRejections;
    bool earlyTermination;
    
    EllipseRansac_out out;
    
    // 构造函数并进行列表初始化
    EllipseRansac(const pupiltracker::TrackerParams& params, const std::vector<cv::Point2f>& edgePoints, int n, const cv::Rect& bb,const cv::Mat_<float>& mDX, const cv::Mat_<float>& mDY): params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0){}
    
    // 构造函数重载
    EllipseRansac(EllipseRansac& other)
    : params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections){}
    
    void operatorTmp(const int k);
    void join(EllipseRansac& other);
};

void EllipseRansac::join(EllipseRansac& other)
{
    //std::cout << "Ransac join" << std::endl;
    if (other.out.bestEllipseGoodness > out.bestEllipseGoodness)
    {
        std::swap(out.bestEllipseGoodness, other.out.bestEllipseGoodness);
        std::swap(out.bestInliers, other.out.bestInliers);
        std::swap(out.bestEllipse, other.out.bestEllipse);
    }
    out.earlyRejections += other.out.earlyRejections;
    earlyTermination |= other.earlyTermination;
    
    out.earlyTermination = earlyTermination;
}

void EllipseRansac::operatorTmp(const int k){
    if (out.earlyTermination) return;
    for(int i=0; i < k; ++i ){
        // Ransac Iteration
        // ----------------
        // sample拟合样本数据
        //std::vector<cv::Point2f> sample;
        
        /*if (params.Seed >= 0)
         
         sample = randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
         else
         sample = randomSubset(edgePoints, n);*/
        // 支持安卓
        std::vector<cv::Point2f> sample;
        if (params.Seed >= 0){
            if (n > edgePoints.size())
                throw std::range_error("Subset size out of range");
            std::set<size_t> vals;
            for (size_t j = edgePoints.size() - (i + params.Seed); j < edgePoints.size(); ++j){
                size_t idx = pupiltracker::random(0, j, params.Seed+j);
                
                if (vals.find(idx) != vals.end()) idx = j;
                
                sample.push_back((cv::Point_<float>)edgePoints[idx]);
                vals.insert(idx);
            }
        }else{
            if (n > edgePoints.size()) throw std::range_error("Subset size out of range");
            std::set<size_t> vals;
            for (size_t j = edgePoints.size() - n; j < edgePoints.size(); ++j){
                size_t idx = pupiltracker::random(0, (int)j);
                
                if (vals.find(idx) != vals.end()) idx = j;
                //sample.at(j) = edgePoints[idx];
                sample.push_back((cv::Point_<float>)edgePoints[idx]);
                vals.insert(idx);
            }
        }
        
        
        cv::RotatedRect ellipseSampleFit = cv::fitEllipse(sample);
        // Normalise ellipse to have width as the major axis.
        if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
        {
            ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
            std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
        }
        
        cv::Size2f s = ellipseSampleFit.size;
        // Discard useless ellipses early
        if (!ellipseSampleFit.center.inside(bb)
            || s.height > params.Radius_Max*2
            || s.width > params.Radius_Max*2
            || (s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2)
            || s.height > 4*s.width
            || s.width > 4*s.height)
        {
            // Bad ellipse! Go to your room!
            continue;
        }
        
        // Use conic section's algebraic distance as an error measure
        pupiltracker::ConicSection conicSampleFit(ellipseSampleFit);
        
        // Check if sample's gradients are correctly oriented
        if (params.EarlyRejection)
        {
            bool gradientCorrect = true;
            for(size_t j = 0; j < sample.size(); ++j)
            {
                cv::Point2f grad = conicSampleFit.algebraicGradientDir((cv::Point_<float>)sample[j]);
                float dx = mDX(cv::Point(((cv::Point_<float>)sample[j]).x, ((cv::Point_<float>)sample[j]).y));
                float dy = mDY(cv::Point(((cv::Point_<float>)sample[j]).x, ((cv::Point_<float>)sample[j]).y));
                
                float dotProd = dx*grad.x + dy*grad.y;
                
                gradientCorrect &= dotProd > 0;
            }
            if (!gradientCorrect)
            {
                out.earlyRejections++;
                continue;
            }
        }
        
        // Assume that the sample is the only inliers
        cv::RotatedRect ellipseInlierFit = ellipseSampleFit;
        pupiltracker::ConicSection conicInlierFit = conicSampleFit;
        std::vector<cv::Point2f> inliers, prevInliers;
        
        // Iteratively find inliers, and re-fit the ellipse
        for (int i = 0; i < params.InlierIterations; ++i)
        {
            // Get error scale for 1px out on the minor axis
            cv::Point2f minorAxis(-std::sin(pupiltracker::PI/180.0*ellipseInlierFit.angle), std::cos(pupiltracker::PI/180.0*ellipseInlierFit.angle));
            cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
            float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
            float errorScale = 1.0f/errOf1px;
            
            // Find inliers
            //inliers.reserve(edgePoints.size());//
            const float MAX_ERR = 2;
            for(size_t i = 0; i < edgePoints.size(); ++i)
            {
                float err = errorScale*conicInlierFit.distance((cv::Point_<float>)edgePoints[i]);
                
                if (err*err < MAX_ERR*MAX_ERR)
                    inliers.push_back((cv::Point_<float>)edgePoints[i]);
            }
            
            if (inliers.size() < n) {
                inliers.clear();
                continue;
            }
            
            // Refit ellipse to inliers
            ellipseInlierFit = fitEllipse(inliers);
            conicInlierFit = pupiltracker::ConicSection(ellipseInlierFit);
            
            // Normalise ellipse to have width as the major axis.
            if (ellipseInlierFit.size.height > ellipseInlierFit.size.width)
            {
                ellipseInlierFit.angle = std::fmod(ellipseInlierFit.angle + 90, 180);
                std::swap(ellipseInlierFit.size.height, ellipseInlierFit.size.width);
            }
        }
        
        if (inliers.empty())
            continue;
        
        // Discard useless ellipses again
        s = ellipseInlierFit.size;
        if (!ellipseInlierFit.center.inside(bb)
            || s.height > params.Radius_Max*2
            || s.width > params.Radius_Max*2
            || (s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2)
            || s.height > 4*s.width
            || s.width > 4*s.height
            )
        {
            // Bad ellipse! Go to your room!
            continue;
        }
        
        // Calculate ellipse goodness
        double ellipseGoodness = 0;
        if (params.ImageAwareSupport){
            for(size_t i = 0; i < inliers.size(); ++i){
                cv::Point2f grad = conicInlierFit.algebraicGradientDir((cv::Point_<float>)inliers[i]);
                //float dx = mDX(inliers[i]);
                //float dy = mDY(inliers[i]);
                float dx = grad.x;
                float dy = grad.y;
                
                double edgeStrength = dx*grad.x + dy*grad.y;
                
                ellipseGoodness += edgeStrength;
            }
        }else{
            ellipseGoodness = inliers.size();
        }
        
        if (ellipseGoodness > out.bestEllipseGoodness)
        {
            std::swap(out.bestEllipseGoodness, ellipseGoodness);
            std::swap(out.bestInliers, inliers);
            std::swap(out.bestEllipse, ellipseInlierFit);
            
            // Early termination, if 90% of points match
            if (params.EarlyTerminationPercentage > 0 && out.bestInliers.size() > params.EarlyTerminationPercentage*edgePoints.size()/100)
            {
                earlyTermination = true;
                break;
            }
        }
        
    }
}


#endif /* EllipseRansac_h */
