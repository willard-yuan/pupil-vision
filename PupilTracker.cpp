//
//  PupilTracker.cpp
//  eTrackerMacFitGaze
//
//  Created by willard on 11/16/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include "PupilTracker.h"

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cvx.h"

class HaarSurroundFeature
{
public:
    HaarSurroundFeature(int r1, int r2) : r_inner(r1), r_outer(r2)
    {
        //  _________________
        // |        -ve      |
        // |     _______     |
        // |    |   +ve |    |
        // |    |   .   |    |
        // |    |_______|    |
        // |         <r1>    |
        // |_________<--r2-->|
        
        // Number of pixels in each part of the kernel
        int count_inner = r_inner*r_inner;
        int count_outer = r_outer*r_outer - r_inner*r_inner;
        
        // Frobenius normalized values
        //
        // Want norm = 1 where norm = sqrt(sum(pixelvals^2)), so:
        //  sqrt(count_inner*val_inner^2 + count_outer*val_outer^2) = 1
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Solving both of these gives:
        //val_inner = std::sqrt( (double)count_outer/(count_inner*count_outer + sq(count_inner)) );
        //val_outer = -std::sqrt( (double)count_inner/(count_inner*count_outer + sq(count_outer)) );
        
        // Square radius normalised values
        //
        // Want the response to be scale-invariant, so scale it by the number of pixels inside it:
        //  val_inner = 1/count = 1/r_outer^2
        //
        // Also want sum(pixelvals) = 0, so:
        //  count_inner*val_inner + count_outer*val_outer = 0
        //
        // Hence:
        val_inner = 1.0 / (r_inner*r_inner);
        val_outer = -val_inner*count_inner/count_outer;
        
    }
    
    double val_inner, val_outer;
    int r_inner, r_outer;
};

//bool pupiltracker::findPupilEllipse(const pupiltracker::TrackerParams& params, const cv::Mat& m, pupiltracker::findPupilEllipse_out& out, pupiltracker::tracker_log& log){
bool pupiltracker::findPupilEllipse(const pupiltracker::TrackerParams& params, const cv::Mat& m, pupiltracker::findPupilEllipse_out& out){
    
    // --------------------
    //  转成灰度图像
    // --------------------
    
    cv::Mat_<uchar> mEye;
    
    // Pick one channel if necessary, and crop it to get rid of borders
    if (m.channels() == 1){
        mEye = m;
    }else if (m.channels() == 3){
        cv::cvtColor(m, mEye, cv::COLOR_BGR2GRAY);
    }else if (m.channels() == 4){
        cv::cvtColor(m, mEye, cv::COLOR_BGRA2GRAY);
    }else{
        throw std::runtime_error("Unsupported number of channels");
    }
    cv::cvtColor(m, mEye, cv::COLOR_BGR2GRAY);
    
    // -----------------------
    // 寻找最强的haar响应
    // -----------------------
    
    //             _____________________
    //            |         Haar kernel |
    //            |                     |
    //  __________|______________       |
    // | Image    |      |       |      |
    // |    ______|______|___.-r-|--2r--|
    // |   |      |      |___|___|      |
    // |   |      |          |   |      |
    // |   |      |          |   |      |
    // |   |      |__________|___|______|
    // |   |    Search       |   |
    // |   |    region       |   |
    // |   |                 |   |
    // |   |_________________|   |
    // |                         |
    // |_________________________|
    //
    
	//cv::Mat_<int32_t> mEyeIntegral; // 积分图像
    cv::Mat_<int> mEyeIntegral; // 积分图像
    int padding = 2*params.Radius_Max;
    
    //计算积分图像
    cv::Mat mEyePad;
    // Need to pad by an additional 1 to get bottom & right edges.
    cv::copyMakeBorder(mEye, mEyePad, padding, padding, padding, padding, cv::BORDER_REPLICATE);
    cv::integral(mEyePad, mEyeIntegral);
    
    
    cv::Point2f pHaarPupil;
    int haarRadius = 0;
    
    //计算haar响应
    const int rstep = 2;
    const int ystep = 4;
    const int xstep = 4;
    
    double minResponse = std::numeric_limits<double>::infinity();

	
	//修改代码开始
	std::pair<double,cv::Point2f> minValOut;
    for (int r = params.Radius_Min; r < params.Radius_Max; r+=rstep){
        // Get Haar feature
        int r_inner = r;
        int r_outer = 3*r;
        HaarSurroundFeature f(r_inner, r_outer);

        for (int i = 0;  i < ((mEye.rows-r - r - 1)/ystep + 1); ){
		    int y = r + i*ystep;
            int* row1_inner = mEyeIntegral[y+padding - r_inner];
            int* row2_inner = mEyeIntegral[y+padding + r_inner + 1];
            int* row1_outer = mEyeIntegral[y+padding - r_outer];
            int* row2_outer = mEyeIntegral[y+padding + r_outer + 1];
                
            int* p00_inner = row1_inner + r + padding - r_inner;
            int* p01_inner = row1_inner + r + padding + r_inner + 1;
            int* p10_inner = row2_inner + r + padding - r_inner;
            int* p11_inner = row2_inner + r + padding + r_inner + 1;
                
            int* p00_outer = row1_outer + r + padding - r_outer;
            int* p01_outer = row1_outer + r + padding + r_outer + 1;
            int* p10_outer = row2_outer + r + padding - r_outer;
            int* p11_outer = row2_outer + r + padding + r_outer + 1;
            
            for (int x = r; x < mEye.cols - r; x+=xstep){
                int sumInner = *p00_inner + *p11_inner - *p01_inner - *p10_inner;
                int sumOuter = *p00_outer + *p11_outer - *p01_outer - *p10_outer - sumInner;
                    
                double response = f.val_inner * sumInner + f.val_outer * sumOuter;
                    
                if (response < minResponse){
                    minValOut.first = response;
                    minValOut.second = cv::Point(x,y);

					minResponse = response;
					minResponse = minValOut.first;
                    // Set return values
                    pHaarPupil = minValOut.second;
                    haarRadius = r;
                }
                    
                p00_inner += xstep;
                p01_inner += xstep;
                p10_inner += xstep;
                p11_inner += xstep;
                    
                p00_outer += xstep;
                p01_outer += xstep;
                p10_outer += xstep;
                p11_outer += xstep;
            }
			y += ystep;
			i += 1;
        }
	}//修改代码结束

    // Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
    haarRadius = (int)(haarRadius * SQRT_2);
    
    // ---------------------------
    // Pupil ROI around Haar point
    // ---------------------------
    cv::Rect roiHaarPupil = cvx::roiAround(cv::Point(pHaarPupil.x, pHaarPupil.y), haarRadius);
    cv::Mat_<uchar> mHaarPupil;
    cvx::getROI(mEye, mHaarPupil, roiHaarPupil);
    
    out.roiHaarPupil = roiHaarPupil;
    out.mHaarPupil = mHaarPupil;
    
    // --------------------------------------------------
    // Get histogram of pupil region, segment with KMeans
    // --------------------------------------------------
    
    const int bins = 256;
    
    cv::Mat_<float> hist;
    //SECTION("Histogram", log)
    {
        int channels[] = {0};
        int sizes[] = {bins};
        float range[2] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&mHaarPupil, 1, channels, cv::Mat(), hist, 1, sizes, ranges);
    }
    
    out.histPupil = hist;
    
    float threshold;
    //SECTION("KMeans", log)
    {
        // Try various candidate centres, return the one with minimal label distance
        float candidate0[2] = {0, 0};
        float candidate1[2] = {128, 255};
        float bestDist = std::numeric_limits<float>::infinity();
        float bestThreshold = std::numeric_limits<float>::quiet_NaN();
        
        for (int i = 0; i < 2; i++)
        {
            cv::Mat_<uchar> labels;
            float centres[2] = {candidate0[i], candidate1[i]};
            float dist = cvx::histKmeans(hist, 0, 256, 2, centres, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0.0));
            
            float thisthreshold = (centres[0] + centres[1])/2;
			if (dist < bestDist && !(thisthreshold != thisthreshold))
            {
                bestDist = dist;
                bestThreshold = thisthreshold;
            }
        }
        if ((bestThreshold != bestThreshold))
        {
            // If kmeans gives a degenerate solution, exit early
            return false;
        }
        
        threshold = bestThreshold;
    }
    
    cv::Mat_<uchar> mPupilThresh;
    //SECTION("Threshold", log)
    {
        cv::threshold(mHaarPupil, mPupilThresh, threshold, 255, cv::THRESH_BINARY_INV);
    }
    
    out.threshold = threshold;
    out.mPupilThresh = mPupilThresh;
    
    // ---------------------------------------------
    // Find best region in the segmented pupil image
    // ---------------------------------------------
    
    cv::Rect bbPupilThresh;
    cv::RotatedRect elPupilThresh;
    
    //SECTION("Find best region", log)
    {
		cv::Mat_<uchar> mPupilContours = mPupilThresh.clone();
		std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mPupilContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        
        if (contours.size() == 0)
            return false;
        
        std::vector<cv::Point>& maxContour = contours[0];
        double maxContourArea = cv::contourArea(maxContour);
        for(size_t i = 0; i < contours.size(); ++i){
			//double area = cv::contourArea(contours.at(i));
			double area = cv::contourArea((cv::InputArray&)contours[i]); //支持安卓
            if (area > maxContourArea)
            {
                maxContourArea = area;
				//maxContour = contours.at(i);
                maxContour = contours[i]; //支持安卓
            }
        }
        
        cv::Moments momentsPupilThresh = cv::moments(maxContour);
        
        bbPupilThresh = cv::boundingRect(maxContour);
        elPupilThresh = cvx::fitEllipse(momentsPupilThresh);
        
        // Shift best region into eye coords (instead of pupil region coords), and get ROI
        bbPupilThresh.x += roiHaarPupil.x;
        bbPupilThresh.y += roiHaarPupil.y;
        elPupilThresh.center.x += roiHaarPupil.x;
        elPupilThresh.center.y += roiHaarPupil.y;
    }
    
    out.bbPupilThresh = bbPupilThresh;
    out.elPupilThresh = elPupilThresh;
    
    // ------------------------------
    // Find edges in new pupil region
    // ------------------------------
    
    cv::Mat_<uchar> mPupil, mPupilOpened, mPupilBlurred, mPupilEdges;
    cv::Mat_<float> mPupilSobelX, mPupilSobelY;
    cv::Rect bbPupil;
    cv::Rect roiPupil = cvx::roiAround(cv::Point(elPupilThresh.center.x, elPupilThresh.center.y), haarRadius);
    //SECTION("Pupil preprocessing", log)
    {
        const int padding = 3;
        
        cv::Rect roiPadded(roiPupil.x-padding, roiPupil.y-padding, roiPupil.width+2*padding, roiPupil.height+2*padding);
        // First get an ROI around the approximate pupil location
        cvx::getROI(mEye, mPupil, roiPadded, cv::BORDER_REPLICATE);
        
        cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mPupil, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2);
        
        if (params.CannyBlur > 0)
        {
            cv::GaussianBlur(mPupilOpened, mPupilBlurred, cv::Size(), params.CannyBlur);
        }
        else
        {
            mPupilBlurred = mPupilOpened;
        }
        
        cv::Sobel(mPupilBlurred, mPupilSobelX, CV_32F, 1, 0, 3);
        cv::Sobel(mPupilBlurred, mPupilSobelY, CV_32F, 0, 1, 3);
        
        cv::Canny(mPupilBlurred, mPupilEdges, params.CannyThreshold1, params.CannyThreshold2);
        
        cv::Rect roiUnpadded(padding,padding,roiPupil.width,roiPupil.height);
        mPupil = cv::Mat(mPupil, roiUnpadded);
        mPupilOpened = cv::Mat(mPupilOpened, roiUnpadded);
        mPupilBlurred = cv::Mat(mPupilBlurred, roiUnpadded);
        mPupilSobelX = cv::Mat(mPupilSobelX, roiUnpadded);
        mPupilSobelY = cv::Mat(mPupilSobelY, roiUnpadded);
        mPupilEdges = cv::Mat(mPupilEdges, roiUnpadded);
        
        bbPupil = cvx::boundingBox(mPupil);
    }
    
    out.roiPupil = roiPupil;
    out.mPupil = mPupil;
    out.mPupilOpened = mPupilOpened;
    out.mPupilBlurred = mPupilBlurred;
    out.mPupilSobelX = mPupilSobelX;
    out.mPupilSobelY = mPupilSobelY;
    out.mPupilEdges = mPupilEdges;
    
    // -----------------------------------------------
    // Get points on edges, optionally using starburst
    // -----------------------------------------------
    
    std::vector<cv::Point2f> edgePoints;
	//找到非零像素点的各个坐标
    //SECTION("Non-zero value finder", log)
    {
        for(int y = 0; y < mPupilEdges.rows; y++)
        {
            uchar* val = mPupilEdges[y];
            for(int x = 0; x < mPupilEdges.cols; x++, val++)
            {
                if(*val == 0)
                    continue;
                edgePoints.push_back(cv::Point2f(x + 0.5f, y + 0.5f));
            }
        }
    }
    
    
    // ---------------------------
    // Fit an ellipse to the edges
    // ---------------------------
    
    cv::RotatedRect elPupil;
    std::vector<cv::Point2f> inliers;
    //SECTION("Ellipse fitting", log)
    {
        // Desired probability that only inliers are selected
        const double p = 0.999;
        // Probability that a point is an inlier
        double w = params.PercentageInliers/100.0;
        // Number of points needed for a model
        const int n = 5;
        
        if (params.PercentageInliers == 0)
            return false;
        
		// 如果找到的轮廓点个数大于5个，则进行椭圆拟合
        if (edgePoints.size() >= n) // Minimum points for ellipse
        {
            // RANSAC!!!
            
            double wToN = std::pow(w,n);
            int k = static_cast<int>(std::log(1-p)/std::log(1 - wToN)  + 2*std::sqrt(1 - wToN)/wToN);
            
            out.ransacIterations = k;
            
            //size_t threshold_inlierCount = std::max<size_t>(n, static_cast<size_t>(out.edgePoints.size() * 0.7));
            
			// 保存拟合的结果
            struct EllipseRansac_out {
                std::vector<cv::Point2f> bestInliers;
                cv::RotatedRect bestEllipse;
                double bestEllipseGoodness;
                int earlyRejections;
                bool earlyTermination;
                //列表初始化
                EllipseRansac_out() : bestEllipseGoodness(-std::numeric_limits<double>::infinity()), earlyTermination(false), earlyRejections(0) {}
            };
			// 
            struct EllipseRansac {
                const TrackerParams& params;
                const std::vector<cv::Point2f>& edgePoints;
                int n;
                const cv::Rect& bb;
                const cv::Mat_<float>& mDX;
                const cv::Mat_<float>& mDY;
                int earlyRejections;
                bool earlyTermination;
                
                EllipseRansac_out out;
                
                EllipseRansac(
                              const TrackerParams& params,
                              const std::vector<cv::Point2f>& edgePoints,
                              int n,
                              const cv::Rect& bb,
                              const cv::Mat_<float>& mDX,
                              const cv::Mat_<float>& mDY)
                : params(params), edgePoints(edgePoints), n(n), bb(bb), mDX(mDX), mDY(mDY), earlyTermination(false), earlyRejections(0)
                {
                }
                
				// 去掉了TBB
                EllipseRansac(EllipseRansac& other)
                : params(other.params), edgePoints(other.edgePoints), n(other.n), bb(other.bb), mDX(other.mDX), mDY(other.mDY), earlyTermination(other.earlyTermination), earlyRejections(other.earlyRejections)
                {
                    //std::cout << "Ransac split" << std::endl;
                }
                
				// 去掉了TBB
				void operatorTmp(const int k)
                {
                    if (out.earlyTermination)
                        return;
                    //std::cout << "Ransac start (" << (r.end()-r.begin()) << " elements)" << std::endl;
					for(int i=0; i < k; ++i )
                    {
                        // Ransac Iteration
                        // ----------------
						// sample拟合样本数据
                        /*std::vector<cv::Point2f> sample;
                        if (params.Seed >= 0)
                            sample = randomSubset(edgePoints, n, static_cast<unsigned int>(i + params.Seed));
                        else
                            sample = randomSubset(edgePoints, n);*/

						// 支持安卓
                        std::vector<cv::Point2f> sample;
                        if (params.Seed >= 0){
							if (n > edgePoints.size())
								throw std::range_error("Subset size out of range");
							std::set<size_t> vals;
							 for (size_t j = edgePoints.size() - (i + params.Seed); j < edgePoints.size(); ++j)
							{
								size_t idx = random(0, j, params.Seed+j); // generate a random integer in range [0, j]
            
								if (vals.find(idx) != vals.end())
									idx = j;
            
								sample.push_back(edgePoints[idx]);
								vals.insert(idx);
							 }
						}else{
							if (n > edgePoints.size())
								throw std::range_error("Subset size out of range");
							 std::set<size_t> vals;
						     for (size_t j = edgePoints.size() - n; j < edgePoints.size(); ++j)
							{
								size_t idx = random(0, j); // generate a random integer in range [0, j]
            
								if (vals.find(idx) != vals.end())
									idx = j;
            
								sample.push_back(edgePoints[idx]);
								vals.insert(idx);
							 }
						}
                        
                        cv::RotatedRect ellipseSampleFit = fitEllipse(sample);
                        // Normalise ellipse to have width as the major axis.
                        if (ellipseSampleFit.size.height > ellipseSampleFit.size.width)
                        {
                            ellipseSampleFit.angle = std::fmod(ellipseSampleFit.angle + 90, 180);
                            std::swap(ellipseSampleFit.size.height, ellipseSampleFit.size.width);
                        }
                        
                        cv::Size s = ellipseSampleFit.size;
                        // Discard useless ellipses early
                        if (!ellipseSampleFit.center.inside(bb)
                            || s.height > params.Radius_Max*2
                            || s.width > params.Radius_Max*2
                            || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                            || s.height > 4*s.width
                            || s.width > 4*s.height
                            )
                        {
                            // Bad ellipse! Go to your room!
                            continue;
                        }
                        
                        // Use conic section's algebraic distance as an error measure
                        ConicSection conicSampleFit(ellipseSampleFit);
                        
                        // Check if sample's gradients are correctly oriented
                        if (params.EarlyRejection)
                        {
                            bool gradientCorrect = true;
                            for(size_t i = 0; i < sample.size(); ++i)
                            {
								//cv::Point2f grad = conicSampleFit.algebraicGradientDir(sample.at(i));
								//float dx = mDX(cv::Point(sample.at(i).x, sample.at(i).y));
                                //float dy = mDY(cv::Point(sample.at(i).x, sample.at(i).y));
								cv::Point2f grad = conicSampleFit.algebraicGradientDir((cv::Point_<float>)sample[i]); //支持安卓
                                //float dx = mDX(cv::Point(sample[i].x, sample[i].y)); //支持安卓
                                //float dy = mDY(cv::Point(sample[i].x, sample[i].y)); //支持安卓
								float dx = mDX(cv::Point(sample[i].x, sample[i].y)); //支持安卓
								float dy = mDY(cv::Point(sample[i].x, sample[i].y)); //支持安卓
                                
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
                        ConicSection conicInlierFit = conicSampleFit;
                        std::vector<cv::Point2f> inliers, prevInliers;
                        
                        // Iteratively find inliers, and re-fit the ellipse
                        for (int i = 0; i < params.InlierIterations; ++i)
                        {
                            // Get error scale for 1px out on the minor axis
                            cv::Point2f minorAxis(-std::sin(PI/180.0*ellipseInlierFit.angle), std::cos(PI/180.0*ellipseInlierFit.angle));
                            cv::Point2f minorAxisPlus1px = ellipseInlierFit.center + (ellipseInlierFit.size.height/2 + 1)*minorAxis;
                            float errOf1px = conicInlierFit.distance(minorAxisPlus1px);
                            float errorScale = 1.0f/errOf1px;
                            
                            // Find inliers
                            //inliers.reserve(edgePoints.size());
							//inliers.reserve(edgePoints.size()); //支持安卓
                            const float MAX_ERR = 2;
							for(size_t i = 0; i < edgePoints.size(); ++i)
                            {
                                float err = errorScale*conicInlierFit.distance(edgePoints.at(i));
                                
                                if (err*err < MAX_ERR*MAX_ERR)
                                    inliers.push_back(edgePoints.at(i));
                            }
                            
                            if (inliers.size() < n) {
                                inliers.clear();
                                continue;
                            }
                            
                            // Refit ellipse to inliers
                            ellipseInlierFit = fitEllipse(inliers);
                            conicInlierFit = ConicSection(ellipseInlierFit);
                            
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
                            || s.height < params.Radius_Min*2 && s.width < params.Radius_Min*2
                            || s.height > 4*s.width
                            || s.width > 4*s.height
                            )
                        {
                            // Bad ellipse! Go to your room!
                            continue;
                        }
                        
                        // Calculate ellipse goodness
                        double ellipseGoodness = 0;
                        if (params.ImageAwareSupport)
                        {
                            for(size_t i = 0; i < inliers.size(); ++i)
                            {
                                cv::Point2f grad = conicInlierFit.algebraicGradientDir(inliers.at(i));
                                float dx = mDX(inliers.at(i));
                                float dy = mDY(inliers.at(i));
                                
                                double edgeStrength = dx*grad.x + dy*grad.y;
                                
                                ellipseGoodness += edgeStrength;
                            }
                        }
                        else
                        {
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
                    //std::cout << "Ransac end" << std::endl;
                } //结束operate
                
                void join(EllipseRansac& other)
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
            }; // EllipseRansac结构体定义
            
			// 定义变量为ransac的EllipseRansac结构体
            EllipseRansac ransac(params, edgePoints, n, bbPupil, out.mPupilSobelX, out.mPupilSobelY);
			//去掉了TBB
            try
            { 
				ransac.operatorTmp(k);
            }
            catch (std::exception& e)
            {
                const char* c = e.what();
                std::cerr << e.what() << std::endl;
            }
            inliers = ransac.out.bestInliers;
            //log.add("goodness", ransac.out.bestEllipseGoodness);
            
            out.earlyRejections = ransac.out.earlyRejections;
            out.earlyTermination = ransac.out.earlyTermination;
            
            
            cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
            ConicSection conicBestFit(ellipseBestFit);
			for(size_t i = 0; i < edgePoints.size(); ++i)
            {
                cv::Point2f grad = conicBestFit.algebraicGradientDir(edgePoints.at(i));
                float dx = out.mPupilSobelX(p);
                float dy = out.mPupilSobelY(p);
                
                out.edgePoints.push_back(EdgePoint(edgePoints.at(i), dx*grad.x + dy*grad.y));
            }
            
            elPupil = ellipseBestFit;
            elPupil.center.x += roiPupil.x;
            elPupil.center.y += roiPupil.y;
        }
        
        if (inliers.size() == 0)
            return false;
        
        cv::Point2f pPupil = elPupil.center;
        
        out.pPupil = pPupil;
        out.elPupil = elPupil;
        out.inliers = inliers;
        
        return true;
    }
    
    return false;
}
