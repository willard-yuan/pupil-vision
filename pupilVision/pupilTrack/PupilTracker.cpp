#include "PupilTracker.h"
#include "cvx.h"
#include "utils.h"
#include "erase_specular.h"
#include "EllipseRansac.h"
#include <libiomp/omp.h>

class HaarSurroundFeature
{
public:
    HaarSurroundFeature(int r1, int r2) : r_inner(r1), r_outer(r2) //构造函数
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

bool pupiltracker::findPupilEllipse(const pupiltracker::TrackerParams& params, const cv::Mat& m, pupiltracker::findPupilEllipse_out& out, bool eraseSpecular){
    
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
    
    if (eraseSpecular) erase_specular(mEye);
    
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

	
    //获取最小haar相应
    std::pair<double,cv::Point2f> minValOut;
#pragma omp parallel for
    for (int r = params.Radius_Min; r < params.Radius_Max; r+=rstep){
        // Get Haar feature
        int r_inner = r;
        int r_outer = 3*r;
        HaarSurroundFeature f(r_inner, r_outer);

        for (int i = 0, y = r + i*ystep;  i < ((mEye.rows-r - r - 1)/ystep + 1); y += ystep,++i){
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
        }
    }

    // Paradoxically, a good Haar fit won't catch the entire pupil, so expand it a bit
    haarRadius = (int)(haarRadius * SQRT_2);
    
    // 在haar最大响应点处获取瞳孔ROI
    
    cv::Rect roiHaarPupil = cvx::roiAround(cv::Point(pHaarPupil.x, pHaarPupil.y), haarRadius);
    cv::Mat_<uchar> mHaarPupil;
    cvx::getROI(mEye, mHaarPupil, roiHaarPupil);
    
    out.roiHaarPupil = roiHaarPupil;
    out.mHaarPupil = mHaarPupil;
    
    // 获取瞳孔区域的直方图，并用Kmeans进行分割
    
    // 计算直方图
    const int bins = 256;
    cv::Mat_<float> hist;
    {
        int channels[] = {0};
        int sizes[] = {bins};
        float range[2] = {0, 256};
        const float* ranges[] = {range};
        cv::calcHist(&mHaarPupil, 1, channels, cv::Mat(), hist, 1, sizes, ranges);
    }
    out.histPupil = hist;
    
    // KMeans聚类
    float threshold;
    {
        // Try various candidate centres, return the one with minimal label distance
        float candidate0[2] = {0, 0};
        float candidate1[2] = {128, 255};
        float bestDist = std::numeric_limits<float>::infinity();
        float bestThreshold = std::numeric_limits<float>::quiet_NaN();
        
        for (int i = 0; i < 2; i++){
            cv::Mat_<uchar> labels;
            float centres[2] = {candidate0[i], candidate1[i]};
            float dist = cvx::histKmeans(hist, 0, 256, 2, centres, labels, cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0.0));
            
            float thisthreshold = (centres[0] + centres[1])/2;
			if (dist < bestDist && !(thisthreshold != thisthreshold)){
                bestDist = dist;
                bestThreshold = thisthreshold;
            }
        }
        if ((bestThreshold != bestThreshold)){
            // If kmeans gives a degenerate solution, exit early
            return false;
        }
        
        threshold = bestThreshold;
    }
    
    // 阈值化处理
    // mPupilThresh:保存阈值化处理后的瞳孔区域
    cv::Mat_<uchar> mPupilThresh;
    cv::threshold(mHaarPupil, mPupilThresh, threshold, 255, cv::THRESH_BINARY_INV);
    out.threshold = threshold;
    out.mPupilThresh = mPupilThresh;
    
    // ---------------------------------------------
    // Find best region in the segmented pupil image
    // ---------------------------------------------
    
    cv::Rect bbPupilThresh;
    cv::RotatedRect elPupilThresh;
    // 寻找最好的瞳孔区域
    {
		cv::Mat_<uchar> mPupilContours = mPupilThresh.clone();
		std::vector<std::vector<cv::Point> > contours;
        cv::findContours(mPupilContours, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        
        if (contours.size() == 0) return false;
        std::vector<cv::Point>& maxContour = contours[0];
        double maxContourArea = cv::contourArea(maxContour);
#pragma omp parallel for
        for(std::size_t i = 0; i < contours.size(); ++i){
            double area = cv::contourArea((cv::InputArray&)contours[i]);
            if (area > maxContourArea){
                maxContourArea = area;
                maxContour = contours[i];
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
    
    // 瞳孔区域预处理
    {
        const int padding = 3;
        
        cv::Rect roiPadded(roiPupil.x-padding, roiPupil.y-padding, roiPupil.width+2*padding, roiPupil.height+2*padding);
        // First get an ROI around the approximate pupil location
        cvx::getROI(mEye, mPupil, roiPadded, cv::BORDER_REPLICATE);
        
        cv::Mat morphologyDisk = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mPupil, mPupilOpened, cv::MORPH_OPEN, morphologyDisk, cv::Point(-1,-1), 2);
        
        if (params.CannyBlur > 0){
            cv::GaussianBlur(mPupilOpened, mPupilBlurred, cv::Size(), params.CannyBlur);
        }else{
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
    out.mPupil = mPupil; // 保存处理的瞳孔区域
    out.mPupilOpened = mPupilOpened;
    out.mPupilBlurred = mPupilBlurred;
    out.mPupilSobelX = mPupilSobelX;
    out.mPupilSobelY = mPupilSobelY;
    out.mPupilEdges = mPupilEdges;
    
    
    std::vector<cv::Point2f> edgePoints;
	//找到非零像素点的各个坐标
#pragma omp parallel
    {
    #pragma omp for
        for(int y = 0; y < mPupilEdges.rows; y++){
            uchar* val = mPupilEdges[y];
            for(int x = 0; x < mPupilEdges.cols; x++, val++){
                if(*val == 0) continue;
                edgePoints.push_back(cv::Point2f(x + 0.5f, y + 0.5f));
            }
        }
    }
    
    
    // ---------------------------
    // Fit an ellipse to the edges
    // ---------------------------
    
    cv::RotatedRect elPupil;
    std::vector<cv::Point2f> inliers;
    // 椭圆拟合
    {
        const double p = 0.999; // Desired probability that only inliers are selected
        double w = params.PercentageInliers/100.0; // Probability that a point is an inlier
        const int n = 5; // Number of points needed for a model
        
        if (params.PercentageInliers == 0)
            return false;
        
		// 如果找到的轮廓点个数大于5个，则进行椭圆拟合
        if (edgePoints.size() >= n) // Minimum points for ellipse
        {
            double wToN = std::pow(w,n);
            int k = static_cast<int>(std::log(1-p)/std::log(1 - wToN)  + 2*std::sqrt(1 - wToN)/wToN);
            
            out.ransacIterations = k;
            
			// 定义变量为ransac的EllipseRansac结构体
            EllipseRansac ransac(params, edgePoints, n, bbPupil, out.mPupilSobelX, out.mPupilSobelY);
            try{
                ransac.operatorTmp(k);
            }catch (std::exception& e){
                //const char* c = e.what();
                std::cerr << e.what() << std::endl;
            }
            inliers = ransac.out.bestInliers;
            
            out.earlyRejections = ransac.out.earlyRejections;
            out.earlyTermination = ransac.out.earlyTermination;
            
            
            cv::RotatedRect ellipseBestFit = ransac.out.bestEllipse;
            ConicSection conicBestFit(ellipseBestFit);
			for(size_t i = 0; i < edgePoints.size(); ++i){
                cv::Point2f grad = conicBestFit.algebraicGradientDir((cv::Point_<float>)edgePoints[i]);
                float dx = out.mPupilSobelX(p);
                float dy = out.mPupilSobelY(p);
                out.edgePoints.push_back(EdgePoint((cv::Point_<float>)edgePoints[i], dx*grad.x + dy*grad.y));
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
