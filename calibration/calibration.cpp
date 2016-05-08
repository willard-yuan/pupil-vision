//
//  main.cpp
//  Hello-Eigen
//
//  Created by willard on 12/30/15.
//  Copyright © 2015 wilard. All rights reserved.
//

// 输入
// 眼睛：一维数组保存int ECoords = [xE1, yE1, xE2, yE2, ... , xE9, yE9]
// 屏幕坐标在这里固定：int SCoords = [xScrn1, yScrn1, xScrn2, yScrn2, ..., xScrn9, yScrn9]
// 测试眼睛坐标：int xTE, int yTE

// 输出
// 屏幕坐标：int xTScrn, int yTScrn

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

#define MAXBUFSIZE  ((int) 1e6)

int round_int( float r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5);
}

MatrixXd readMatrix(const char *filename)
{
    int cols = 0, rows = 0;
    double buff[MAXBUFSIZE];
    
    // Read numbers from file into buffer.
    ifstream infile;
    infile.open(filename);
    while (! infile.eof())
    {
        string line;
        getline(infile, line);
        
        int temp_cols = 0;
        stringstream stream(line);
        while(! stream.eof())
            stream >> buff[cols*rows+temp_cols++];
        
        if (temp_cols == 0)
            continue;
        
        if (cols == 0)
            cols = temp_cols;
        
        rows++;
    }
    
    infile.close();
    
    //rows--;
    
    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];
    
    return result;
};

MatrixXd normalize(MatrixXd cal_pt_cloud, float widthPupil, float heightPupil, float widthScreen, float heightScreen){
    //normalize return as float
    cal_pt_cloud.col(0) = cal_pt_cloud.col(0)/widthPupil;
    cal_pt_cloud.col(1) = cal_pt_cloud.col(1)/heightPupil;
    cal_pt_cloud.col(2) = cal_pt_cloud.col(2)/widthScreen;
    cal_pt_cloud.col(3) = cal_pt_cloud.col(3)/heightScreen;
    return cal_pt_cloud;
    }

vector<float> denormalize(vector<float> pos, float widthScreen, float heightScreen){
    //denormalize
    pos[0] *= widthScreen;
    pos[1] *= heightScreen;
    return pos;
    }

MatrixXd make_model(MatrixXd cal_pt_cloud, int n=7){
    int pointsNum = (int)cal_pt_cloud.rows();
    MatrixXd X(pointsNum, 1);
    MatrixXd Y(pointsNum, 1);
    for(int i = 0; i < pointsNum; i++){
        //X(i, 0) = ECoords[2*i];
        //Y(i, 0) = ECoords[2*i + 1];
        X(i, 0) = cal_pt_cloud(i, 0);
        Y(i, 0) = cal_pt_cloud(i, 1);
    }
    MatrixXd XX = X.cwiseProduct(X);
    MatrixXd YY = Y.cwiseProduct(Y);
    MatrixXd XY = X.cwiseProduct(Y);
    MatrixXd XXYY = XX.cwiseProduct(YY);
    MatrixXd Ones(pointsNum,1);
    for(int i = 0; i < pointsNum; i++){
        Ones(i,0) = 1;
    }
    MatrixXd ZX(pointsNum, 1);
    MatrixXd ZY(pointsNum, 1);
    for(int i = 0; i < pointsNum; i++){
        ZX(i, 0) = cal_pt_cloud(i, 2);
        ZY(i, 0) = cal_pt_cloud(i, 3);
    }
    MatrixXd M(pointsNum, 9);
    M << X, Y, XX, YY, XY, XXYY, Ones, ZX, ZY;
    return M;
}

vector<MatrixXd> fit_poly_surface(MatrixXd cal_pt_cloud, int n=7){
    vector<MatrixXd> cx_cy_errx_erry;
    long int pointsNum = cal_pt_cloud.rows();
    MatrixXd M = make_model(cal_pt_cloud, n);
    
    MatrixXd Mblock(pointsNum, n);
    for(int i = 0; i < pointsNum; i++)
        for(int j = 0; j < n; j++)
            Mblock(i, j) = M(i, j);
    JacobiSVD<MatrixXd> svd(Mblock, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd Vt = svd.matrixV();
    
    // 确认准确
    MatrixXd w = svd.singularValues();
    
    MatrixXd InvW(n, 1);
    for(int i = 0; i < n; i++){
        InvW(i,0) = 1./w(i,0);
    }
    
    MatrixXd diagInvW = InvW.asDiagonal();
    
    MatrixXd Ut = U.transpose();
    
    MatrixXd dotWUt = diagInvW*Ut;
    MatrixXd pseudINV = Vt*dotWUt;
    
    MatrixXd tmpx(pointsNum, 1);
    MatrixXd tmpy(pointsNum, 1);
    for(int i = 0; i < pointsNum; i++){
        tmpx(i, 0) = M(i, n);
        tmpy(i, 0) = M(i, n+1);
    }
    MatrixXd cx = pseudINV*tmpx;
    MatrixXd cy = pseudINV*tmpy;
    
    //compute model error in world screen units if screen_res specified
    MatrixXd err_x = Mblock*cx - tmpx;
    MatrixXd err_y = Mblock*cy - tmpy;
    
    cx_cy_errx_erry.push_back(cx);
    cx_cy_errx_erry.push_back(cy);
    cx_cy_errx_erry.push_back(err_x);
    cx_cy_errx_erry.push_back(err_y);
    return cx_cy_errx_erry;
}

vector<float> make_map_function_fn(MatrixXd cx, MatrixXd cy, int n, float xTE, float yTE){
    float xTScrn, yTScrn;
    vector<float> tScrn;
    if(n == 7){
        xTScrn = cx(0,0)*xTE + cx(1,0)*yTE + cx(2,0)*xTE*xTE + cx(3,0)*yTE*yTE + cx(4,0)*xTE*yTE + cx(5,0)*yTE*yTE*xTE*xTE + cx(6,0);
        tScrn.push_back(xTScrn);
        yTScrn = cy(0,0)*xTE + cy(1,0)*yTE + cy(2,0)*xTE*xTE + cy(3,0)*yTE*yTE + cy(4,0)*xTE*yTE + cy(5,0)*yTE*yTE*xTE*xTE + cy(6,0);
        tScrn.push_back(yTScrn);
    }
    return tScrn;
}

MatrixXd fit_error_screen(MatrixXd err_x, MatrixXd err_y, float screen_x, float screen_y){
    vector<float> errs;
    err_x *= screen_x/2.;
    err_y *= screen_y/2.;
    //err_dist = np.sqrt(err_x*err_x + err_y*err_y)
    //err_mean = np.sum(err_dist)/len(err_dist)
    MatrixXd sqrtTmp = err_x.cwiseProduct(err_x) + err_y.cwiseProduct(err_y);
    MatrixXd err_dist = sqrtTmp.cwiseSqrt();
    float err_mean = err_dist.sum()/(float)err_dist.rows();
    
    //err_rms=np.sqrt(np.sum(err_dist*err_dist)/len(err_dist))
    float sumErr_dist = (err_dist.cwiseProduct(err_dist)).sum();
    float err_rms = std::sqrt(sumErr_dist/(float)err_dist.rows());
    return err_dist;
}

vector<float> obtainThreeColumPoints(MatrixXd data, std::vector<float> xEsDifTmp){
    
    int threhold = 8;
    
    vector<float> iColumPoints;
    
    std::vector<int> selctIdex1;
    std::vector<int> selctIdex2;
    
    // 竖直第i列3个点
    for(int i = 2; i < data.rows(); i++){
        if(xEsDifTmp[i-2] < threhold && xEsDifTmp[i-1] < threhold && xEsDifTmp[i] < threhold){
            selctIdex1.push_back(i);
        }
    }
    for(int i = 2; i < data.rows(); i++){
        if(xEsDifTmp[i-2] < threhold && xEsDifTmp[i-1] < threhold && xEsDifTmp[i] < threhold){
            selctIdex2.push_back(i-2);
        }
    }
    std::vector<int> sortedIdx;
    std::sort(selctIdex1.begin(), selctIdex1.end());
    std::sort(selctIdex2.begin(), selctIdex2.end());
    std::set_union(selctIdex1.begin(), selctIdex1.end(), selctIdex2.begin(), selctIdex2.end(), std::back_inserter(sortedIdx));
    
    float sumTmp = 0;
    float sumTmpH = 0;
    
    //第i列坐标索引
    vector<int> ttTmp;
    ttTmp.push_back(sortedIdx[0]);
    for(int i = 1; i < sortedIdx.size(); i++){
        if(sortedIdx[i] - sortedIdx[i-1] > 2){
            ttTmp.push_back(sortedIdx[i-1]);
            ttTmp.push_back(sortedIdx[i]);
        }
    }
    ttTmp.push_back(sortedIdx[sortedIdx.size()-1]);
    
    //求第i列第1个坐标
    sumTmp = 0;
    sumTmpH = 0;
    for(int i = ttTmp[0]; i <= ttTmp[1]; i++){
        sumTmpH = sumTmpH + data(i,0);
        sumTmp = sumTmp + data(i,1);
    }
    float pointsH_A = sumTmpH/(ttTmp[1]-ttTmp[0]+1); //第i列第1个横坐标
    float pointsV_A = sumTmp/(ttTmp[1]-ttTmp[0]+1); //第i列第1个纵坐标
    iColumPoints.push_back(pointsH_A);
    iColumPoints.push_back(pointsV_A);
    
    //求第i列第2个坐标
    sumTmp = 0;
    sumTmpH = 0;
    for(int i = ttTmp[2]; i <= ttTmp[3]; i++){
        sumTmpH = sumTmpH + data(i,0);
        sumTmp = sumTmp + data(i,1);
    }
    float pointsH_D = sumTmpH/(ttTmp[3]-ttTmp[2]+1); //第i列第2个横坐标
    float pointsV_D = sumTmp/(ttTmp[3]-ttTmp[2]+1); //第i列第2个纵坐标
    iColumPoints.push_back(pointsH_D);
    iColumPoints.push_back(pointsV_D);
    
    //求第i列第3个坐标
    sumTmp = 0;
    sumTmpH = 0;
    for(int i = ttTmp[4]; i <= ttTmp[5]; i++){
        sumTmpH = sumTmpH + data(i,0);
        sumTmp = sumTmp + data(i,1);
    }
    float pointsH_G = sumTmpH/(ttTmp[5]-ttTmp[4]+1); //第i列第2个横坐标
    float pointsV_G = sumTmp/(ttTmp[5]-ttTmp[4]+1); //第i列第2个纵坐标
    iColumPoints.push_back(pointsH_G);
    iColumPoints.push_back(pointsV_G);
    
    return iColumPoints;
}

MatrixXd selctGoodPoints(MatrixXd data){
    
    MatrixXd selct_pt_cloud(9, 4);
    
    std::vector<float> xEs;
    for (size_t i = 0; i < data.rows(); i++){
        xEs.push_back(data(i,0));
    }
    for(int i = 0; i < xEs.size(); i++){
        std::cout << i << ":" << xEs[i] << std::endl;
    }

    cv::Mat data_mat((int)data.rows(), 1, CV_32FC1, &xEs[0]);  // ** Make 1 column Mat from vector
    std::vector<int> labels;
    std::vector<float> centers;
    cv::kmeans(data_mat, 3, labels,     // ** Pass 1 column Mat from Mat to kmeans
               cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10, 0.1),
               3, cv::KMEANS_PP_CENTERS, centers);
    
    std::sort(centers.begin(), centers.end(), greater<float>()); //由大到小排序
    
    std::vector<float> xEsDifTmp1;
    std::vector<float> xEsDifTmp2;
    std::vector<float> xEsDifTmp3;
    float difTmp;
    for(int i = 0; i < xEs.size(); i++){
        difTmp = abs(xEs[i] - round_int(centers[0])); //四舍五入方法
        xEsDifTmp1.push_back(difTmp);
        difTmp = abs(xEs[i] - round_int(centers[1]));
        xEsDifTmp2.push_back(difTmp);
        difTmp = abs(xEs[i] - round_int(centers[2]));
        xEsDifTmp3.push_back(difTmp);
    }
    
    
    vector<float> firstColumPoints = obtainThreeColumPoints(data, xEsDifTmp1);
    vector<float> secondColumPoints = obtainThreeColumPoints(data, xEsDifTmp2);
    vector<float> thirdColumPoints = obtainThreeColumPoints(data, xEsDifTmp3);
    
    //第1列---------------------------------------
    //A点
    selct_pt_cloud(0,0) = firstColumPoints[0];
    selct_pt_cloud(0,1) = firstColumPoints[1];
    selct_pt_cloud(0,2) = 32;
    selct_pt_cloud(0,3) = 33;
    //D点
    selct_pt_cloud(3,0) = firstColumPoints[2];
    selct_pt_cloud(3,1) = firstColumPoints[3];
    selct_pt_cloud(3,2) = 32;
    selct_pt_cloud(3,3) = 232;
    //G点
    selct_pt_cloud(6,0) = firstColumPoints[4];
    selct_pt_cloud(6,1) = firstColumPoints[5];
    selct_pt_cloud(6,2) = 32;
    selct_pt_cloud(6,3) = 432;
    
    //第2列---------------------------------------
    //B点
    selct_pt_cloud(1,0) = secondColumPoints[0];
    selct_pt_cloud(1,1) = secondColumPoints[1];
    selct_pt_cloud(1,2) = 510;
    selct_pt_cloud(1,3) = 33;
    //E点
    selct_pt_cloud(4,0) = secondColumPoints[2];
    selct_pt_cloud(4,1) = secondColumPoints[3];
    selct_pt_cloud(4,2) = 510;
    selct_pt_cloud(4,3) = 232;
    //H点
    selct_pt_cloud(7,0) = secondColumPoints[4];
    selct_pt_cloud(7,1) = secondColumPoints[5];
    selct_pt_cloud(7,2) = 510;
    selct_pt_cloud(7,3) = 432;
    
    //第3列---------------------------------------
    //C点
    selct_pt_cloud(2,0) = thirdColumPoints[0];
    selct_pt_cloud(2,1) = thirdColumPoints[1];
    selct_pt_cloud(2,2) = 987;
    selct_pt_cloud(2,3) = 33;
    //F点
    selct_pt_cloud(5,0) = thirdColumPoints[2];
    selct_pt_cloud(5,1) = thirdColumPoints[3];
    selct_pt_cloud(5,2) = 987;
    selct_pt_cloud(5,3) = 232;
    //I点
    selct_pt_cloud(8,0) = thirdColumPoints[4];
    selct_pt_cloud(8,1) = thirdColumPoints[5];
    selct_pt_cloud(8,2) = 987;
    selct_pt_cloud(8,3) = 432;

    return selct_pt_cloud;
}

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();
    
    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);
    
    matrix.conservativeResize(numRows,numCols);
}

vector<float> linearModel_lTop_rDown(MatrixXd &ES1, MatrixXd &ES2, vector<float> &xyE){
    vector<float> xyS;
    float xS = (ES1(0,2)-ES2(0,2))*(xyE[0]-ES2(0,0))/(ES1(0,0)-ES2(0,0)) + ES2(0,2);
    float yS = (ES1(0,3)-ES2(0,3))*(xyE[1]-ES2(0,1))/(ES1(0,1)-ES2(0,1)) + ES2(0,3);
    xyS.push_back(xS);
    xyS.push_back(yS);
    return xyS;
}

vector<float> linearModel_lDown_rTop(MatrixXd &ES1, MatrixXd &ES2, vector<float> &xyE){
    vector<float> xyS;
    float xS = (ES1(0,2)-ES2(0,2))*(xyE[0]-ES2(0,0))/(ES1(0,0)-ES2(0,0)) + ES2(0,2);
    float yS = (ES1(0,3)-ES2(0,3))*(xyE[1]-ES2(0,1))/(ES1(0,1)-ES2(0,1)) + ES2(0,3);
    xyS.push_back(xS);
    xyS.push_back(yS);
    return xyS;
}

//二元二次非线性模型
// @归一化后的9个坐标矩阵
// @归一化的眼睛坐标
// @归一化的屏幕坐标
vector<float> nolinearModel(MatrixXd &cal_pt_cloud, vector<float> &xyE){
    vector<float> xyS;
    MatrixXd T;
    MatrixXd S(4, 4);
    MatrixXd C(4, 4);
    if(xyE[0] > cal_pt_cloud(4,0)){
        if(xyE[1] <= cal_pt_cloud(4,1)){
            S << cal_pt_cloud(0,2), cal_pt_cloud(1,2), cal_pt_cloud(3,2), cal_pt_cloud(4,2),
            cal_pt_cloud(0,3), cal_pt_cloud(1,3), cal_pt_cloud(3,3), cal_pt_cloud(4,3),
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(0,0), cal_pt_cloud(1,0), cal_pt_cloud(3,0), cal_pt_cloud(4,0),
            cal_pt_cloud(0,1), cal_pt_cloud(1,1), cal_pt_cloud(3,1), cal_pt_cloud(4,1),
            cal_pt_cloud(0,0)*cal_pt_cloud(0,1), cal_pt_cloud(1,0)*cal_pt_cloud(1,1), cal_pt_cloud(3,0)*cal_pt_cloud(3,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(4,1);
            xyC << xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }else{
            S << cal_pt_cloud(3,2), cal_pt_cloud(4,2), cal_pt_cloud(6,2), cal_pt_cloud(7,2),
            cal_pt_cloud(3,3), cal_pt_cloud(4,3), cal_pt_cloud(6,3), cal_pt_cloud(7,3),
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(3,0), cal_pt_cloud(4,0), cal_pt_cloud(6,0), cal_pt_cloud(7,0),
            cal_pt_cloud(3,1), cal_pt_cloud(4,1), cal_pt_cloud(6,1), cal_pt_cloud(7,1),
            cal_pt_cloud(3,0)*cal_pt_cloud(3,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(6,0)*cal_pt_cloud(6,1), cal_pt_cloud(7,0)*cal_pt_cloud(7,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(4,1);
            xyC << xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }
    }else{
        if(xyE[1] <= cal_pt_cloud(4,1)){
            S << cal_pt_cloud(1,2), cal_pt_cloud(2,2), cal_pt_cloud(4,2), cal_pt_cloud(5,2),
            cal_pt_cloud(1,3), cal_pt_cloud(2,3), cal_pt_cloud(4,3), cal_pt_cloud(5,3),
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(1,0), cal_pt_cloud(2,0), cal_pt_cloud(4,0), cal_pt_cloud(5,0),
            cal_pt_cloud(1,1), cal_pt_cloud(2,1), cal_pt_cloud(4,1), cal_pt_cloud(5,1),
            cal_pt_cloud(1,0)*cal_pt_cloud(1,1), cal_pt_cloud(2,0)*cal_pt_cloud(2,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(5,0)*cal_pt_cloud(5,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(4,1);
            xyC << xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }else{
            S << cal_pt_cloud(4,2), cal_pt_cloud(5,2), cal_pt_cloud(7,2), cal_pt_cloud(8,2),
            cal_pt_cloud(4,3), cal_pt_cloud(5,3), cal_pt_cloud(7,3), cal_pt_cloud(8,3),
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(4,0), cal_pt_cloud(5,0), cal_pt_cloud(7,0), cal_pt_cloud(8,0),
            cal_pt_cloud(4,1), cal_pt_cloud(5,1), cal_pt_cloud(7,1), cal_pt_cloud(8,1),
            cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(5,0)*cal_pt_cloud(5,1), cal_pt_cloud(7,0)*cal_pt_cloud(7,1), cal_pt_cloud(8,0)*cal_pt_cloud(8,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(4,1);
            xyC << xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }
    }
    return xyS;
}

//二元二次非线性模型
// @归一化后的9个坐标矩阵
// @归一化的眼睛坐标
// @归一化的屏幕坐标
vector<float> nolinear2Model(MatrixXd &cal_pt_cloud, vector<float> &xyE){
    vector<float> xyS;
    MatrixXd T;
    MatrixXd S(6, 4);
    MatrixXd C(6, 4);
    if(xyE[0] > cal_pt_cloud(4,0)){
        if(xyE[1] <= cal_pt_cloud(4,1)){
            S << cal_pt_cloud(0,2), cal_pt_cloud(1,2), cal_pt_cloud(3,2), cal_pt_cloud(4,2),
            cal_pt_cloud(0,3), cal_pt_cloud(1,3), cal_pt_cloud(3,3), cal_pt_cloud(4,3),
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(0,0)*cal_pt_cloud(0,0), cal_pt_cloud(1,0)*cal_pt_cloud(1,0), cal_pt_cloud(3,0)*cal_pt_cloud(3,0), cal_pt_cloud(4,0)*cal_pt_cloud(4,0),
            cal_pt_cloud(0,1)*cal_pt_cloud(0,1), cal_pt_cloud(1,1)*cal_pt_cloud(0,1), cal_pt_cloud(3,1)*cal_pt_cloud(0,1), cal_pt_cloud(4,1)*cal_pt_cloud(0,1),
            cal_pt_cloud(0,0), cal_pt_cloud(1,0), cal_pt_cloud(3,0), cal_pt_cloud(4,0),
            cal_pt_cloud(0,1), cal_pt_cloud(1,1), cal_pt_cloud(3,1), cal_pt_cloud(4,1),
            cal_pt_cloud(0,0)*cal_pt_cloud(0,1), cal_pt_cloud(1,0)*cal_pt_cloud(1,1), cal_pt_cloud(3,0)*cal_pt_cloud(3,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(6,1);
            xyC << xyE[0]*xyE[0],
            xyE[1]*xyE[1],
            xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }else{
            S << cal_pt_cloud(3,2), cal_pt_cloud(4,2), cal_pt_cloud(6,2), cal_pt_cloud(7,2),
            cal_pt_cloud(3,3), cal_pt_cloud(4,3), cal_pt_cloud(6,3), cal_pt_cloud(7,3),
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(3,0)*cal_pt_cloud(3,0), cal_pt_cloud(4,0)*cal_pt_cloud(3,0), cal_pt_cloud(6,0)*cal_pt_cloud(3,0), cal_pt_cloud(7,0)*cal_pt_cloud(3,0),
            cal_pt_cloud(3,1)*cal_pt_cloud(3,1), cal_pt_cloud(4,1)*cal_pt_cloud(3,1), cal_pt_cloud(6,1)*cal_pt_cloud(3,1), cal_pt_cloud(7,1)*cal_pt_cloud(3,1),
            cal_pt_cloud(3,0), cal_pt_cloud(4,0), cal_pt_cloud(6,0), cal_pt_cloud(7,0),
            cal_pt_cloud(3,1), cal_pt_cloud(4,1), cal_pt_cloud(6,1), cal_pt_cloud(7,1),
            cal_pt_cloud(3,0)*cal_pt_cloud(3,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(6,0)*cal_pt_cloud(6,1), cal_pt_cloud(7,0)*cal_pt_cloud(7,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(6,1);
            xyC << xyE[0]*xyE[0],
            xyE[1]*xyE[1],
            xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }
    }else{
        if(xyE[1] <= cal_pt_cloud(4,1)){
            S << cal_pt_cloud(1,2), cal_pt_cloud(2,2), cal_pt_cloud(4,2), cal_pt_cloud(5,2),
            cal_pt_cloud(1,3), cal_pt_cloud(2,3), cal_pt_cloud(4,3), cal_pt_cloud(5,3),
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(1,0)*cal_pt_cloud(1,0), cal_pt_cloud(2,0)*cal_pt_cloud(2,0), cal_pt_cloud(4,0)*cal_pt_cloud(4,0), cal_pt_cloud(5,0)*cal_pt_cloud(5,0),
            cal_pt_cloud(1,1)*cal_pt_cloud(1,1), cal_pt_cloud(2,1)*cal_pt_cloud(2,1), cal_pt_cloud(4,1)*cal_pt_cloud(4,1), cal_pt_cloud(5,1)*cal_pt_cloud(5,1),
            cal_pt_cloud(1,0), cal_pt_cloud(2,0), cal_pt_cloud(4,0), cal_pt_cloud(5,0),
            cal_pt_cloud(1,1), cal_pt_cloud(2,1), cal_pt_cloud(4,1), cal_pt_cloud(5,1),
            cal_pt_cloud(1,0)*cal_pt_cloud(1,1), cal_pt_cloud(2,0)*cal_pt_cloud(2,1), cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(5,0)*cal_pt_cloud(5,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(6,1);
            xyC << xyE[0]*xyE[0],
            xyE[1]*xyE[1],
            xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }else{
            S << cal_pt_cloud(4,2), cal_pt_cloud(5,2), cal_pt_cloud(7,2), cal_pt_cloud(8,2),
            cal_pt_cloud(4,3), cal_pt_cloud(5,3), cal_pt_cloud(7,3), cal_pt_cloud(8,3),
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0;
            C << cal_pt_cloud(4,0)*cal_pt_cloud(4,0), cal_pt_cloud(5,0)*cal_pt_cloud(5,0), cal_pt_cloud(7,0)*cal_pt_cloud(7,0), cal_pt_cloud(8,0)*cal_pt_cloud(4,0),
            cal_pt_cloud(4,1)*cal_pt_cloud(4,1), cal_pt_cloud(5,1)*cal_pt_cloud(5,1), cal_pt_cloud(7,1)*cal_pt_cloud(7,1), cal_pt_cloud(8,1)*cal_pt_cloud(8,1),
            cal_pt_cloud(4,0), cal_pt_cloud(5,0), cal_pt_cloud(7,0), cal_pt_cloud(8,0),
            cal_pt_cloud(4,1), cal_pt_cloud(5,1), cal_pt_cloud(7,1), cal_pt_cloud(8,1),
            cal_pt_cloud(4,0)*cal_pt_cloud(4,1), cal_pt_cloud(5,0)*cal_pt_cloud(5,1), cal_pt_cloud(7,0)*cal_pt_cloud(7,1), cal_pt_cloud(8,0)*cal_pt_cloud(8,1),
            1, 1, 1, 1;
            T = S*C.inverse();
            MatrixXd xyC(6,1);
            xyC << xyE[0]*xyE[0],
            xyE[1]*xyE[1],
            xyE[0],
            xyE[1],
            xyE[0]*xyE[1],
            1;
            MatrixXd tmp = T*xyC;
            xyS.push_back(tmp(0,0));
            xyS.push_back(tmp(1,0));
        }
    }
    return xyS;
}

int main()
{
    int n = 7;
    float widthPupil = 320.0;
    float heightPupil = 240.0;
    float widthScreen = 1024.0;
    float heightScreen = 464.0;
    
    MatrixXd cloudRead = readMatrix("/Users/willard/codes/cpp/openCVison/Hello-Eigen/Hello-Eigen/foo2.txt");
    //removeRow(cloudRead, cloudRead.rows()-1);
    std::cout << cloudRead << std::endl << std::endl;
    MatrixXd cal_pt_cloud = selctGoodPoints(cloudRead);
    std::cout << cal_pt_cloud << std::endl << std::endl;
    
    // 四个中心
    MatrixXd Centers(4, 4);
    // C1
    Centers(0, 0) = (cal_pt_cloud(0,0) + cal_pt_cloud(1,0) + cal_pt_cloud(3,0) + cal_pt_cloud(4,0))/4.0;
    Centers(0, 1) = (cal_pt_cloud(0,1) + cal_pt_cloud(1,1) + cal_pt_cloud(3,1) + cal_pt_cloud(4,1))/4.0;
    Centers(0, 2) = (cal_pt_cloud(0,2) + cal_pt_cloud(1,2) + cal_pt_cloud(3,2) + cal_pt_cloud(4,2))/4.0;
    Centers(0, 3) = (cal_pt_cloud(0,3) + cal_pt_cloud(1,3) + cal_pt_cloud(3,3) + cal_pt_cloud(4,3))/4.0;
    
    // C2
    Centers(1, 0) = (cal_pt_cloud(1,0) + cal_pt_cloud(2,0) + cal_pt_cloud(4,0) + cal_pt_cloud(5,0))/4.0;
    Centers(1, 1) = (cal_pt_cloud(1,1) + cal_pt_cloud(2,1) + cal_pt_cloud(4,1) + cal_pt_cloud(5,1))/4.0;
    Centers(1, 2) = (cal_pt_cloud(1,2) + cal_pt_cloud(2,2) + cal_pt_cloud(4,2) + cal_pt_cloud(5,2))/4.0;
    Centers(1, 3) = (cal_pt_cloud(1,3) + cal_pt_cloud(2,3) + cal_pt_cloud(4,3) + cal_pt_cloud(5,3))/4.0;
    
    // C3
    Centers(2, 0) = (cal_pt_cloud(3,0) + cal_pt_cloud(4,0) + cal_pt_cloud(6,0) + cal_pt_cloud(7,0))/4.0;
    Centers(2, 1) = (cal_pt_cloud(3,1) + cal_pt_cloud(4,1) + cal_pt_cloud(6,1) + cal_pt_cloud(7,1))/4.0;
    Centers(2, 2) = (cal_pt_cloud(3,2) + cal_pt_cloud(4,2) + cal_pt_cloud(6,2) + cal_pt_cloud(7,2))/4.0;
    Centers(2, 3) = (cal_pt_cloud(3,3) + cal_pt_cloud(4,3) + cal_pt_cloud(6,3) + cal_pt_cloud(7,3))/4.0;
    
    // C4
    Centers(3, 0) = (cal_pt_cloud(4,0) + cal_pt_cloud(5,0) + cal_pt_cloud(7,0) + cal_pt_cloud(8,0))/4.0;
    Centers(3, 1) = (cal_pt_cloud(4,1) + cal_pt_cloud(5,1) + cal_pt_cloud(7,1) + cal_pt_cloud(8,1))/4.0;
    Centers(3, 2) = (cal_pt_cloud(4,2) + cal_pt_cloud(5,2) + cal_pt_cloud(7,2) + cal_pt_cloud(8,2))/4.0;
    Centers(3, 3) = (cal_pt_cloud(4,3) + cal_pt_cloud(5,3) + cal_pt_cloud(7,3) + cal_pt_cloud(8,3))/4.0;
    
    std::cout << Centers << std::endl << std::endl;

    cal_pt_cloud = normalize(cal_pt_cloud, widthPupil, heightPupil, widthScreen, heightScreen); // 归一化9个点
    
    std::cout << cal_pt_cloud << std::endl << std::endl;
    
    Centers = normalize(Centers, widthPupil, heightPupil, widthScreen, heightScreen); // 归一化4个中心点
    
    std::cout << Centers << std::endl << std::endl;
    
    vector<MatrixXd> cx_cy_errx_erry = fit_poly_surface(cal_pt_cloud, n);
    MatrixXd cx = cx_cy_errx_erry[0];
    MatrixXd cy = cx_cy_errx_erry[1];
    MatrixXd err_x = cx_cy_errx_erry[2];
    MatrixXd err_y = cx_cy_errx_erry[3];
    
    
    std::ofstream file("test.txt");
    if (file.is_open())
    {
        file << "Here is the matrix m:\n" << cx << '\n';
        file << "Here is the matrix m:\n" << cy << '\n';
        file.close();
    }
    
    float xTE = 154.818/widthPupil;
    float yTE = 74/heightPupil;
    vector<float> tScrn = make_map_function_fn(cx, cy, n, xTE, yTE);
    MatrixXd errs = fit_error_screen(err_x, err_y, widthScreen, heightScreen);
    
    tScrn = denormalize(tScrn, widthScreen, heightScreen);
    //std::cout << "x: " << (int)tScrn[0] << " y: " << (int)tScrn[1] << endl;
    
    
    vector<float> xyE;
    xyE.push_back(xTE);
    xyE.push_back(yTE);
    vector<float> xyS;
    
    //二次非线性模型
    xyS = nolinear2Model(cal_pt_cloud, xyE);
    float xS1 = xyS[0]*widthScreen;
    float yS1 = xyS[1]*heightScreen;
    cout << xS1 << "," << yS1 << endl;
    
    // 第一列竖区域
    std::cout << "眼睛" << xyE[0] << ", " << xyE[1] << std::endl << std::endl;
    if(xyE[0] >= Centers(0,0)){
        if(xyE[1] <= Centers(0,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(0);
            //MatrixXd ES2 = Centers.row(0);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(0,0), cal_pt_cloud(0,1), cal_pt_cloud(0,2), cal_pt_cloud(0,3);
            ES2 << Centers(0,0), Centers(0,1), Centers(0,2), Centers(0,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] > Centers(0,1) && xyE[1] < cal_pt_cloud(3,1)){
            //MatrixXd ES1 = Centers.row(0);
            //MatrixXd ES2 = cal_pt_cloud.row(3);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(0,0), Centers(0,1), Centers(0,2), Centers(0,3);
            ES1 << cal_pt_cloud(3,0), cal_pt_cloud(3,1), cal_pt_cloud(3,2), cal_pt_cloud(3,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= cal_pt_cloud(3,1) && xyE[1] < Centers(0,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(3);
            //MatrixXd ES2 = Centers.row(2);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(3,0), cal_pt_cloud(3,1), cal_pt_cloud(3,2), cal_pt_cloud(3,3);
            ES2 << Centers(2,0), Centers(2,1), Centers(2,2), Centers(2,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(2,1)){
            //MatrixXd ES1 = Centers.row(2);
            //MatrixXd ES2 = cal_pt_cloud.row(6);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(2,0), Centers(2,1), Centers(2,2), Centers(2,3);
            ES2 << cal_pt_cloud(6, 0), cal_pt_cloud(6, 1), cal_pt_cloud(6, 2), cal_pt_cloud(6, 3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }
    }
    // 第二列竖区域
    if(xyE[0] <= Centers(0,0) && xyE[0] > cal_pt_cloud(1,0)){
        if( xyE[1] < Centers(0,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(1);
            //MatrixXd ES2 = Centers.row(0);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(1,0), cal_pt_cloud(1,1), cal_pt_cloud(1,2), cal_pt_cloud(1,3);
            ES2 << Centers(0,0), Centers(0,1), Centers(0,2), Centers(0,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(0,1) && xyE[1] < cal_pt_cloud(3,1)){
            //MatrixXd ES1 = Centers.row(1);
            //MatrixXd ES2 = cal_pt_cloud.row(4);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(1, 0), Centers(1, 1), Centers(1, 2), Centers(1, 3);
            ES2 << cal_pt_cloud(4,0), cal_pt_cloud(4,1), cal_pt_cloud(4,2), cal_pt_cloud(4,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= cal_pt_cloud(3,1) && xyE[1] < Centers(0,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(4);
            //MatrixXd ES2 = Centers.row(2);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(4, 0), cal_pt_cloud(4, 1), cal_pt_cloud(4, 2), cal_pt_cloud(4, 3);
            ES2 << Centers(2, 0), Centers(2, 1), Centers(2, 2), Centers(2, 3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(2,1)){
            //MatrixXd ES1 = Centers.row(2);
            //MatrixXd ES2 = cal_pt_cloud.row(7);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(2,0), Centers(2,1), Centers(2,2), Centers(2,3);
            ES2 << cal_pt_cloud(7,0),cal_pt_cloud(7,1), cal_pt_cloud(7,2), cal_pt_cloud(7,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }
    }
    // 第三列竖区域
    if(xyE[0] <= cal_pt_cloud(1,0) && xyE[0] > Centers(1,0)){
        if( xyE[1] < Centers(1,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(1);
            //MatrixXd ES2 = Centers.row(1);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(1,0), cal_pt_cloud(1,1), cal_pt_cloud(1,2), cal_pt_cloud(1,3);
            ES2 << Centers(1,0), Centers(1,1), Centers(1,2), Centers(1,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(1,1) && xyE[1] < cal_pt_cloud(4,1)){
            //MatrixXd ES1 = Centers.row(1);
            //MatrixXd ES2 = cal_pt_cloud.row(4);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(1,0),Centers(1,1),Centers(1,2),Centers(1,3);
            ES2 << cal_pt_cloud(4,0),cal_pt_cloud(4,1),cal_pt_cloud(4,2),cal_pt_cloud(4,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= cal_pt_cloud(4,1) && xyE[1] < Centers(3,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(4);
            //MatrixXd ES2 = Centers.row(3);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(4,0),cal_pt_cloud(4,1), cal_pt_cloud(4,2), cal_pt_cloud(4,3);
            ES2 << Centers(3,0),Centers(3,1),Centers(3,2),Centers(3,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(3,1)){
            //MatrixXd ES1 = Centers.row(3);
            //MatrixXd ES2 = cal_pt_cloud.row(7);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(3,0),Centers(3,1),Centers(3,2),Centers(3,3);
            ES2 << cal_pt_cloud(7,0),cal_pt_cloud(7,1),cal_pt_cloud(7,2),cal_pt_cloud(7,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }
    }
    // 第四列竖区域
    if(xyE[0] <= Centers(1,0)){
        if( xyE[1] < Centers(1,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(2);
            //MatrixXd ES2 = Centers.row(1);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(2,0),cal_pt_cloud(2,1),cal_pt_cloud(2,2),cal_pt_cloud(2,3);
            ES2 << Centers(1,0),Centers(1,1),Centers(1,2),Centers(1,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(1,1) && xyE[1] < cal_pt_cloud(5,1)){
            //MatrixXd ES1 = Centers.row(1);
            //MatrixXd ES2 = cal_pt_cloud.row(5);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(1,0),Centers(1,1),Centers(1,2),Centers(1,3);
            ES2 << cal_pt_cloud(5,0),cal_pt_cloud(5,1),cal_pt_cloud(5,2),cal_pt_cloud(5,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= cal_pt_cloud(5,1) && xyE[1] < Centers(3,1)){
            //MatrixXd ES1 = cal_pt_cloud.row(5);
            //MatrixXd ES2 = Centers.row(3);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << cal_pt_cloud(5,0),cal_pt_cloud(5,1),cal_pt_cloud(5,2),cal_pt_cloud(5,3);
            ES2 << Centers(3,0),Centers(3,1),Centers(3,2),Centers(3,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }else if(xyE[1] >= Centers(3,1)){
            //MatrixXd ES1 = Centers.row(3);
            //MatrixXd ES2 = cal_pt_cloud.row(8);
            MatrixXd ES1(1,4);
            MatrixXd ES2(1,4);
            ES1 << Centers(3,0),Centers(3,1),Centers(3,2),Centers(3,3);
            ES2 << cal_pt_cloud(8,0),cal_pt_cloud(8,1),cal_pt_cloud(8,2),cal_pt_cloud(8,3);
            xyS = linearModel_lDown_rTop(ES1, ES2, xyE);
        }
    }
    
    //finish
    
    float xS = xyS[0]*widthScreen;
    float yS = xyS[1]*heightScreen;
    cout << xS << "," << yS << endl;
    
    //剔除不合要求的点
    int threshold = 55; //设置阈值
    int inlinersCount = 0;
    for(int i = 0; i < cal_pt_cloud.rows(); i++){
        if(errs(i, 0) <= threshold){
            ++inlinersCount;
        }
    }
    
    MatrixXd inliners(inlinersCount, 4);
    int inlinersIndex = 0;
    for(int j = 0; j < cal_pt_cloud.rows(); j++){
        if(errs(j, 0) <= threshold){
            for(int i = 0; i < 4; i++){
                inliners(inlinersIndex, i) = cal_pt_cloud(j, i);
            }
            ++inlinersIndex;
        }
    }
    
    vector<MatrixXd> new_cx_cy_errx_erry = fit_poly_surface(inliners, n);
    MatrixXd new_cx = new_cx_cy_errx_erry[0];
    MatrixXd new_cy = new_cx_cy_errx_erry[1];
    MatrixXd new_err_x = new_cx_cy_errx_erry[2];
    MatrixXd new_err_y = new_cx_cy_errx_erry[3];
    vector<float> new_tScrn = make_map_function_fn(new_cx, new_cy, n, xTE, yTE);
    MatrixXd new_errs = fit_error_screen(new_err_x, new_err_y, widthScreen, heightScreen);
    
    new_tScrn = denormalize(new_tScrn, widthScreen, heightScreen);
    
    std::cout << "inliners:=======" << inliners.rows() <<"*" << inliners.cols() << "===============" << endl;
    std::cout << "x: " << (int)new_tScrn[0] << " y: " << (int)new_tScrn[1] << endl;
}
