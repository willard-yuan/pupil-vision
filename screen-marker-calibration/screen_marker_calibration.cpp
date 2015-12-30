//
//  main.cpp
//  Hello-Eigen
//
//  Created by willard on 12/30/15.
//  Copyright © 2015 wilard. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

#define MAXBUFSIZE  ((int) 1e6)

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
    
    rows--;
    
    // Populate matrix with numbers.
    MatrixXd result(rows,cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i,j) = buff[ cols*i+j ];
    
    return result;
};

MatrixXd make_model(MatrixXd cal_pt_cloud,int n=7){
    long int pointsNum = cal_pt_cloud.rows();
    MatrixXd X = cal_pt_cloud.col(0);
    MatrixXd Y = cal_pt_cloud.col(1);
    MatrixXd XX = X.cwiseProduct(X);
    MatrixXd YY = Y.cwiseProduct(Y);
    MatrixXd XY = X.cwiseProduct(Y);
    MatrixXd XXYY = XX.cwiseProduct(YY);
    MatrixXd Ones = MatrixXd::Ones(pointsNum,1);
    MatrixXd ZX = cal_pt_cloud.col(2);
    MatrixXd ZY = cal_pt_cloud.col(3);
    MatrixXd M(pointsNum, 9);
    M << X, Y, XX, YY, XY, XXYY, Ones, ZX, ZY;
    return M;
}

int main()
{
    int model_n = 7;
    MatrixXd cal_pt_cloud = readMatrix("/Users/willard/codes/python/pupil-v0.2.0/pupil_src/capture/test.txt");
    
    long int pointsNum = cal_pt_cloud.rows();
    MatrixXd X = cal_pt_cloud.col(0);
    MatrixXd Y = cal_pt_cloud.col(1);
    MatrixXd XX = X.cwiseProduct(X);
    MatrixXd YY = Y.cwiseProduct(Y);
    MatrixXd XY = X.cwiseProduct(Y);
    MatrixXd XXYY = XX.cwiseProduct(YY);
    MatrixXd Ones = MatrixXd::Ones(pointsNum,1);
    MatrixXd ZX = cal_pt_cloud.col(2);
    MatrixXd ZY = cal_pt_cloud.col(3);
    MatrixXd M(pointsNum, 9);
    M << X, Y, XX, YY, XY, XXYY, Ones, ZX, ZY;

    JacobiSVD<MatrixXd> svd(M.block(0, 0, pointsNum, model_n), ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd w = svd.singularValues();
    MatrixXd Vt = svd.matrixV();
    
    std::cout << w << std::endl;
}
