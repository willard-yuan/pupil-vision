//  Created by willard on 12/8/15.
//  Copyright © 2015 wilard. All rights reserved.
//
//  Another method to erase specular can be seen: https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/methods.py#L198

#include "erase_specular.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

using namespace std;

const cv::Mat ERASE_SPEC_KERNEL = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

void erase_specular(cv::Mat& eye_grey) {

    // Rather arbitrary decision on how large a specularity may be
    int max_spec_contour_area = (eye_grey.size().width + eye_grey.size().height)/2;

    cv::GaussianBlur(eye_grey, eye_grey, cv::Size(5, 5), 0);

    // Close to suppress eyelashes
    cv::morphologyEx(eye_grey, eye_grey, cv::MORPH_CLOSE, ERASE_SPEC_KERNEL);

    // Compute thresh value (using of highest and lowest pixel values)
    double m, M; // m(in) and (M)ax values in image
    minMaxLoc(eye_grey, &m, &M, NULL, NULL);
    double thresh = (m + M) * 3/4;

    // Threshold the image
    cv::Mat eye_thresh;
    cv::threshold(eye_grey, eye_thresh, thresh, 255, cv::THRESH_BINARY);

    // Find all contours in threshed image (possible specularities)
    vector< vector<cv::Point> > all_contours, contours;
    cv::findContours(eye_thresh, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    // Only save small ones (assumed to be spec.s)
    for (int i=0; i<all_contours.size(); i++){
        if( contourArea(all_contours[i]) < max_spec_contour_area )
            contours.push_back(all_contours[i]);
    }

    // Draw the contours into an inpaint mask
    cv::Mat small_contours_mask = cv::Mat::zeros(eye_grey.size(), eye_grey.type());
    cv::drawContours(small_contours_mask, contours, -1, 255, -1);
    cv::dilate(small_contours_mask, small_contours_mask, ERASE_SPEC_KERNEL);

    // Inpaint within contour bounds
    cv::inpaint(eye_grey, small_contours_mask, eye_grey, 2, cv::INPAINT_TELEA);
}
