#include "LKEngine.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv)
{
  cv::VideoCapture cap(0);
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  rcv::LKEngine lk;
    
  std::vector<cv::Point2f> prev_points;

  while(true)
  {
    cv::Mat image;
    cap >> image;

    cv::Mat image_bw;
    cv::cvtColor(image, image_bw, CV_RGB2GRAY);

    cv::goodFeaturesToTrack(image_bw, prev_points, 50, .01, 15);

    lk.updateImage(image_bw);

    std::vector<cv::Point2f> curr_points;
    std::vector<char> status;
    std::vector<float> error;
    CvSize win_size = cvSize(15,15);

    lk.trackAndFilter(prev_points, curr_points);

    for(size_t i=0; i<prev_points.size(); ++i)
    {
      cv::circle(image, prev_points[i], 5, cv::Scalar(0), 3);
      cv::circle(image, curr_points[i], 5, cv::Scalar(128), 3);
      cv::line(image, prev_points[i], curr_points[i], cv::Scalar(128), 1);
    }

    cv::imshow("image", image);

    cv::waitKey(50);
  }
  return 0;
}
