#include "lk_engine.h"
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

    cv::cvtColor(image, image, CV_RGB2GRAY);

    auto features_start = std::chrono::monotonic_clock::now();
    cv::goodFeaturesToTrack(image, prev_points, 50, .01, 15);
    std::chrono::microseconds features_time =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::monotonic_clock::now() - features_start);
    std::cout << "Features took " << features_time.count() / 1000.0 << "ms" << std::endl;

    auto update_start = std::chrono::monotonic_clock::now();
    lk.updateImage(image);
    std::chrono::microseconds update_time =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::monotonic_clock::now() - update_start);
    std::cout << "Update took " << update_time.count() / 1000.0 << "ms" << std::endl;



    std::vector<cv::Point2f> curr_points;
    std::vector<char> status;
    std::vector<float> error;
    CvSize win_size = cvSize(15,15);

    auto track_start = std::chrono::monotonic_clock::now();
    lk.track(prev_points, curr_points, status, error, win_size);
    std::chrono::microseconds track_time =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::monotonic_clock::now() - track_start);
    std::cout << "Track took " << track_time.count() / 1000.0 << "ms" << std::endl;

    cv::imshow("image", image);

    cv::waitKey(50);
  }
  return 0;
}
