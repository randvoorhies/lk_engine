#include "lk_engine.h"
#include <iostream>

int main(int argc, char** argv)
{
  cv::VideoCapture cap(0);
  cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  rcv::LKEngine lk;

  while(true)
  {
    cv::Mat image;
    cap >> image;

    lk.updateImage(image);

    cv::imshow("image", image);

    cv::waitKey(50);
  }
  return 0;
}
