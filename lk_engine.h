#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/shared_ptr.hpp>

namespace rcv
{
  class LKEngine
  {
    public:
      LKEngine();
      void updateImage(cv::Mat const image);
      void trackPoints(std::vector<cv::Point2f> const & prev_points, CvSize win_size);

    private:
      void allocate();

      cv::Mat curr_image_;
      cv::Mat prev_image_;
      boost::shared_ptr<CvArr> prev_pyramid_;
      boost::shared_ptr<CvArr> curr_pyramid_;
      int level_;

      int flags_;
  };
}

// ######################################################################
rcv::LKEngine::LKEngine() :
  level_(1),
  flags_(0)
{
}

// ######################################################################
void rcv::LKEngine::updateImage(cv::Mat const image)
{
  if(!prev_pyramid_ || !curr_pyramid_) allocate();
}

// ######################################################################
void rcv::LKEngine::allocate()
{
  prev_pyramid_ = boost::shared_ptr<CvArr>(cvCreateImage(curr_image_.size(), IPL_DEPTH_8U, 1), cvFree_);
  curr_pyramid_ = boost::shared_ptr<CvArr>(cvCreateImage(curr_image_.size(), IPL_DEPTH_8U, 1), cvFree_);
}

// ######################################################################
void rcv::LKEngine::trackPoints(std::vector<cv::Point2f> const & prev_points, CvSize win_size)
{
  std::vector<cv::Point2f> curr_points(prev_points.size());
  std::vector<char> status(prev_points.size());
  std::vector<float> error(prev_points.size());

  CvTermCriteria term_criteria
    = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 );
  
  IplImage prev_ipl = prev_image_;
  IplImage curr_ipl = curr_image_;

  CvPoint2D32f const * prev_points_arr = reinterpret_cast<CvPoint2D32f const *>(&prev_points[0]);
  CvPoint2D32f * curr_points_arr = reinterpret_cast<CvPoint2D32f*>(&curr_points[0]);

  cvCalcOpticalFlowPyrLK(&prev_ipl, &curr_ipl, prev_pyramid_.get(), curr_pyramid_.get(),
      prev_points_arr, curr_points_arr, prev_points.size(),
      win_size, level_, &status[0], &error[0], term_criteria, flags_);

}
