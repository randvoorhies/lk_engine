#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <boost/shared_ptr.hpp>

namespace rcv
{
  // ######################################################################
  class LKEngine
  {
    public:
      LKEngine();
      void updateImage(cv::Mat const image);
      void track(std::vector<cv::Point2f> const & prev_points,
          std::vector<cv::Point2f> & curr_points, std::vector<char> & status, std::vector<float> & error);
      void trackAndFilter(std::vector<cv::Point2f> & prev_points, std::vector<cv::Point2f> & curr_points);

    private:
      void allocate();

      cv::Mat curr_image_;
      cv::Mat prev_image_;
      boost::shared_ptr<CvArr> prev_pyramid_;
      boost::shared_ptr<CvArr> curr_pyramid_;
      CvSize win_size_;
      int level_;
      int flags_;
  };
}

// ######################################################################
rcv::LKEngine::LKEngine() :
  win_size_(cvSize(15,15)),
  level_(1),
  flags_(0)
{ }

// ######################################################################
void rcv::LKEngine::updateImage(cv::Mat const image)
{
  prev_image_ = curr_image_;

  if(image.type() == CV_8UC1)
    curr_image_ = image;
  else if(image.type() == CV_8UC3)
    cv::cvtColor(image, curr_image_, CV_RGB2GRAY);

  if(!prev_pyramid_ || !curr_pyramid_)
  {
    allocate();
    curr_image_.copyTo(prev_image_);
  }

  std::swap(prev_pyramid_, curr_pyramid_);
  if(flags_ & CV_LKFLOW_PYR_B_READY)
    flags_ |= CV_LKFLOW_PYR_A_READY;

  flags_ &= ~CV_LKFLOW_PYR_B_READY;
}

// ######################################################################
void rcv::LKEngine::allocate()
{
  prev_pyramid_ = boost::shared_ptr<CvArr>(cvCreateImage(curr_image_.size(), IPL_DEPTH_8U, 1), cvFree_);
  curr_pyramid_ = boost::shared_ptr<CvArr>(cvCreateImage(curr_image_.size(), IPL_DEPTH_8U, 1), cvFree_);
}

// ######################################################################
void rcv::LKEngine::track(std::vector<cv::Point2f> const & prev_points,
    std::vector<cv::Point2f> & curr_points, std::vector<char> & status, std::vector<float> & error)
{
  curr_points.resize(prev_points.size());
  status.resize(prev_points.size());
  error.resize(prev_points.size());

  CvTermCriteria term_criteria
    = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 );
  
  IplImage prev_ipl = prev_image_;
  IplImage curr_ipl = curr_image_;

  CvPoint2D32f const * prev_points_arr = reinterpret_cast<CvPoint2D32f const *>(&prev_points[0]);
  CvPoint2D32f * curr_points_arr       = reinterpret_cast<CvPoint2D32f*>(&curr_points[0]);

  cvCalcOpticalFlowPyrLK(&prev_ipl, &curr_ipl, prev_pyramid_.get(), curr_pyramid_.get(),
      prev_points_arr, curr_points_arr, prev_points.size(),
      win_size_, level_, &status[0], &error[0], term_criteria, flags_);

  flags_ |= CV_LKFLOW_PYR_A_READY;
  flags_ |= CV_LKFLOW_PYR_B_READY;
}

// ######################################################################
void rcv::LKEngine::trackAndFilter(std::vector<cv::Point2f> & prev_points, std::vector<cv::Point2f> & curr_points)
{
  std::vector<char> status;
  std::vector<float> error;
  track(prev_points, curr_points, status, error);

  std::vector<cv::Point2f> prev_points_filt;
  std::vector<cv::Point2f> curr_points_filt;
  for(size_t i=0; i<prev_points.size(); ++i)
  {
    if(status[i])
    {
      prev_points_filt.push_back(prev_points[i]);
      curr_points_filt.push_back(curr_points[i]);
    }
  }
  prev_points = std::move(prev_points_filt);
  curr_points = std::move(curr_points_filt);
}

