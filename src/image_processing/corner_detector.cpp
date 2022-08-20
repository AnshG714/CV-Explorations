#include "corner_detector.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace CornerDetector {

// TODO: Extract these kernels to a common class.
Mat _construct_sobel_kernel_x() {
  return (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
}

Mat _construct_sobel_kernel_y() {
  return (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
}

void hessian_detector(Mat &src, Mat &dst, float threshold) {
  Mat S_x, S_y;
  Mat S_xx, S_yy, S_xy;
  filter2D(src, S_x, -1, _construct_sobel_kernel_x());
  filter2D(src, S_y, -1, _construct_sobel_kernel_y());
  filter2D(S_x, S_xx, -1, _construct_sobel_kernel_x());
  filter2D(S_y, S_yy, -1, _construct_sobel_kernel_y());
  filter2D(S_y, S_xy, -1, _construct_sobel_kernel_x());
  cvtColor(dst, dst, COLOR_GRAY2BGR);
  // std::pair<int, int> corners;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      Mat hessian = (Mat_<float>(2, 2) << S_xx.ptr(i)[j], S_xy.ptr(i)[j],
                     S_xy.ptr(i)[j], S_yy.ptr(i)[j]);
      Mat eigenvalues;
      eigen(hessian, eigenvalues);
      if (eigenvalues.at<float>(0) > threshold &&
          eigenvalues.at<float>(1) > threshold) {
        circle(dst, Point(i, j), 5, Scalar(0, 0, 255));
      }
    }
  }
}

} // namespace CornerDetector
} // namespace ImageProcessing