#include "edge_detector.hpp"
#include "smoothing.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace EdgeDetector {

Mat _construct_sobel_kernel_x() {
  return (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
}

Mat _construct_sobel_kernel_y() {
  return (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
}

void sobel_detector(Mat &src, Mat &dst) {
  Mat S_x, S_y;
  src.convertTo(src, CV_32FC1);
  filter2D(src, S_x, -1, _construct_sobel_kernel_x());
  filter2D(src, S_y, -1, _construct_sobel_kernel_y());
  addWeighted(S_x, 0.5, S_y, 0.5, 5, dst);
  dst.convertTo(dst, CV_8UC1);
}

/*
  Applies nonomax suppression to the gradient array.
  relation between directional values and orientation of edge:
  - 0 <= direction < 22.5 OR 157.5 <= direction < 202.5 OR 337.5 <= direction <=
  360
    ====> HORIZONTAL
  - 22.5 < direction <= 67.5 OR 202.5 < direction <= 247.5
    ====> SECONDARY DIAGONAL
  - 67.5 < direction <= 112.5 OR 247.5 < direction <= 292.5
    ====> VERTICAL
  - 112.5 < direction <= 157.5 OR 292.5 < direction <= 337.5
    ====> PRIMARY DIAGONAL
  */
void _apply_non_maxima_suppression(Mat &gradients, Mat &directions) {
  Mat z = Mat::zeros(gradients.rows, gradients.cols, CV_32FC1);
  for (int i = 1; i < gradients.rows - 1; ++i) {
    float32_t *grad_row = gradients.ptr<float32_t>(i);
    float32_t *dir_row = directions.ptr<float32_t>(i);
    float32_t *z_row = z.ptr<float32_t>(i);
    for (int j = 1; j < gradients.cols - 1; ++j) {
      float32_t gradient = grad_row[j];
      float32_t direction = dir_row[j];
      float32_t prev, next;
      if (0 <= direction < 22.5 || 157.5 <= direction < 202.5 ||
          direction > 337.5) {
        // Horizontal
        prev = gradients.ptr<float32_t>(i)[j - 1];
        next = gradients.ptr<float32_t>(i)[j + 1];
      } else if (22.5 < direction <= 67.5 || 202.5 < direction <= 247.5) {
        // Secondary diagonal / 45° line
        prev = gradients.ptr<float32_t>(i + 1)[j - 1];
        next = gradients.ptr<float32_t>(i - 1)[j + 1];
      } else if (67.5 < direction <= 112.5 || 247.5 < direction <= 292.5) {
        // Vertical
        prev = gradients.ptr<float32_t>(i - 1)[j];
        next = gradients.ptr<float32_t>(i + 1)[j];
      } else {
        // Secondary diagonal / 135° line
        prev = gradients.ptr<float32_t>(i - 1)[j - 1];
        next = gradients.ptr<float32_t>(i + 1)[j + 1];
      }

      if (gradient > prev && gradient > next) {
        z_row[j] = gradient;
      }
    }
  }
  gradients = std::move(z);
}

void canny_detector(Mat &src, Mat &dst) {
  // first, smoothen
  ImageProcessing::Smoothing::GaussianBlur(src, dst, 3, 0, 0, 1.5);

  // use Sobel operator for getting S_x and S_y
  Mat S_x, S_y;
  dst.convertTo(dst, CV_32FC1);
  filter2D(src, S_x, -1, _construct_sobel_kernel_x());
  filter2D(src, S_y, -1, _construct_sobel_kernel_y());

  // compute L2 Norm and direction vector
  Mat gradients, directions;
  magnitude(S_x, S_y, gradients);
  double max;
  minMaxIdx(gradients, nullptr, &max);
  gradients = (gradients / max) * 255;
  phase(S_x, S_y, directions, true);

  // Apply non-maxima suppression
  _apply_non_maxima_suppression(gradients, directions);
}

} // namespace EdgeDetector
} // namespace ImageProcessing