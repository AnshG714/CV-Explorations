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

void harris_detector(Mat &src, Mat &dst, float sigma, float alpha) {
  Mat L_x, L_y, L_x_2, L_y_2, L_x_y;
  filter2D(dst, L_x, -1, _construct_sobel_kernel_x());
  filter2D(dst, L_y, -1, _construct_sobel_kernel_y());

  pow(L_x, 2, L_x_2);
  pow(L_y, 2, L_y_2);
  L_x_y = L_x.mul(L_y);

  GaussianBlur(L_x_2, L_x_2, Size(3, 3), 1, 1);
  GaussianBlur(L_y_2, L_y_2, Size(3, 3), 1, 1);
  GaussianBlur(L_x_y, L_x_y, Size(3, 3), 1, 1);

  // default window size is 3.
  cvtColor(dst, dst, COLOR_GRAY2BGR);
  Mat cornerness_matrix(src.rows, src.cols, CV_32F);

  for (int i = 3; i < dst.rows - 3; i++) {
    for (int j = 3; j < dst.cols - 3; j++) {
      Rect roi(i - 1, j - 1, 3, 3);
      float S_x2 = sum(L_x_2(roi))[0];
      float S_y2 = sum(L_y_2(roi))[0];
      float S_xy = sum(L_x_y(roi))[0];

      Mat harris_matrix = (Mat_<float>(2, 2) << S_x2, S_xy, S_xy, S_y2);
      float cornerness =
          determinant(harris_matrix) - alpha * trace(harris_matrix)[0];

      // if (cornerness > 600000) {
      //   circle(dst, Point(i, j), 5, Scalar(0, 0, 255));
      // }
      cornerness_matrix.at<float>(j, i) = cornerness;
    }
  }

  normalize(cornerness_matrix, cornerness_matrix, 1, 0, NormTypes::NORM_MINMAX);
  for (int i = 3; i < src.rows - 3; i++) {
    for (int j = 3; j < src.cols - 3; j++) {
      if (cornerness_matrix.ptr<float>(i)[j] > 0.7) {
        circle(dst, Point(i, j), 5, Scalar(0, 0, 255));
      }
    }
  }
}

} // namespace CornerDetector
} // namespace ImageProcessing