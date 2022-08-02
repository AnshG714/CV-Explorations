#include "smoothing.hpp"
#include <opencv2/opencv.hpp>
#include <src/metrics/metrics.hpp>

using namespace cv;
using namespace std;

uint8_t _median_value_from_histogram(Mat &histogram, int kernel_size) {
  const int total_readings = kernel_size * kernel_size;
  int acc = 0, ind = 0;
  const int total_bins = 255;
  vector<float> histogram_vec = Metrics::vec_from_hist(histogram);

  while (ind < total_bins) {
    if (acc + histogram_vec[ind] > (total_readings + 1) / 2) {
      return ind;
    }
    ind++;
    acc += histogram_vec[ind];
  }

  return 255;
}

/**
 * Smooth image through the use of the median filter.
 */
Mat _apply_median_filter(Mat &img, int kernel_size) {
  assert(kernel_size % 2 == 1);
  int kernel_radius = (kernel_size - 1) / 2;
  Mat padded_img(img.rows + 2 * kernel_radius, img.cols + 2 * kernel_radius,
                 CV_8UC1);
  Mat ret_img(img.rows, img.cols, CV_8UC1);

  copyMakeBorder(img, padded_img, kernel_radius, kernel_radius, kernel_radius,
                 kernel_radius, BORDER_REPLICATE);

  Mat initialRoi, hist;
  for (int i = kernel_radius; i < img.rows; i++) {
    for (int j = kernel_radius; j < img.cols; j++) {
      if (j == kernel_radius) {
        initialRoi = padded_img(Rect(0, 0, kernel_size, kernel_size));
        hist = Metrics::calcHist(initialRoi);
      } else {
        for (int k = -kernel_radius; k <= kernel_radius; k++) {
          uchar *kernel_row = padded_img.row(i + k).data;
          hist.at<float>((int)kernel_row[j - kernel_radius - 1]) -= 1;
          hist.at<float>((int)kernel_row[j + kernel_radius]) += 1;
        }
      }
      ret_img.data[(i - kernel_radius) * img.rows + (j - kernel_radius)] =
          _median_value_from_histogram(hist, kernel_size);
    }
  }

  return ret_img;
}

/*
smooths image through the use of the box filter, which replaces each
pixel with its window mean.
*/
Mat _construct_box_filter(int k) {
  int side_size = 2 * k + 1;
  return Mat::ones(side_size, side_size, CV_64F) / (side_size * side_size);
}

Mat _apply_mean_filter(Mat &src, Mat &dst, int kernel_size) {
  filter2D(src, dst, -1, _construct_box_filter(kernel_size));
  return dst;
}

Mat smooth_image(Mat &src, Mat &dst, int kernel_size, const int smooth_type) {
  switch (smooth_type) {
  case SmoothingTypes::MEAN:
    return _apply_mean_filter(src, dst, kernel_size);
  case SmoothingTypes::MEDIAN:
    return _apply_median_filter(src, kernel_size);
  default:
    return _apply_mean_filter(src, dst, kernel_size);
  }
}