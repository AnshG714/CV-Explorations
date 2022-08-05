#include "smoothing.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>
#include <src/metrics/metrics.hpp>

using namespace cv;
using namespace std;

namespace ImageProcessing {
namespace Smoothing {
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
void MedianBlur(Mat &src, Mat &dst, int kernel_size) {
  assert(kernel_size % 2 == 1);
  int kernel_radius = (kernel_size - 1) / 2;
  Mat padded_img(src.rows + 2 * kernel_radius, src.cols + 2 * kernel_radius,
                 CV_8UC1);

  copyMakeBorder(src, padded_img, kernel_radius, kernel_radius, kernel_radius,
                 kernel_radius, BORDER_REPLICATE);

  Mat initialRoi, hist;
  for (int i = kernel_radius; i < src.rows; i++) {
    for (int j = kernel_radius; j < src.cols; j++) {
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
      dst.data[(i - kernel_radius) * src.rows + (j - kernel_radius)] =
          _median_value_from_histogram(hist, kernel_size);
    }
  }
}

/*
smooths image through the use of the box filter, which replaces each
pixel with its window mean.
*/
Mat _construct_box_filter(int k) {
  int side_size = 2 * k + 1;
  return Mat::ones(side_size, side_size, CV_64F) / (side_size * side_size);
}

void BoxBlur(Mat &src, Mat &dst, int kernel_size) {
  filter2D(src, dst, -1, _construct_box_filter(kernel_size));
}

Mat _construct_gaussian_filter(int ksize, double mean_x, double mean_y,
                               double sigma) {
  assert(ksize % 2 == 1 && ksize > 1);
  Mat ret = Mat(ksize, ksize, CV_32F);
  int origin_ind = ksize / 2;
  float accum_sum = 0;
  for (int i = 0; i < ksize; i++) {
    float32_t *row = ret.ptr<float32_t>(i);
    for (int j = 0; j < ksize; j++) {
      int relative_x = j - origin_ind;
      int relative_y = i - origin_ind;
      float exp_numerator =
          pow(relative_x - mean_x, 2) + pow(relative_y - mean_y, 2);
      float exp_denominator = 2 * pow(sigma, 2);
      float gaussian_coeff =
          exp(-(exp_numerator / exp_denominator)) / (2 * M_PI * pow(sigma, 2));
      accum_sum += gaussian_coeff;
      row[j] = gaussian_coeff;
    }
  }
  ret /= accum_sum;
  return ret;
}

void GaussianBlur(Mat &src, Mat &dst, int kernel_size, float mean_x,
                  float mean_y, float sigma) {
  filter2D(src, dst, -1,
           _construct_gaussian_filter(kernel_size, mean_x, mean_y, sigma));
}

} // namespace Smoothing
} // namespace ImageProcessing