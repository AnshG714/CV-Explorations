#include "metrics.hpp"
#include <opencv2/opencv.hpp>

namespace Metrics {

Mat calcHist(Mat &img, bool normalized) {
  // we consider grayscale images.
  int channels[] = {0};
  const int histogramDims = 1;
  int histSize = 256;
  float range[] = {0, 256};
  const float *histRange[] = {range};
  bool uniform = true, accumulate = false;

  Mat hist;
  calcHist(&img, 1, channels, Mat(), hist, 1, &histSize, histRange, uniform,
           accumulate);

  if (normalized) {
    hist /= sum(hist);
  }
  return hist;
}

vector<float> vec_from_hist(Mat &histogram) {
  // this will only work for '1D' histograms!
  vector<float> ret;
  for (int i = 0; i < 255; i++) {
    ret.push_back(histogram.at<float>(i));
  }
  return ret;
}

} // namespace Metrics