#include "histogram_equalization.hpp"
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

vector<float> compute_relative_frequences(Mat &src) {
  // we consider grayscale images.
  const int channels = 0;
  const int histogramDims = 1;
  const int histSize = 256;
  float range[] = {0, 256};
  const float *histRange[] = {range};
  bool uniform = true, accumulate = false;

  Mat hist;
  calcHist(&src, 1, &channels, Mat(), hist, 1, &histSize, histRange, uniform,
           accumulate);
  hist /= sum(hist);
  vector<float> cumsum;
  float current_sum = 0;
  for (int i = 0; i < hist.rows; i++) {
    current_sum += (float)hist.at<float>(i);
    cumsum.push_back(current_sum);
  }

  return cumsum;
}

void histogram_equalize(Mat &src, Mat &dest) {
  // will need to change this for multi-channel images.
  vector<float> relative_frequencies = compute_relative_frequences(src);
  for (int i = 0; i < src.rows; i++) {
    uint8_t *row = (uint8_t *)src.row(i).data;
    for (int j = 0; j < src.cols; j++) {
      uint8_t intensity = row[j];
      uint8_t new_intensity = relative_frequencies[intensity] * 255;
      dest.data[i * src.cols + j] = new_intensity;
    }
  }
}