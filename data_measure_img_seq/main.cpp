#include <cmath>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using cv::Mat;
using std::cout;
using std::endl;
using std::string;
using std::vector;

vector<float> compute_temporal_measure(const string &filename,
                                       float (*func)(Mat *)) {
  cv::VideoCapture capture(filename);
  Mat frame;
  vector<float> data;

  if (!capture.isOpened()) {
    throw "Error when reading steam_avi";
  }

  for (;;) {
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    data.push_back(func(&frame));
  }
  return data;
}

float compute_mean_of_frame(Mat *frame_ptr) {
  Mat frame = *frame_ptr;
  int sum_of_values = cv::sum(frame)[0];
  return (float)(sum_of_values / (frame.rows * frame.cols));
}

float compute_variance_of_frame(Mat *frame_ptr) {
  Mat frame = *frame_ptr;
  cv::MatExpr squared_values = frame.mul(frame);
  int squared_values_sum = cv::sum(squared_values)[0];
  return (float)(squared_values_sum - pow(compute_mean_of_frame(frame_ptr), 2));
}

float compute_contrast_for_frame(Mat *frame_ptr) {
  Mat frame = *frame_ptr;
  int cn = frame.channels();
  cv::Scalar_<uint8_t> current_pixel;
  float contrast_agg = 0;

  for (int i = 0; i < frame.rows; i++) {
    uint8_t *row_ptr = (uint8_t *)frame.row(i).data;
    for (int j = 0; j < frame.cols; j++) {
      uint8_t val0 = row_ptr[j * cn + 0];
      uint8_t val1 = row_ptr[j * cn + 1];
      uint8_t val2 = row_ptr[j * cn + 2];
      float intensity = (float)((val0 + val1 + val2) / 3);

      int adjacent_sum = 0;
      int total_divide = 0;

      // top pixel
      if (i > 0) {
        uint8_t *top_row_ptr = (uint8_t *)frame.row(i - 1).data;
        adjacent_sum += top_row_ptr[j * cn + 0] + top_row_ptr[j * cn + 1] +
                        top_row_ptr[j * cn + 2];
        total_divide += 3;
      }

      // bottom pixel
      if (i < frame.rows - 1) {
        uint8_t *bottom_row_ptr = (uint8_t *)frame.row(i + 1).data;
        adjacent_sum += bottom_row_ptr[j * cn + 0] +
                        bottom_row_ptr[j * cn + 1] + bottom_row_ptr[j * cn + 2];
        total_divide += 3;
      }

      // left pixel
      if (j > 0) {
        adjacent_sum += row_ptr[(j - 1) * cn + 0] + row_ptr[(j - 1) * cn + 1] +
                        row_ptr[(j - 1) * cn + 2];
        total_divide += 3;
      }

      // right pixel
      if (j < frame.cols - 1) {
        adjacent_sum += row_ptr[(j + 1) * cn + 0] + row_ptr[(j + 1) * cn + 1] +
                        row_ptr[(j + 1) * cn + 2];
        total_divide += 3;
      }

      float avg_intensity_surrounding = (float)(adjacent_sum / total_divide);
      float diff = cv::abs(avg_intensity_surrounding - intensity);
      contrast_agg += diff;
    }
  }

  return (float)(contrast_agg / (frame.rows * frame.cols));
}

vector<float> get_normalized_values(vector<float> &function_values) {
  cv::Scalar mean_f_scalar, stddev_f_scalar;
  cv::meanStdDev(function_values, mean_f_scalar, stddev_f_scalar);
  float mean_f = mean_f_scalar[0], std_f = stddev_f_scalar[0];

  vector<float> ret;
  for (auto i : function_values) {
    ret.push_back((i - mean_f) / std_f);
  }

  return ret;
}

int main(int argc, char **argv) {
  const string filename = "/Users/anshgodha/Desktop/image-data-cv/"
                          "data_measure_img_seq/afreightc.avi";
  auto means = compute_temporal_measure(filename, compute_mean_of_frame);
  auto variances =
      compute_temporal_measure(filename, compute_variance_of_frame);
  auto contrasts =
      compute_temporal_measure(filename, compute_contrast_for_frame);

  auto normalized_means = get_normalized_values(means);
  auto normalized_variances = get_normalized_values(variances);
  auto normalized_contrasts = get_normalized_values(contrasts);

  // uncomment the plots below to see non-normalized plots. Comment out the
  // normalized plots.
  // plt::plot(means, {{"label", "means"}});
  // plt::plot(variances, {{"label", "variances"}});
  // plt::plot(contrasts, {{"label", "contrasts"}});

  plt::plot(normalized_means, {{"label", "normalized means"}});
  plt::plot(normalized_variances, {{"label", "normalized variances"}});
  plt::plot(normalized_contrasts, {{"label", "normalized contrasts"}});

  plt::legend();
  plt::show();

  return 0;
}