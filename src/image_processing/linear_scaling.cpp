#include "linear_scaling.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

void linear_scaling(Mat &src, Mat &dest) {
  double u_min;
  double u_max;
  minMaxLoc(src, &u_min, &u_max, 0, 0);
  int a = -u_min;
  int b = 255 / (u_max - u_min);
  for (int i = 0; i < src.rows; i++) {
    uint8_t *row = src.row(i).data;
    for (int j = 0; j < src.cols; j++) {
      uint8_t intensity = row[j];
      uint8_t new_intensity = b * (intensity + a);
      dest.data[i * src.cols + j] = new_intensity;
    }
  }
}