/*
  This file shall not be used as a judgement of my coding skills. By reading
  further, you agree to this.
*/

#include "src/utils/load_resource.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("small-icon.jpg");
  cout << img.channels() << endl;
  cvtColor(img, img, COLOR_BGR2GRAY);
  img.convertTo(img, CV_32FC1);

  Mat S_x, S_y;
  Mat kern_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
  Mat kern_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
  filter2D(img, S_x, -1, kern_x);
  filter2D(img, S_y, -1, kern_y);

  // compute L2 Norm and direction vector
  Mat l2_norm, directions;
  magnitude(S_x, S_y, l2_norm);

  double max;
  minMaxIdx(l2_norm, nullptr, &max);
  cout << max << endl;
  l2_norm = (l2_norm / max) * 255;
  phase(S_x, S_y, directions, true);
  cout << (5 <= 7 <= 10) << endl;
  return 0;
}