/*
  This file shall not be used as a judgement of my coding skills. By reading
  further, you agree to this.
*/

#include "src/utils/load_resource.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat test = (Mat_<float>(2, 2) << 1.2, 1.9, 4.5, 3.4);
  Mat eigenvalues;
  eigen(test, eigenvalues);
  cout << trace(test) << endl;
  return 0;
}