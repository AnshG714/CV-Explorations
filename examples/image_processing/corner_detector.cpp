#include <opencv2/opencv.hpp>
#include <src/image_processing/corner_detector.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("board.jpg");
  cvtColor(img, img, COLOR_BGR2GRAY);
  // ImageProcessing::CornerDetector::hessian_detector(img, img, 253.);
  // cornerHarris(img, img, 2, 3, 0.04);
  ImageProcessing::CornerDetector::harris_detector(img, img, 1);
  imshow("Corners Detected [Hessian]", img);
  waitKey(0);
}