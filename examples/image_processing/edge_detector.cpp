#include <opencv2/opencv.hpp>
#include <src/image_processing/edge_detector.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("small-icon.jpg");
  cvtColor(img, img, COLOR_BGR2GRAY);
  ImageProcessing::EdgeDetector::sobel_detector(img, img);
  imshow("Edges Detected [sobel]", img);
  waitKey(0);
}