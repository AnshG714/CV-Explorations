#include <opencv2/opencv.hpp>
#include <src/image_processing/edge_detector.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("potrait.jpg");
  Mat img2 = img.clone();
  cvtColor(img, img, COLOR_BGR2GRAY);
  ImageProcessing::EdgeDetector::sobel_detector(img, img);
  imshow("Edges Detected [sobel]", img);

  cvtColor(img2, img2, COLOR_BGR2GRAY);
  ImageProcessing::EdgeDetector::canny_detector(img2, img2);
  // cv::Canny(img2, img2, 120, 240);
  imshow("Edges Detected [canny]", img2);
  waitKey(0);
}