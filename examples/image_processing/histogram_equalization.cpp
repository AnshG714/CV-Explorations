#include <opencv2/opencv.hpp>
#include <src/image_processing/histogram_equalization.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("flower.jpg");
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::imshow("Initial Image", img);
  Mat img2 = img.clone();
  ImageProcessing::HistogramEqualization::histogram_equalize(img, img2);
  cv::namedWindow("Histogram Equalized image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Histogram Equalized image", img2);
  cv::waitKey(0);
}