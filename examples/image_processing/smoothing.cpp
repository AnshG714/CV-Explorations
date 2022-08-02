#include <opencv2/opencv.hpp>
#include <src/image_processing/smoothing.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;

int main(int argc, char **argv) {
  Mat img = load_image_path("grainy_potrait.jpg");
  cv::namedWindow("Initial Image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Initial Image", img);
  // Mat img2 = img.clone();
  // smooth_image(img, img2, 10);
  cvtColor(img, img, COLOR_BGR2GRAY);
  Mat img2 = _apply_median_filter(img, 7);
  cv::namedWindow("Smoothed image", cv::WINDOW_AUTOSIZE);
  cv::imshow("Smoothed image", img2);
  cv::waitKey(0);
}
