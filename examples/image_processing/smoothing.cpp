#include <opencv2/opencv.hpp>
#include <src/image_processing/smoothing.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  // Mat img = load_image_path("flower.jpg");
  // Mat img2 = img.clone();
  // Mat img3 = img.clone();
  // ImageProcessing::Smoothing::BoxBlur(img, img2, 7);
  // ImageProcessing::Smoothing::GaussianBlur(img, img3, 15, 8, 8, 10);
  // cv::imshow("Image smoothing, original", img);
  // cv::imshow("Image smoothing, box", img2);
  // cv::imshow("Image smoothing, gaussian", img3);

  Mat img = load_image_path("grainy_potrait.jpg");
  Mat img2 = img.clone();
  cvtColor(img, img, COLOR_BGR2GRAY);
  cvtColor(img2, img2, COLOR_BGR2GRAY);
  ImageProcessing::Smoothing::MedianBlur(img, img2, 7);
  cv::imshow("Original", img);
  cv::imshow("Smoothed image", img2);
  cv::waitKey(0);
}
