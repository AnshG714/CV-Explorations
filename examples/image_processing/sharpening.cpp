#include <opencv2/opencv.hpp>
#include <src/image_processing/sharpening.hpp>
#include <src/utils/load_resource.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  Mat img = load_image_path("flower.jpg");
  Mat img2 = img.clone();
  ImageProcessing::Sharpening::sharpen(img, img2, 7, 1.1);
  cv::imshow("original", img);
  cv::imshow("Image sharpened", img2);
  cv::waitKey(0);
}