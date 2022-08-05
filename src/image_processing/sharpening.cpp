#include "sharpening.hpp"
#include "smoothing.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace Sharpening {
void sharpen(Mat &src, Mat &dst, int kernel_size, float lambda) {
  Mat img2 = src.clone();
  ImageProcessing::Smoothing::BoxBlur(img2, img2, kernel_size);
  img2 = ((1 + lambda) * src - img2);
  img2.copyTo(dst);
}
} // namespace Sharpening
} // namespace ImageProcessing