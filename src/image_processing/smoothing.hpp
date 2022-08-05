#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace Smoothing {
void BoxBlur(Mat &src, Mat &dst, int kernel_size);
void MedianBlur(Mat &src, Mat &dst, int kernel_size);
void GaussianBlur(Mat &src, Mat &dst, int kernel_size, float mean_x,
                  float mean_y, float sigma);
} // namespace Smoothing
} // namespace ImageProcessing