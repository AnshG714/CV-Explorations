#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace Sharpening {
void sharpen(Mat &src, Mat &dst, int kernel_size, float lambda);
} // namespace Sharpening
} // namespace ImageProcessing