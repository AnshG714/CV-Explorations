#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace CornerDetector {
void hessian_detector(Mat &src, Mat &dst, float threshold);
} // namespace CornerDetector
} // namespace ImageProcessing