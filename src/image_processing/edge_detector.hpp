#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace EdgeDetector {
void sobel_detector(Mat &src, Mat &dst);
void canny_detector(Mat &src, Mat &dst);
} // namespace EdgeDetector
} // namespace ImageProcessing