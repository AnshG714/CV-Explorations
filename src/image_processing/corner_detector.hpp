#include <opencv2/opencv.hpp>

using namespace cv;

namespace ImageProcessing {
namespace CornerDetector {
void hessian_detector(Mat &src, Mat &dst, float threshold);
void harris_detector(Mat &src, Mat &dst, float sigma, float alpha = (1 / 25.));

} // namespace CornerDetector
} // namespace ImageProcessing