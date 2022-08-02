#include <opencv2/opencv.hpp>

using namespace cv;

enum SmoothingTypes { MEAN = 0, MEDIAN = 1 };

Mat smooth_image(Mat &src, Mat &dst, int kernel_size,
                 const int smooth_type = SmoothingTypes::MEAN);
Mat _apply_median_filter(Mat &img, int kernel_size);
