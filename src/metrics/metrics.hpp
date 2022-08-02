#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace Metrics {
Mat calcHist(Mat &img, bool normalized = false);
vector<float> vec_from_hist(Mat &histogram);
} // namespace Metrics