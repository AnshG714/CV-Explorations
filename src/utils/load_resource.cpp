#include <opencv2/opencv.hpp>
#include <string>

using std::string;
using namespace cv;

Mat load_image_path(string s) {
#ifdef RESOURCES_PATH
  return cv::imread(string(RESOURCES_PATH) + string("/") + s);
#else
  throw std::runtime_error("Could not load resource path.");
#endif
}