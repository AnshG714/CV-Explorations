// main.cpp

#include <cmath>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <src/image_processing/histogram_equalization.hpp>
#include <vector>

using namespace cv;
using namespace std;
using std::cout;
using std::endl;

void mouseMoveHandler(int event, int x, int y, int flags, void *param) {
  Mat &img = *((Mat *)(param));
  Mat img_copy = img.clone();
  Size img_size = img.size();
  Rect r =
      Rect(Point(max(x - 50, 0), max(y - 50, 0)),
           Point(min(x + 50, img_size.width), min(y + 50, img_size.height)));

  Vec3b pixel = img.at<Vec3b>(y, x);

  double avg_pixel_value = (double)cv::sum(pixel)[0] / 3;
  double mean_window_value = cv::sum(cv::sum(img(r)))[0] / (3 * r.area());
  long double variance_window_value =
      cv::sum(cv::sum(img(r).dot(img(r))))[0] / (3 * r.area()) -
      pow(mean_window_value, 2);
  double std_window_value = sqrt(variance_window_value);
  Mat MeanVec;
  Mat StdVec;
  cv::meanStdDev(img(r), MeanVec, StdVec);
  if (event == EVENT_MOUSEMOVE) {
    rectangle(img_copy, r, Scalar(255, 0, 0), 2, 8, 0);
    cv::imshow("Simple Demo", img_copy);
  }

  if (event == EVENT_LBUTTONDOWN) {
    cout << "The Pixel Intensity is: " << avg_pixel_value << endl;
    cout << "The mean window value is: " << mean_window_value << endl;
    cout << "The standard deviation in this window is: " << std_window_value
         << endl;
    cout << "Mean: " << MeanVec << ", Std" << StdVec << endl;
  }
}

void displayImageHistograms(Mat *img) {
  vector<Mat> bgr_planes;
  split(*img, bgr_planes);

  Mat b_plane = bgr_planes[0], g_plane = bgr_planes[1], r_plane = bgr_planes[2];
  Mat b_hist, g_hist, r_hist;

  int nimages = 1;
  const int *channels = 0;
  const int histogramDims = 1;
  const int histSize = 256;
  float range[] = {0, 256};
  const float *histRange[] = {range};
  bool uniform = true, accumulate = false;

  // compute histogram values for each channel
  calcHist(&b_plane, 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform,
           accumulate);
  calcHist(&g_plane, 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform,
           accumulate);
  calcHist(&r_plane, 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform,
           accumulate);

  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double)hist_w / histSize);
  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

  for (int i = 1; i < histSize; i++) {
    line(histImage,
         Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
         Scalar(255, 0, 0), 2, 8, 0);

    line(histImage,
         Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
         Scalar(0, 255, 0), 2, 8, 0);

    line(histImage,
         Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
         Scalar(0, 0, 255), 2, 8, 0);
  }

  cv::imshow("Histograms!", histImage);
}

int main(int argc, char **argv) {

  if (argc != 2) {
    cout << "Expecting a image file to be passed to program" << endl;
    return -1;
  }

  cv::Mat img = cv::imread(argv[1]);

  if (img.empty()) {
    cout << "Not a valid image file" << endl;
    return -1;
  }

  cv::namedWindow("Simple Demo", cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback("Simple Demo", mouseMoveHandler, &img);
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::imshow("Simple Demo", img);
  displayImageHistograms(&img);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return 0;
}