#pragma once

#include <iostream>
#include <limits>
#include <thread>

#include <opencv2/opencv.hpp>
#include <nxLib.h>

#include <ensenso/config.hpp>


class Ensenso {
  double min_depth; // [m]
  double max_depth; // [m]
  double factor_depth;

  const cv::Size size {752, 480}; // [px]

  const bool repeat_shadow_overfill {true};
  const double repeat_shadow_threshold {0.65};

  NxLibItem root; // Reference to the API tree root
  NxLibItem camera; // Reference to the nxLib camera

  template<typename T, typename U>
  T clampLimits(U value) {
    return std::min<U>(std::max<U>(value, std::numeric_limits<T>::min()), std::numeric_limits<T>::max());
  }

  void configureCapture(const EnsensoConfig config);


public:
  Ensenso(EnsensoConfig config);
  ~Ensenso();

  EnsensoConfig raw_capture_config;
  EnsensoConfig depth_capture_config;

  void configureRawCaptureParams(EnsensoConfig config);
  void configureDepthCaptureParams(EnsensoConfig config);

  bool checkDepthShadowOverfill(const cv::Mat& image) {
    if (!repeat_shadow_overfill) return false;

    const float zero_percentage = 1.0 - ((float)cv::countNonZero(image)) / image.size().area();
    return zero_percentage > repeat_shadow_threshold;
  }

  cv::Mat takeDepthImage();
  std::pair<cv::Mat, cv::Mat> takeImages();
};
