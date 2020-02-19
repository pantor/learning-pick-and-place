#include <iostream>
#include <memory>

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <bin_picking/OrthographicImage.h>
#include <bin_picking/GetOrthographicImages.h>
#include <realsense/config.hpp>
#include <realsense/realsense.hpp>


class RealsenseNode {
  std::unique_ptr<Realsense> camera;
  std::vector<std::string> default_suffixes {"rd", "rc"};

  template<class T>
  bool inVector(T element, const std::vector<T>& vector) {
    return std::find(vector.begin(), vector.end(), element) != vector.end();
  }

  bin_picking::OrthographicImage convert(const cv::Mat& image, RealsenseConfig config, const std::string& encoding, const std::string& suffix) {
    const sensor_msgs::ImagePtr image_ptr = cv_bridge::CvImage(std_msgs::Header(), encoding, image).toImageMsg();
    bin_picking::OrthographicImage result;
    result.image = *image_ptr;
    result.pixel_size = config.pixel_size;
    result.min_depth = config.min_depth;
    result.max_depth = config.max_depth;
    result.camera = suffix;
    return result;
  }

  bool getImages(bin_picking::GetOrthographicImages::Request &req, bin_picking::GetOrthographicImages::Response &res) {
    if (req.camera_suffixes.empty()) {
      req.camera_suffixes = default_suffixes;
    }

    bool take_rd = inVector<std::string>("rd", req.camera_suffixes);
    bool take_rc = inVector<std::string>("rc", req.camera_suffixes);

    if (take_rd && take_rc) {
      const std::pair<cv::Mat, cv::Mat> images = camera->takeImages();
      res.images.push_back(convert(images.first, camera->config, "mono16", "rd"));
      res.images.push_back(convert(images.second, camera->config, "rgb8", "rc"));

    } else if (take_rd) {
      cv::Mat image = camera->takeDepthImage();
      res.images.push_back(convert(image, camera->config, "mono16", "rd"));

    } else if (take_rc) {
      const std::pair<cv::Mat, cv::Mat> images = camera->takeImages();
      res.images.push_back(convert(images.second, camera->config, "rgb8", "rc"));
    }

    return true;
  }

public:
  RealsenseNode(RealsenseConfig config) {
    camera = std::make_unique<Realsense>(config);

    ros::NodeHandle node_handle;
    auto s = node_handle.advertiseService("realsense/images", &RealsenseNode::getImages, this);
    ros::spin();
  }
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "realsense_node");

  window app(752, 480, "");

  RealsenseConfig config;
  RealsenseNode node(config);

  return 0;
}
