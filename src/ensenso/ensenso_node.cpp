#include <algorithm>
#include <memory>

#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <bin_picking/OrthographicImage.h>
#include <bin_picking/GetOrthographicImages.h>
#include <ensenso/config.hpp>
#include <ensenso/ensenso.hpp>


class EnsensoNode {
  std::unique_ptr<Ensenso> camera;
  std::vector<std::string> default_suffixes {{"ed"}};

  template<class T>
  bool inVector(T element, const std::vector<T>& vector) {
    return std::find(vector.begin(), vector.end(), element) != vector.end();
  }

  bin_picking::OrthographicImage convert(const cv::Mat& image, EnsensoConfig config, const std::string& encoding, const std::string& suffix) {
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

    bool take_ed = inVector<std::string>("ed", req.camera_suffixes);
    bool take_er = inVector<std::string>("er", req.camera_suffixes);

    if (take_ed && take_er) {
      std::pair<cv::Mat, cv::Mat> images;

      int i {0};
      do {
        images = camera->takeImages();
        i += 1;
      } while (camera->checkDepthShadowOverfill(images.second) && i <= 8);

      ROS_WARN_COND(i > 1, "Needed to retake image %d times.", i);

      res.images.push_back(convert(images.second, camera->depth_capture_config, "mono16", "ed"));
      res.images.push_back(convert(images.first, camera->raw_capture_config, "mono16", "er"));

    } else if (take_ed) {
      cv::Mat image;

      int i {0};
      do {
        image = camera->takeDepthImage();
        i += 1;
      } while (camera->checkDepthShadowOverfill(image) && i <= 8);

      ROS_WARN_COND(i > 1, "Needed to retake image %d times.", i);

      res.images.push_back(convert(image, camera->depth_capture_config, "mono16", "ed"));

    } else if (take_er) {
      const std::pair<cv::Mat, cv::Mat> images = camera->takeImages();
      res.images.push_back(convert(images.first, camera->raw_capture_config, "mono16", "er"));
    }

    return true;
  }


public:
  EnsensoNode(EnsensoConfig config) {
    camera = std::make_unique<Ensenso>(config);

    ros::NodeHandle node_handle;
    auto s = node_handle.advertiseService("ensenso/images", &EnsensoNode::getImages, this);
    ros::spin();
  }
};


int main(int argc, char *argv[]) {
  ros::init(argc, argv, "ensenso_node");

  EnsensoConfig config;
  EnsensoNode node {config};

  return 0;
}
