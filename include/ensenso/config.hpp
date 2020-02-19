#pragma once

struct EnsensoConfig {
  std::string id {"182472"};
  double pixel_size {2000.0};
  double min_depth {0.22};
  double max_depth {0.41};

  // Default settings
  bool auto_exposure {true};
  double exposure_time {0}; // [ms]
  int target_brightness {200};

  bool auto_gain {true};
  double gain {0};

  bool projector {true};
  bool front_light {false};

  std::string stereo_matching_method {"SgmAligned"};

  bool use_open_gl {true};
  bool use_cuda {true};
  int cuda_device {0};
};
