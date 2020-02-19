#include <frankr/robot.hpp>


Robot::Robot(const std::string &name): moveit::planning_interface::MoveGroupInterface(name) {
  sub_wrench = node_handle.subscribe("franka_state_controller/F_ext", 10, &Robot::wrenchCallback, this);
  sub_states = node_handle.subscribe("franka_state_controller/franka_states", 10, &Robot::stateCallback, this);

  spinner.start();
}

Robot::Robot(const std::string &name, double dynamic_rel): Robot(name) {
  velocity_rel = dynamic_rel;
  acceleration_rel = dynamic_rel;
}

void Robot::stateCallback(const franka_msgs::FrankaState& msg) {
  if (msg.current_errors.cartesian_reflex) {
    has_reflex_error = true;
    ROS_INFO_THROTTLE(1.0, "Cartesian reflex error!");
  }
}

void Robot::wrenchCallback(const geometry_msgs::WrenchStamped& msg) {
  if (is_moving && !current_motion_data->did_break) {
    if (current_motion_data->check_z_force_condition && (std::pow(msg.wrench.force.z, 2) > std::pow(current_motion_data->max_z_force, 2))) {
      ROS_INFO("Exceeded z force.");
      this->stop();
      current_motion_data->did_break = true;
      this->is_moving = false;

      this->flag = true;
      wait_condition_variable.notify_all();
    }

    if (current_motion_data->check_xy_force_condition && (std::pow(msg.wrench.force.x, 2) + std::pow(msg.wrench.force.y, 2) > std::pow(current_motion_data->max_xy_force, 2))) {
      ROS_INFO("Exceeded xy force.");
      this->stop();
      current_motion_data->did_break = true;
      this->is_moving = false;

      this->flag = true;
      wait_condition_variable.notify_all();
    }
  }
}

Affine Robot::currentPose(const Affine& frame) {
  return restartMoveItIfCommandFails([&]() { return Affine(this->getCurrentPose().pose) * frame.inverse(); }, 5); // [s]
}

int Robot::restartMoveIt() {
  // Set /move_group node to respawn=true
  ROS_WARN("Restart MoveIt!");
  // system("rosnode kill /move_group"); // Soft kill by ROS
  int result = system("kill $(ps ax | grep [/]move_group | awk '{print $1}')");
  std::this_thread::sleep_for(std::chrono::seconds(4));
  ROS_INFO("Restarted MoveIt.");
  return result;
}

bool Robot::recoverFromErrors() {
  ac.waitForServer();

  franka_control::ErrorRecoveryGoal goal;
  ac.sendGoal(goal);

  bool success = ac.waitForResult(ros::Duration(5.0));
  if (success) {
    has_reflex_error = false;
  }
  return success;
}

bool Robot::moveJoints(const std::array<double, 7>& joint_values, MotionData& data) {
  current_motion_data = std::make_shared<MotionData>(data);
  this->setMaxVelocityScalingFactor(velocity_rel * data.velocity_rel);
  this->setMaxAccelerationScalingFactor(acceleration_rel * data.acceleration_rel);

  std::vector<double> joint_values_vector {joint_values.begin(), joint_values.end()};
  this->setJointValueTarget(joint_values_vector);

  bool execution_success = false;
  if (this->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
    this->is_moving = true;
    auto execution = this->execute(my_plan);
    this->is_moving = false;
    execution_success = (execution == moveit::planning_interface::MoveItErrorCode::SUCCESS);
  } else {
    ROS_FATAL_STREAM("Error in planning motion");
    return false;
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;

  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}

bool Robot::movePtp(const Affine& frame, const Affine& affine, MotionData& data) {
  return moveWaypointsPtp(frame, { Waypoint(affine) }, data);
}

bool Robot::moveRelativePtp(const Affine& frame, const Affine& affine, MotionData& data) {
  return moveWaypointsPtp(frame, { Waypoint(affine, Waypoint::ReferenceType::RELATIVE) }, data);
}

bool Robot::moveWaypointsPtp(const Affine& frame, const std::vector<Waypoint>& waypoints, MotionData& data) {
  EigenSTL::vector_Affine3d affines {};
  for (auto waypoint: waypoints) {
    if (waypoint.reference_type == Waypoint::ReferenceType::ABSOLUTE) {
      affines.push_back(waypoint.target_affine.data * frame.data);
    } else {
      Eigen::Affine3d base_affine = affines.empty() ? Affine(this->getCurrentPose().pose).data * frame.data : affines.back() * frame.data;

      base_affine.translate(waypoint.target_affine.data.translation());
      base_affine = base_affine * frame.data.inverse();
      base_affine.rotate(waypoint.target_affine.data.rotation());

      affines.push_back(base_affine);
    }
  }

  current_motion_data = std::make_shared<MotionData>(data);
  this->setMaxVelocityScalingFactor(velocity_rel * data.velocity_rel);
  this->setMaxAccelerationScalingFactor(acceleration_rel * data.acceleration_rel);

  bool execution_success {false};
  for (auto affine: affines) {
    this->setPoseTarget(affine);

    if (this->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS) {
      this->stop();
      this->is_moving = true;
      auto execution = this->execute(my_plan);
      this->is_moving = false;
      execution_success = (execution == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    } else {
      ROS_FATAL_STREAM("Error in planning motion");
      return false;
    }
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;

  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}

bool Robot::moveCartesian(const Affine& frame, const Affine& affine, MotionData& data) {
  return moveWaypointsCartesian(frame, { Waypoint(affine) }, data);
}

bool Robot::moveRelativeCartesian(const Affine& frame, const Affine& affine, MotionData& data) {
  return moveWaypointsCartesian(frame, { Waypoint(affine, Waypoint::ReferenceType::RELATIVE) }, data);
}

bool Robot::moveWaypointsCartesian(const Affine& frame, const std::vector<Waypoint>& waypoints, MotionData& data) {
  geometry_msgs::Pose start_pose = this->getCurrentPose().pose;

  std::vector<geometry_msgs::Pose> poses;
  poses.push_back(start_pose);

  for (auto waypoint: waypoints) {
    if (waypoint.reference_type == Waypoint::ReferenceType::ABSOLUTE) {
      poses.push_back(Affine(waypoint.target_affine.data * frame.data).toPose());
    } else {
      Eigen::Affine3d base_affine = Affine(poses.back()).data * frame.data;

      base_affine.translate(waypoint.target_affine.data.translation());
      base_affine = base_affine * frame.data.inverse();
      base_affine.rotate(waypoint.target_affine.data.rotation());

      poses.push_back(Affine(base_affine).toPose());
    }
  }

  current_motion_data = std::make_shared<MotionData>(data);

  moveit_msgs::RobotTrajectory trajectory;
  bool execution_success {false};

  double fraction = restartMoveItIfCommandFails([&]() { return this->computeCartesianPath(poses, 0.005, 0.0, trajectory); }, 10); // [s]

  if (fraction >= 0.0) {
    robot_trajectory::RobotTrajectory rt(this->getCurrentState()->getRobotModel(), this->getName());
    rt.setRobotTrajectoryMsg(*this->getCurrentState(), trajectory);

    // Scaling because of MoveIt cannot use absolute velocity and acceleration scaling, so this is a hacky solution...
    const double cartesian_scaling = 1.0 + (0.48 - poses.back().position.x) * 3.2;

    trajectory_processing::IterativeParabolicTimeParameterization iptp;
    iptp.computeTimeStamps(rt, velocity_rel * data.velocity_rel * cartesian_scaling, acceleration_rel * data.acceleration_rel * cartesian_scaling);

    rt.getRobotTrajectoryMsg(trajectory);
    my_plan.trajectory_ = trajectory;

    this->stop();
    this->is_moving = true;

    auto execution = restartMoveItIfCommandFails([&]() { return this->execute(my_plan); }, 15); // [s]

    this->is_moving = false;
    execution_success = (execution == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    if (!execution_success) {
      recoverFromErrors();

      this->is_moving = true;

      execution = restartMoveItIfCommandFails([&]() { return this->execute(my_plan); }, 15); // [s]

      this->is_moving = false;
    }
  } else {
    ROS_FATAL_STREAM("Error in planning motion");
    return false;
  }

  if (current_motion_data->break_callback && current_motion_data->did_break) {
    current_motion_data->break_callback();
  }

  data = *current_motion_data;
  current_motion_data = std::make_shared<MotionData>();
  return execution_success;
}
