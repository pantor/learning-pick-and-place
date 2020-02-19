#pragma once

#define BIN_PICKING_GEOMETRY

#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Geometry>
#include <unsupported/Eigen/EulerAngles>


struct Affine {
private:
  using Euler = Eigen::EulerAngles<double, Eigen::EulerSystemZYX>;

  Eigen::Vector3d get_angles() const {
    Eigen::Vector3d angles = Euler::FromRotation<false, false, false>(data.rotation()).angles();
    Eigen::Vector3d angles_equal;
    angles_equal << angles[0] - M_PI, M_PI - angles[1], angles[2] - M_PI;

    if (angles_equal[1] > M_PI) {
      angles_equal[1] -= 2 * M_PI;
    }
    if (angles_equal[2] < -M_PI) {
      angles_equal[2] += 2 * M_PI;
    }

    if (angles.norm() < angles_equal.norm()) {
      return angles;
    }
    return angles_equal;
  }


public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  Eigen::Affine3d data;

  Affine() {
    this->data = Eigen::Affine3d::Identity();
  }

  Affine(const Eigen::Affine3d& data) {
    this->data = data;
  }

  Affine(double x, double y, double z, double a, double b, double c) {
    data = Eigen::Translation<double, 3>(x, y, z) * Euler(a, b, c).toRotationMatrix();
  }

  Affine(const std::array<double, 6>& v): Affine(v[0], v[1], v[2], v[3], v[4], v[5]) { }

  Affine(const Eigen::Matrix<double, 6, 1>& v): Affine(v(0), v(1), v(2), v(3), v(4), v(5)) { }

  Affine(const geometry_msgs::Pose& pose) {
    auto position = pose.position;
    auto orientation = pose.orientation;

    Eigen::Translation<double, 3> t {position.x, position.y, position.z};
    Eigen::Quaternion<double> q {orientation.w, orientation.x, orientation.y, orientation.z};

    data = t * q;
  }

  Affine operator *(const Affine &a) const {
    Eigen::Affine3d result;
    result = data * a.data;
    return Affine(result);
  }

  Affine inverse() const {
    return Affine(data.inverse());
  }

  bool isApprox(const Affine &a) const {
    return data.isApprox(a.data);
  }

  Eigen::Ref<Eigen::Affine3d::MatrixType> matrix() {
    return data.matrix();
  }

  void translate(const Eigen::Vector3d &v) {
    data.translate(v);
  }

  void pretranslate(const Eigen::Vector3d &v) {
    data.pretranslate(v);
  }

  Eigen::Vector3d translation() const {
    Eigen::Vector3d v;
    v << data.translation();
    return v;
  }

  double x() const {
    return data.translation()(0);
  }

  void set_x(double x) {
    data.translation()(0) = x;
  }

  double y() const {
    return data.translation()(1);
  }

  void set_y(double y) {
    data.translation()(1) = y;
  }

  double z() const {
    return data.translation()(2);
  }

  void set_z(double z) {
    data.translation()(2) = z;
  }

  void rotate(const Eigen::Affine3d::LinearMatrixType &r) {
    data.rotate(r);
  }

  void prerotate(const Eigen::Affine3d::LinearMatrixType &r) {
    data.prerotate(r);
  }

  Eigen::Affine3d::LinearMatrixType rotation() const {
    Eigen::Affine3d::LinearMatrixType result;
    result << data.rotation();
    return result;
  }

  double a() const {
    return get_angles()(0);
  }

  double b() const {
    return get_angles()(1);
  }

  double c() const {
    return get_angles()(2);
  }

  void set_a(double a) {
    Eigen::Matrix<double, 3, 1> angles;
    angles << get_angles();
    data = Eigen::Translation<double, 3>(data.translation()) * Euler(a, angles(1), angles(2)).toRotationMatrix();
  }

  void set_b(double b) {
    Eigen::Matrix<double, 3, 1> angles;
    angles << get_angles();
    data = Eigen::Translation<double, 3>(data.translation()) * Euler(angles(0), b, angles(2)).toRotationMatrix();
  }

  void set_c(double c) {
    Eigen::Matrix<double, 3, 1> angles;
    angles << get_angles();
    data = Eigen::Translation<double, 3>(data.translation()) * Euler(angles(0), angles(1), c).toRotationMatrix();
  }

  geometry_msgs::Pose toPose() const {
    Eigen::Quaternion<double> q;
    q = data.rotation();
    auto t = data.translation();

    geometry_msgs::Pose result;
    result.position.x = t(0);
    result.position.y = t(1);
    result.position.z = t(2);
    result.orientation.x = q.x();
    result.orientation.y = q.y();
    result.orientation.z = q.z();
    result.orientation.w = q.w();
    return result;
  }

  std::array<double, 6> toArray() const {
    Eigen::Matrix<double, 6, 1> v;
    v << data.translation(), get_angles();
    return {v(0), v(1), v(2), v(3), v(4), v(5)};
  }

  Affine getInnerRandom() const {
    std::random_device r;
    std::default_random_engine engine(r());

    Eigen::Matrix<double, 6, 1> random;
    Eigen::Matrix<double, 6, 1> max;
    max << data.translation(), get_angles();

    for (int i = 0; i < 6; i++) {
      std::uniform_real_distribution<double> distribution(-max(i), max(i));
      random(i) = distribution(engine);
    }

    return Affine(random);
  }

  std::string toString() const {
    Eigen::Matrix<double, 6, 1> v;
    v << data.translation(), get_angles();

    return "[" + std::to_string(v(0)) + ", " + std::to_string(v(1)) + ", " + std::to_string(v(2))
      + ", " + std::to_string(v(3)) + ", " + std::to_string(v(4)) + ", " + std::to_string(v(5)) + "]";
  }
};
