#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <frankr/gripper.hpp>
#include <frankr/robot.hpp>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal


PYBIND11_MODULE(cfrankr, m) {
  py::class_<Affine>(m, "Affine")
    .def(py::init<double, double, double, double, double, double>(), "x"_a=0.0, "y"_a=0.0, "z"_a=0.0, "a"_a=0.0, "b"_a=0.0, "c"_a=0.0)
    .def(py::init<std::array<double, 6>>())
    .def(py::init<Eigen::Matrix<double, 6, 1>>())
    .def(py::self * py::self)
    .def("matrix", &Affine::matrix)
    .def("inverse", &Affine::inverse)
    .def("is_approx", &Affine::isApprox)
    .def("translate", &Affine::translate)
    .def("pretranslate", &Affine::pretranslate)
    .def("translation", &Affine::translation)
    .def_property("x", &Affine::x, &Affine::set_x)
    .def_property("y", &Affine::y, &Affine::set_y)
    .def_property("z", &Affine::z, &Affine::set_z)
    .def("rotate", &Affine::rotate)
    .def("prerotate", &Affine::prerotate)
    .def("rotation", &Affine::rotation)
    .def_property("a", &Affine::a, &Affine::set_a)
    .def_property("b", &Affine::b, &Affine::set_b)
    .def_property("c", &Affine::c, &Affine::set_c)
    .def("to_array", &Affine::toArray)
    .def("get_inner_random", &Affine::getInnerRandom)
    .def("__repr__", &Affine::toString);

  py::class_<MotionData>(m, "MotionData")
    .def(py::init<>())
    .def_readwrite("velocity_rel", &MotionData::velocity_rel)
    .def_readwrite("acceleration_rel", &MotionData::acceleration_rel)
    .def_readwrite("did_break", &MotionData::did_break)
    .def("with_dynamics", &MotionData::withDynamics)
    .def("with_z_force_condition", &MotionData::withZForceCondition)
    .def("with_xy_force_condition", &MotionData::withXYForceCondition);

  py::class_<Waypoint> waypoint(m, "Waypoint");
  waypoint.def(py::init<Affine>())
    .def(py::init<Affine, Waypoint::ReferenceType>())
    .def_readwrite("target_affine", &Waypoint::target_affine)
    .def_readwrite("reference_type", &Waypoint::reference_type);

  py::enum_<Waypoint::ReferenceType>(waypoint, "ReferenceType")
    .value("ABSOLUTE", Waypoint::ReferenceType::ABSOLUTE)
    .value("RELATIVE", Waypoint::ReferenceType::RELATIVE)
    .export_values();

  py::class_<Gripper>(m, "Gripper")
    .def(py::init<const std::string&>())
    .def(py::init<const std::string&, double, double>())
    .def_readwrite("gripper_force", &Gripper::gripper_force)
    .def_readwrite("gripper_speed", &Gripper::gripper_speed)
    .def_readonly("max_width", &Gripper::max_width)
    .def("width", &Gripper::width)
    .def("stop", &Gripper::stop)
    .def("homing", &Gripper::homing)
    .def("is_grasping", &Gripper::isGrasping)
    .def("move", &Gripper::move)
    .def("open", &Gripper::open)
    .def("clamp", &Gripper::clamp)
    .def("release", &Gripper::release);

  py::class_<Robot>(m, "Robot")
    .def(py::init<const std::string &>())
    .def(py::init<const std::string &, double>())
    .def_readwrite("velocity_rel", &Robot::velocity_rel)
    .def_readwrite("acceleration_rel", &Robot::acceleration_rel)
    .def_readwrite("acceleration_rel", &Robot::acceleration_rel)
    .def_readonly("is_moving", &Robot::is_moving)
    .def_readonly("has_reflex_error", &Robot::has_reflex_error)
    .def("current_pose", &Robot::currentPose)
    .def("recover_from_errors", &Robot::recoverFromErrors)
    .def("move_joints", &Robot::moveJoints)
    .def("move_ptp", &Robot::movePtp)
    .def("move_relative_ptp", &Robot::moveRelativePtp)
    .def("move_waypoints_ptp", &Robot::moveWaypointsPtp)
    .def("move_cartesian", &Robot::moveCartesian)
    .def("move_relative_cartesian", &Robot::moveRelativeCartesian)
    .def("move_waypoints_cartesian", &Robot::moveWaypointsCartesian);
}
