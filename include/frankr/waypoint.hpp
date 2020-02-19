#pragma once

#include <frankr/geometry.hpp>


struct Waypoint {
  enum class ReferenceType {
    ABSOLUTE,
    RELATIVE
  };

  Affine target_affine;

  ReferenceType reference_type {ReferenceType::ABSOLUTE};

  Waypoint(Affine target_affine): target_affine(target_affine) { }
  Waypoint(Affine target_affine, ReferenceType reference_type): target_affine(target_affine), reference_type(reference_type) { }
};
