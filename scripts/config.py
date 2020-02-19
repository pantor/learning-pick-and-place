from pathlib import Path

import numpy as np

from cfrankr import Affine
from utils.epoch import Epoch
from utils.param import Bin, Mode, SelectionMethod


class Config:
    start_bin = Bin.Left
    mode = Mode.Measure

    # Camera
    camera_suffixes = ('ed',)
    # camera_suffixes = ('ed', 'rd', 'rc')

    collection = 'placing-eval'
    # collection = 'placing-screw-1'

    model_input_suffixes = ('ed',)
    # model_input_suffixes = ('ed', 'rd', 'rc')

    grasp_model = ('cylinder-cube-1', 'model-6-arch-more-layer')
    # grasp_model = ('cylinder-screw-3', 'model-7')

    place_model = 'placing-3-32-part-type-2'
    # place_model = 'placing-3-15-screw-type-2'


    # Epochs
    epochs = [
        Epoch(
            number_episodes=2000,
            selection_method=SelectionMethod.Max,
            percentage_secondary=0.0,
            secondary_selection_method=SelectionMethod.Prob,
        )
    ]

    # General structure
    change_bins = True
    bin_empty_at_max_probability = 0.10
    shift_objects = False
    set_zero_reward = False
    home_gripper = True

    # Images
    take_after_images = False
    take_direct_images = False
    take_lateral_images = False
    take_goal_images = True
    predict_images = False

    # Overview image
    lateral_overview_image = False
    overview_image_angles = [(-0.6, 0.0), (-0.3, 0.0), (0.3, 0.0), (0.6, 0.0)]

    # Lateral images
    # lateral_images_angles = [(b, 0) for b in np.linspace(-0.6, 0.6, 7)]
    lateral_images_angles = [(b, c) for b in np.linspace(-0.6, 0.6, 5) for c in np.linspace(-0.6, 0.6, 5)]

    # Goal images
    use_goal_images = True
    keep_goal_image = True
    goal_images_dataset = 'baby'
    if take_goal_images and not use_goal_images:
        raise Exception('Use goal images needs to be enabled for take goal images')

    # URL
    database_url = 'http://127.0.0.1:8080/api/'

    # Bin
    box = {
        'center': [-0.001, -0.0065, 0.372],  # [m]
        'size': [0.172, 0.281, 0.068],  # [m]
    }

    # Distances
    image_distance_from_pose = 0.350  # [m]
    default_image_pose = Affine(z=-image_distance_from_pose)
    approach_distance_from_pose = 0.120 if mode != Mode.Perform else 0.075  # [m]
    lower_random_pose = [-0.055, -0.105, 0.0, -1.4, 0.0, 0.0]  # [m, rad]
    upper_random_pose = [0.055, 0.105, 0.0, 1.4, 0.0, 0.0]  # [m, rad]

    # Dynamics and forces
    general_dynamics_rel = 0.32 if mode != Mode.Perform else 0.4
    gripper_speed = 0.06 if mode != Mode.Perform else 0.07  # [m/s]
    gripper_force = 40.0  # [N], default=20

    # Model training
    train_model = False
    train_async = True
    train_model_every_number_cycles = 50
    train_script = Path.home() / 'Documents' / 'bin_picking' / 'scripts' / 'learning' / 'placing.py'

    # Grasping
    grasp_type = 'DEFAULT'  # DEFAULT, SPECIFIC, TYPE
    check_grasp_second_time = False
    adjust_grasp_second_time = False
    change_bin_at_number_of_failed_grasps = 12  # default=10-15
    release_during_grasp_action = False
    release_in_other_bin = True
    release_as_fast_as_possible = False
    random_pose_before_release = False
    max_random_affine_before_release = Affine(0.055, 0.10, 0.0, 1.2)  # [m, rad]
    move_down_distance_for_release = 0.11  # [m]
    measurement_gripper_force = 20.0  # [N], 15
    performance_gripper_force = 40.0  # [N]

    # Je mehr, desto tiefer
    gripper_classes = [0.05, 0.07, 0.086]  # [m]
    grasp_z_offset = 0.015  # [m]

    if 'screw' in grasp_model[0]:
        gripper_classes = [0.025, 0.05, 0.07, 0.085]  # [m]
        grasp_z_offset = 0.008  # [m]

    # Shift
    grasp_shift_threshold = 0.6
    shift_empty_threshold = 0.65  # default: 0.29
    shift_distance = 0.03  # [m]

    # Place
    place_z_offset = -0.009  # [m]

    # Evaluation
    evaluation_path = Path.home() / 'Documents' / 'data' / 'cylinder-cube-1' / 'evaluation' / 'eval.txt'
    change_bin_at_number_of_success_grasps = 11
    number_objects_in_bin = 20

