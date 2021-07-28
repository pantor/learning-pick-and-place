# Learning Pick-and-place

<p align="center">
 We've released our easy-to-use Python package <b>Griffig</b>!<br>
 You can find more information in its <a href="https://github.com/pantor/griffig">repository</a> and on the website <a href="https://griffig.xyz">griffig.xyz</a><br>
<hr>
</p>

In this repository, we've published the code for our publication [*Self-supervised Learning for Precise Pick-and-place without Object Model*](https://pantor.github.io/learning-pick-and-place/) (RA-L 2020). As only parts of the code were specifically written for this publication, we introduce the code structure regarding the overall project idea.

<p align="center">
 <a href="https://drive.google.com/file/d/16NdOv_DnqTnZuyejMnWwtg-25bPpuqwT/view?usp=sharing">
  <img width="440" src="docs/system.JPG?raw=true" alt="Video" />
 </a>
 <br>
 Click the image for a quick demonstration!
</p>


## Structure

The overall structure is as follows:
 - *Scripts* The main part of the project is written in Python. This includes the general program logic, calculating the next action with Tensorflow Keras, data management, learning, ...
 - *Learning* The core part of this repository is learning for various tasks in robotic manipulation. All code for that lies within the `scripts/learning` directory.
  - *Database Server* This is a database server for collecting and uploading data and images. The server has a web interface for showing all episodes in a dataset and displaying the latest action live.
 - *Include / Src* The low-level control of the hardware, in particular for the robot and the cameras, written in C++. The robot uses MoveIt! for control. The camera drivers for Ensenso and RealSense are included, either via direct access or an optional ros node. The latter is helpful because the Ensenso needs a long time to connect and crashes sometimes afterwards.

This project is a ROS package with launch files and a package.xml. The ROS node /move_group is set to respawn=true. This enables to call rosnode kill /move_group to restart it.


## Installation

For the robotic hardware, make sure to load `launch/gripper-config.json` as the Franka end-effector configuration. Currently, following dependencies need to be installed:
- ROS Kinetic
- libfranka & franka_ros
- EnsensoSDK

And all requirements for Python 3.6 via Pip and `python3.6 -m pip install -r requirements.txt`. Patching CvBridge for Python3 and CMake >= 3.12 is given by a snippet in GitLab. It is recommended to export to PYTHONPATH in `.bashrc`: `export PYTHONPATH=$PYTHONPATH:$HOME/Documents/bin_picking/scripts`.


## Start

For an easy start, run `sh terminal-setup.sh` for a complete terminal setup. Start the mongodb daemon. Then run `roslaunch bin_picking moveit.launch`, `rosrun bin_picking grasping.py` and check the database server.


## Hyperparameters

| Group                       | Parameter                               |  Commonly used value           |
| --------------------------- | --------------------------------------- | ------------------------------:|
| **Manipulation Primitives** | Pre-shaped gripper widths               | [0.025, 0.05, 0.07, 0.086] m   |
|                             | Grasp z-offset                          | 0.015 m                        |
|                             | Place z-offset                          | -0.009 m                       |
| **Experiment**              | Approach distance                       | 0.12 m                         |
|                             | Image distance                          | 0.35 m                         |
|                             | Box size                                | 0.172 x 0.281 x 0.07 m         |
|                             | Gripper force                           | 20.0 N                         |
|                             | Change bins for grasping                | True                           |
|                             | Bin empty at max grasp reward           | 0.1                            |
|                             | Change bins at failed grasps            | 12                             |
|                             | Number of selected grasp embeddings     | 200                            |
|                             | Number of selected place embeddings     | 200                            |
|                             | Bin empty at max grasp reward           | 0.1                            |
| **Learning**                | Camera image size                       | 752 x 480 px                   |
|                             | Window image size                       | 200 x 200 px                   |
|                             | Scaled window image size                | 32 x 32 px                     |
|                             | Inference image size                    | 110 x 110 px                   |
|                             | Grasp Loss Weight                       | `1`                            |
|                             | Place Loss Weight                       | `1 + 5 * place_reward`         |
|                             | Merge Loss Weight                       | `4 * (1 + 5 * place_reward)`   |
|                             | Embedding Size z                        | 48                             |
|                             | Training Batch Size                     | 64                             |
|                             | Optimizer                               | Adam with initial LR: 1e-4     |
|                             | LR Scheduler                            | Reduce on plateau: Factor: 0.2, Patience: 20|
|                             | Neural Network Architecture             | [Source Code](https://github.com/pantor/learning-pick-and-place/blob/master/scripts/learning/placing.py) |
| **Image Distribution**      | Use Hindsight                           | True                           |
|                             | Use Further Hindsight                   | True                           |
|                             | Use Negative Foresight                  | True                           |
|                             | Use Own Goal                            | True                           |
|                             | Use Different Goals                     | True                           |
|                             | Jittered Hindsight Images               | 3                              |
|                             | Jittered Hindsight Images (x-axis only) | 3                              |
|                             | Jittered Goal Images                    | 2                              |
|                             | Different Episodes Images               | 2                              |
|                             | Different Episodes Images (reward > 0)  | 4                              |
|                             | Different Object Images (reward > 0)    | 4                              |
|                             | Different Jittered Object Images (reward > 0)| 0                         |
|                             | Jittered Pose Distribution              | Triangular Low: 1 mm, 0.06 rad |
|                             | Jittered Pose Distribution              | Triangular Mid: 6 mm, 0.32 rad |
|                             | Jittered Pose Distribution              | Triangular High: 15 mm, 1.5 rad|


## Robot Learning Database

The robot learning database is a database, server and viewer for research around robotic grasping. It is based on MongoDB, Flask, Vue.js. It shows an overview of all episodes as well as live actions. It can also delete recorded episodes. The server can be started via `python3.6 database/app.py`, afterwards open [localhost](127.0.0.1:8080) in your browser.
