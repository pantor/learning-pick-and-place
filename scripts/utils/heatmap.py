import cv2
import numpy as np

from actions.action import Affine
from inference.inference import Inference
from orthographical import OrthographicImage
from utils.image import draw_line


class Heatmap:
    def __init__(self, model, box, a_space=None):
        self.model = model
        self.inf = Inference(model, box)
        if a_space is not None:
            self.inf.a_space = a_space

    def calculate_heat(self, reward):
        size_input = (752, 480)
        size_reward_center = (reward.shape[1] / 2, reward.shape[2] / 2)
        scale = 200 / 32 * (80.0 / reward.shape[1])

        a_space_idx = range(len(self.inf.a_space))

        heat_values = np.zeros(size_input[::-1], dtype=np.float)
        for i in a_space_idx:
            a = self.inf.a_space[i]
            rot_mat = cv2.getRotationMatrix2D(size_reward_center, -a * 180.0 / np.pi, scale)
            rot_mat[0][2] += size_input[0] / 2 - size_reward_center[0]
            rot_mat[1][2] += size_input[1] / 2 - size_reward_center[1]
            heat_values += cv2.warpAffine(reward[i], rot_mat, size_input, borderValue=0)

        norm = (5 * heat_values.max() + len(a_space_idx)) / 6
        # norm = heat_values.max()

        return heat_values * 255.0 / norm

    def render(
            self,
            image: OrthographicImage,
            goal_image: OrthographicImage = None,
            alpha=0.5,
            save_path=None,
            reward_index=None,
            draw_directions=False,
            indices=None,
        ):
        base = image.mat
        inputs = [self.inf.get_images(image)]

        if goal_image:
            base = goal_image.mat
            inputs += [self.inf.get_images(goal_image)]

        reward = self.model.predict(inputs)
        if reward_index is not None:
            reward = reward[reward_index]

        # reward = np.maximum(reward, 0)
        reward_mean = np.mean(reward, axis=3)
        # reward_mean = reward[:, :, :, 0]

        heat_values = self.calculate_heat(reward_mean)

        heatmap = cv2.applyColorMap(heat_values.astype(np.uint8), cv2.COLORMAP_JET)
        base_heatmap = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB) / 255 + alpha * heatmap
        result = OrthographicImage(base_heatmap, image.pixel_size, image.min_depth, image.max_depth)

        if indices is not None:
            self.draw_indices(result, reward, indices)

        if draw_directions:
            for _ in range(10):
                self.draw_arrow(result, reward, np.unravel_index(reward.argmax(), reward.shape))
                reward[np.unravel_index(reward.argmax(), reward.shape)] = 0

        if save_path:
            cv2.imwrite(str(save_path), result.mat)
        return result.mat

    def draw_indices(self, image: OrthographicImage, reward, indices):
        point_color = (255, 255, 255)

        for index in indices:
            pose = self.inf.pose_from_index(index, reward.shape, image)
            pose.x /= reward.shape[1] / 40
            pose.y /= reward.shape[2] / 40

            draw_line(image, pose, Affine(-0.001, 0), Affine(0.001, 0), color=point_color, thickness=1)
            draw_line(image, pose, Affine(0, -0.001), Affine(0, 0.001), color=point_color, thickness=1)

    def draw_arrow(self, image: OrthographicImage, reward, index):
        pose = self.inf.pose_from_index(index, reward.shape, image)

        arrow_color = (255, 255, 255)
        draw_line(image, pose, Affine(0, 0), Affine(0.036, 0), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, -0.008), color=arrow_color, thickness=2)
        draw_line(image, pose, Affine(0.036, 0.0), Affine(0.026, 0.008), color=arrow_color, thickness=2)
