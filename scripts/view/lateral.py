from pathlib import Path
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model  # pylint: disable=E0401

from orthographical import OrthographicImage
from config import Config
from data.loader import Loader
from utils.image import draw_around_box, crop


class LateralViewer:
    def __init__(self, model_path):
        self.size_input = (752, 480)
        self.size_original_cropped = (200, 200)
        self.size_output = (32, 32)
        self.size_cropped = (128, 128)

        self.model = load_model(str(model_path), compile=False)

    def transform_image(self, image: OrthographicImage):
        scale = self.size_output[0] / self.size_original_cropped[0]
        rot_mat = cv2.getRotationMatrix2D((self.size_input[0] / 2, self.size_input[1] / 2), 0.0, scale)
        rot_mat[:, 2] += [
            (self.size_cropped[0] - self.size_input[0]) / 2,
            (self.size_cropped[1] - self.size_input[1]) / 2,
        ]
        return cv2.warpAffine(image.mat, rot_mat, self.size_cropped, borderMode=cv2.BORDER_REPLICATE)

    def predict(self, image: OrthographicImage):
        draw_around_box(image, box=Config.box)

        mat_image = self.transform_image(image.mat)

        cv2.imshow('image', mat_image)

        scale_factor = float(self.size_resized[0]) / self.size_input[0]
        image = image.rotate_x(-0.5, [0.0, 0.30])

        image = (image.mat / 127.5) - 1.0
        images = np.array([np.expand_dims(image, axis=3)])
        image_prediction = self.model.predict(images)[0]
        image_prediction = 0.5 * (image_prediction + 1.0)
        return np.squeeze(image_prediction, axis=2)


if __name__ == '__main__':
    lateral = LateralViewer(
        Path.home() / 'Documents' / 'deep-stitch' / 'pix2pix' / 'models' / 'side-generator-model2.h5'
    )

    action, image = Loader.get_action('cylinder-2', '2018-12-19-17-47-59-970', 0, 'ed-v')
    image = image.rotate_x(-0.1, [0.0, 0.3])

    start = time.time()

    lateral_image = lateral.predict(image)

    print(f'Image time [s]: {time.time() - start:.3}')

    image = cv2.resize(image, (376, 240))
    lateral_image = crop(lateral_image, (82, 128))

    cv2.imshow('image', image.mat)
    cv2.imshow('lateral_image', cv2.resize(lateral_image, (376, 240)))
    cv2.waitKey(1500)
