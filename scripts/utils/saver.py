from enum import Enum
import json
from typing import List

import cv2
import numpy as np
import requests

from actions.action import RobotPose
from orthographical import OrthographicImage
from cfrankr import Affine


class Saver:
    def __init__(self, url, collection):
        self.url = url
        self.collection = collection

    def save_image(self, images: List[OrthographicImage], episode_id: str, action_id: int, suffix: str, action=None):
        for image in images:
            image_data = cv2.imencode('.png', image.mat)[1].tostring()
            suffix_final = f'{image.camera}-{suffix}'

            if action:
                action.images[suffix_final] = {
                    'info': {
                        'pixel_size': image.pixel_size,
                        'min_depth': image.min_depth,
                        'max_depth': image.max_depth,
                    },
                    'pose': Affine(image.pose),
                }

            try:
                requests.post(self.url + 'upload-image', files={
                    'file': ('image.png', image_data, 'image/png', {'Expires': '0'})
                }, data={
                    'collection': self.collection,
                    'episode_id': episode_id,
                    'action_id': action_id,
                    'suffix': suffix_final,
                })
            except requests.exceptions.RequestException:
                raise Exception('Could not save image!')

    @classmethod
    def _jsonify_episode(cls, x):
        if isinstance(x, RobotPose):
            return {'x': x.x, 'y': x.y, 'z': np.nan_to_num(x.z), 'a': x.a, 'b': x.b, 'c': x.c, 'd': x.d}
        if isinstance(x, Affine):
            return {'x': x.x, 'y': x.y, 'z': np.nan_to_num(x.z), 'a': x.a, 'b': x.b, 'c': x.c}
        if isinstance(x, np.int64):
            return int(x)
        if isinstance(x, np.float32):
            return float(x)
        if isinstance(x, Enum):
            return x.name
        return x.__dict__

    def save_action_plan(self, action, episode_id: str):
        try:
            requests.post(self.url + 'new-attempt', data={
                'json': json.dumps({
                    'action': action,
                    'collection': self.collection,
                    'episode_id': episode_id
                }, default=Saver._jsonify_episode),
            })
        except requests.exceptions.RequestException:
            raise Exception('Could not save action plan!')

    def save_episode(self, episode):
        try:
            requests.post(self.url + 'new-episode', data={
                'collection': self.collection,
                'id': episode.id,
                'json': json.dumps({'episode': episode}, default=Saver._jsonify_episode),
            })
        except requests.exceptions.RequestException:
            raise Exception('Could not save action result!')
