from pathlib import Path
from typing import Any, List, Union, Tuple

import cv2
import numpy as np
from pymongo import MongoClient
import tensorflow as tf
from tensorflow.keras import Model, models  # pylint: disable=E0401

from actions.action import Action, RobotPose
from cfrankr import Affine
from orthographical import OrthographicImage


class Loader:
    database = MongoClient()['robot-learning']

    data_path = Path.home() / 'Documents' / 'data'
    image_format = 'png'

    def __new__(cls, host='localhost', port=27017, data_path=None, image_format='png'):
        cls.data_path = Path(data_path) if data_path else cls.data_path
        cls.image_format = image_format
        cls.database = MongoClient(host, port)['robot-learning']
        return cls

    @classmethod
    def get_collections(cls):
        return cls.database.collection_names()

    @classmethod
    def get_episode(cls, collection: str, episode_id: str):
        return cls.database[collection].find_one({'id': episode_id})

    @classmethod
    def get_episode_count(cls, collections: Union[str, List[str]], query=None, suffixes=None) -> int:
        query = query or {}

        if isinstance(collections, str):
            collections = [collections]

        if suffixes:
            for s in suffixes:
                query[f'actions.0.images.{s}'] = {'$exists': True}

        return sum(cls.database[c].count(query) for c in collections)

    @classmethod
    def yield_episodes(cls, collections: Union[str, List[str]], query=None, suffixes=None, projection=None):
        query = query or {}

        if isinstance(collections, str):
            collections = [collections]

        if suffixes:
            for s in suffixes:
                query[f'actions.0.images.{s}'] = {'$exists': True}

        for collection in collections:
            if collection not in cls.database.collection_names():
                raise Exception(f'Database has no collection named {collection}!')

            for episode in cls.database[collection].find(query, projection):
                yield collection, episode

    @classmethod
    def get_collection_path(cls, collection: str) -> Path:
        return cls.data_path / collection

    @classmethod
    def get_image_path(cls, collection: str, episode_id: str, action_id: int, suffix: str, image_format=None) -> Path:
        suffix = f'{action_id}-{suffix}' if action_id > 0 else suffix  # For downward compatibility
        return cls.data_path / collection / 'measurement' / f'image-{episode_id}-{suffix}.{image_format if image_format else cls.image_format}'

    @classmethod
    def get_image(cls, collection: str, episode_id: str, action_id: int, suffix: str, images=None, image_data=None, as_float=False) -> OrthographicImage:
        image = cv2.imread(str(cls.get_image_path(collection, episode_id, action_id, suffix)), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f'Image {collection} {episode_id} {action_id} {suffix} not found.')

        if not as_float and image.dtype == np.uint8:
            image = image.astype(np.uint16)
            image *= 255
        elif as_float:
            mult = 255 if image.dtype == np.uint8 else 255 * 255
            image = image.astype(np.float32)
            image /= mult

        if image_data:
            image_data_result = {
                'info': image_data['info'],
                'pose': RobotPose(data=image_data['pose']),
            }

        elif images:
            image_data_result = images[suffix]

        else:
            episode = cls.database[collection].find_one({'id': episode_id}, {'actions.images': 1})
            if not episode or not episode['actions'] or not (0 <= action_id < len(episode['actions'])):
                raise Exception(f'Internal mismatch of image {collection} {episode_id} {action_id} {suffix} not found.')

            image_data_result = Action(data=episode['actions'][action_id]).images[suffix]

        return OrthographicImage(
            image,
            image_data_result['info']['pixel_size'],
            image_data_result['info']['min_depth'],
            image_data_result['info']['max_depth'],
            suffix.split('-')[0],
            image_data_result['pose'].to_array(),
            # list(image_data_result['pose'].values()),
            # Affine(image_data_result['pose']).to_array(),
        )

    @classmethod
    def get_action(cls, collection: str, episode_id: str, action_id: int, suffix: Union[str, List[str]] = None) -> Any:
        episode = Loader.get_episode(collection, episode_id)

        if not episode:
            raise Exception(f'Episode {episode_id} not found')

        if not episode['actions'] or not (0 <= action_id < len(episode['actions'])):
            raise Exception(f'Episode {episode_id} has not enough actions.')

        action = Action(data=episode['actions'][action_id])
        if not suffix:
            return action

        if isinstance(suffix, str):
            suffix = [suffix]

        return (action, *[cls.get_image(collection, episode_id, action_id, s, images=action.images) for s in suffix])

    @classmethod
    def get_model_path(cls, name: Union[str, Tuple[str, str]], collection: str = None) -> Path:
        if isinstance(name, str) and not collection:
            return cls.data_path / 'models' / f'{name}.h5'

        elif isinstance(name, tuple):  # To load (collection, name) as tuple
            collection, name = name

        return cls.data_path / collection / 'models' / f'{name}.h5'

    @classmethod
    def get_model(
            cls,
            name: Union[str, Tuple[str, str]],
            collection: str = None,
            output_layer: Union[str, List[str]] = None,
            custom_objects=None
        ) -> Model:
        for device in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        model_path = cls.get_model_path(name, collection)
        model = models.load_model(str(model_path), compile=False, custom_objects=custom_objects)

        if not output_layer:
            return model

        if isinstance(output_layer, str):
            output_layer = [output_layer]

        return Model(inputs=model.input, outputs=[model.get_layer(l).output for l in output_layer])
