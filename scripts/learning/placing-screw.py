import argparse
import copy
import hashlib
from functools import lru_cache
import os
from time import time

import cv2
from loguru import logger
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk  # pylint: disable=E0401
import tensorflow.keras.layers as tkl  # pylint: disable=E0401

from actions.action import RobotPose
from actions.indexer import GraspIndexer
from config import Config
from data.loader import Loader
from learning.utils.layers import conv_block_gen, dense_block_gen
from learning.utils.metrics import Losses, SplitMeanSquaredError, SplitBinaryAccuracy, SplitPrecision, SplitRecall
from utils.image import draw_around_box, get_area_of_interest_new


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class PlacingDataset:
    def __init__(self, episodes, seed=None):
        self.episodes = episodes
        self.episodes_place_success_index = list(map(lambda e: e[0], filter(lambda e: len(e[1]['actions']) > 1 and e[1]['actions'][1]['reward'] > 0, enumerate(episodes))))
        # self.episodes_different_objects_index = list(map(lambda e: e[0], filter(lambda e: '2019-12-16-17-01-18-409' <= e[1]['id'] <= '2020-01-16-17-09-47-989' and e[1]['actions'][1]['reward'] > 0, enumerate(episodes))))

        self.size_input = (752, 480)
        self.size_memory_scale = 4
        self.size_cropped = (200, 200)
        self.size_result = (32, 32)

        self.size_cropped_area = (self.size_cropped[0] // self.size_memory_scale, self.size_cropped[1] // self.size_memory_scale)

        self.use_hindsight = True
        self.use_further_hindsight = False
        self.use_negative_foresight = True
        self.use_own_goal = True
        self.use_different_episodes_as_goals = True

        self.jittered_hindsight_images = 3
        self.jittered_hindsight_x_images = 3
        self.jittered_goal_images = 2
        self.different_episodes_images = 2
        self.different_object_images = 5
        # self.different_jittered_object_images = 2

        self.box_distance = 0.281  # [m]

        # self.indexer = GraspIndexer([0.05, 0.07, 0.086])  # [m]
        self.indexer = GraspIndexer([0.025, 0.05, 0.07, 0.086])  # [m]

        self.cameras = ('ed', 'rd', 'rc')

        self.random_gen = np.random.RandomState(seed)

    @lru_cache(maxsize=None)
    def load_image(self, collection, episode_id, action_id, suffix):
        image = Loader.get_image(collection, episode_id, action_id, suffix, as_float=True)
        draw_around_box(image, box=Config.box)

        image.mat = cv2.resize(image.mat, (self.size_input[0] // self.size_memory_scale, self.size_input[1] // self.size_memory_scale))
        image.pixel_size /= self.size_memory_scale
        return image

    def area_of_interest(self, image, pose):
        area = get_area_of_interest_new(
            image,
            RobotPose(all_data=pose),
            size_cropped=self.size_cropped_area,
            size_result=self.size_result,
            border_color=image.value_from_depth(self.box_distance) / (255 * 255),
        )
        if len(area.mat.shape) == 2:
            return np.expand_dims(area.mat, 2)
        return area.mat

    def jitter_pose(self, pose, scale_x=0.05, scale_y=0.05, scale_a=1.5):
        new_pose = copy.deepcopy(pose)

        low = [np.minimum(0.002, scale_x), np.minimum(0.002, scale_y), np.minimum(0.05, scale_a)]
        high = [scale_x, scale_y, scale_a]
        dx, dy, da = self.random_gen.choice([-1, 1], size=3) * self.random_gen.uniform(low, high, size=3)

        new_pose['x'] += np.cos(pose['a']) * dx - np.sin(pose['a']) * dy
        new_pose['y'] += np.sin(pose['a']) * dx + np.cos(pose['a']) * dy
        new_pose['a'] += da
        return new_pose

    def generator(self, index):
        e = self.episodes[index]

        result = []

        collection = e['collection']
        episode_id = e['id']

        grasp = e['actions'][0]
        grasp_before = tuple(self.load_image(collection, episode_id, 0, camera + '-v') for camera in self.cameras)
        grasp_before_area = tuple(self.area_of_interest(image, grasp['pose']) for image in grasp_before)
        grasp_index = self.indexer.from_pose(grasp['pose'])

        # Only grasp
        if len(e['actions']) == 1:
            zeros = (np.zeros(self.size_result + (1,)), np.zeros(self.size_result + (1,)), np.zeros(self.size_result + (3,)))
            result = [grasp_before_area + zeros + zeros + (
                (grasp['reward'], grasp_index, 0.4),
                (0, 0, 0),
                (0, 0, 0),
            )]
            return [np.array(t, dtype=np.float32) for t in zip(*result)]


        place = e['actions'][1]
        place_before = tuple(self.load_image(collection, episode_id, 1, camera + '-v') for camera in self.cameras)
        place_after = tuple(self.load_image(collection, episode_id, 1, camera + '-after') for camera in self.cameras)


        # Generate goal has no action_id
        def generate_goal(g_collection, g_episode_id, g_suffix, g_pose, g_reward=0, g_index=None, g_merge_weight=1.0, jitter=None):
            if g_collection == collection and g_episode_id == episode_id and g_suffix == 'v':
                place_goal_before = place_before
                place_goal = place_before
            elif g_collection == collection and g_episode_id == episode_id and g_suffix == 'after':
                place_goal_before = place_before
                place_goal = place_after
            else:
                goal_e = self.episodes[g_index]
                g_collection = g_collection if g_collection else goal_e['collection']
                g_episode_id = g_episode_id if g_episode_id else goal_e['id']
                g_pose = g_pose if g_pose else goal_e['actions'][1]['pose']

                place_goal_before = tuple(self.load_image(g_collection, g_episode_id, 1, camera + '-v') for camera in self.cameras)
                place_goal = tuple(self.load_image(g_collection, g_episode_id, 1, camera + '-' + g_suffix) for camera in self.cameras)

            if isinstance(jitter, dict):
                g_pose = self.jitter_pose(g_pose, **jitter)

            place_before_area = tuple(self.area_of_interest(image, g_pose) for image in place_goal_before)
            place_goal_area = tuple(self.area_of_interest(image, g_pose) for image in place_goal)

            goal_reward = g_reward

            reward_grasp = grasp['reward']
            reward_place = goal_reward * place['reward']
            reward_merge = reward_place

            grasp_weight = g_reward
            place_weight = (1.0 + 5.0 * reward_place) * reward_grasp
            merge_weight = (1.0 + 5.0 * reward_merge) * g_merge_weight

            return grasp_before_area + place_before_area + place_goal_area + (
                (reward_grasp, grasp_index, grasp_weight),
                (reward_place, 0, place_weight),
                (reward_merge, 0, merge_weight),
            )

        if self.use_hindsight:
            result.append(generate_goal(collection, episode_id, 'after', place['pose'], g_reward=1))

            result += [
                generate_goal(collection, episode_id, 'after', place['pose'], jitter={})
                for _ in range(self.jittered_hindsight_images)
            ]
            result += [
                generate_goal(collection, episode_id, 'after', place['pose'], jitter={'scale_x': 0.025, 'scale_y': 0, 'scale_a': 0})
                for _ in range(self.jittered_hindsight_x_images)
            ]

        if self.use_further_hindsight and 'bin_episode' in place:
            for i in range(index + 1, len(self.episodes)):
                place_later = self.episodes[i]['actions'][1]
                if place_later['bin_episode'] != place['bin_episode']:
                    break

                if place_later['reward'] > 0:
                    result.append(generate_goal(None, None, 'after', place['pose'], g_index=i, g_reward=1))

        if self.use_negative_foresight:
            result.append(generate_goal(collection, episode_id, 'v', place['pose']))

        if self.use_own_goal and 'ed-goal' in place['images']:
            result.append(generate_goal(collection, episode_id, 'goal', place['pose'], g_index=index))

            result += [
                generate_goal(collection, episode_id, 'goal', place['pose'], g_index=index, g_merge_weight=0.5, jitter={})
                for _ in range(self.jittered_goal_images)
            ]

        if self.use_different_episodes_as_goals:
            result += [
                generate_goal(None, None, 'after', None, g_index=goal_index, g_merge_weight=0.3)
                for goal_index in self.random_gen.choice(self.episodes_place_success_index, size=self.different_episodes_images)
            ]

            # result += [
            #     generate_goal(None, None, 'after', None, g_index=goal_index, g_merge_weight=0.3)
            #     for goal_index in self.random_gen.choice(self.episodes_different_objects_index, size=self.different_object_images)
            # ]

            # result += [
            #     generate_goal(None, None, 'after', None, g_index=goal_index, g_merge_weight=0.3, jitter={})
            #     for goal_index in self.random_gen.choice(self.episodes_different_objects_index, size=self.different_jittered_object_images)
            # ]

        return [np.array(t, dtype=np.float32) for t in zip(*result)]

    def tf_generator(self, index):
        r = tf.py_function(
            self.generator,
            [index],
            (tf.float32,) * (3 * len(self.cameras) + 3),
        )
        r[0].set_shape((None, 32, 32, 1))
        r[1].set_shape((None, 32, 32, 1))
        r[2].set_shape((None, 32, 32, 3))
        r[3].set_shape((None, 32, 32, 1))
        r[4].set_shape((None, 32, 32, 1))
        r[5].set_shape((None, 32, 32, 3))
        r[6].set_shape((None, 32, 32, 1))
        r[7].set_shape((None, 32, 32, 1))
        r[8].set_shape((None, 32, 32, 3))
        r[9].set_shape((None, 3))
        r[10].set_shape((None, 3))
        r[11].set_shape((None, 3))
        return (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]), (r[9], r[10], r[11])

    def get_data(self, shuffle=None):
        data = tf.data.Dataset.range(0, len(self.episodes))
        if shuffle:
            shuffle_number = len(self.episodes) if shuffle == 'all' else int(shuffle)
            data = data.shuffle(shuffle_number)
        data = data.map(self.tf_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return data.interleave(lambda *x: tf.data.Dataset.from_tensor_slices(x), cycle_length=1)


class Placing:
    def __init__(self, collections, mongo_host='localhost', data_path=None, image_format='png'):
        self.loader = Loader(mongo_host, data_path=data_path, image_format=image_format)
        self.model_path = self.loader.get_model_path(f'placing-3-15-screw-type-2')  # [.h5]

        train_batch_size = 64
        validation_batch_size = 512

        self.image_shape = {
            'ed': (None, None, 1),
            'rd': (None, None, 1),
            'rc': (None, None, 3),
        }

        self.z_size = 32

        self.percent_validation_set = 0.2

        number_primitives = 4 if 'screw' in str(self.model_path.stem) else 3

        load_model = False
        use_beta_checkpoint_path = True
        checkpoint_path = self.model_path if not use_beta_checkpoint_path else self.model_path.with_suffix('.beta' + self.model_path.suffix)

        episodes = self.loader.yield_episodes(
            collections,
            query={'actions.0.type': 'grasp'},
            projection={'_id': 0, 'id': 1, 'actions.pose': 1, 'actions.reward': 1, 'actions.images': 1}
        )
        train_episodes, validation_episodes = self.split_set(episodes)

        train_data = PlacingDataset(train_episodes, seed=42).get_data(shuffle='all')
        train_data = train_data.shuffle(len(train_episodes) * 6)
        train_data = train_data.batch(train_batch_size)
        train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

        validation_data = PlacingDataset(validation_episodes, seed=43).get_data()
        validation_data = validation_data.cache()
        validation_data = validation_data.batch(validation_batch_size)
        validation_data = validation_data.prefetch(tf.data.experimental.AUTOTUNE)


        self.grasp_model = self.define_grasp_model(number_primitives=number_primitives)
        self.place_model = self.define_place_model()
        self.merge_model = self.define_merge_model()

        images_grasp_before = [
            tk.Input(shape=self.image_shape['ed'], name='image_grasp_before_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_grasp_before_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_grasp_before_rc'),
        ]

        images_place_before = [
            tk.Input(shape=self.image_shape['ed'], name='image_place_before_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_place_before_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_place_before_rc'),
        ]

        images_place_goal = [
            tk.Input(shape=self.image_shape['ed'], name='image_place_goal_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_place_goal_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_place_goal_rc'),
        ]

        reward_m, z_m = self.grasp_model(images_grasp_before)
        reward_p, z_p = self.place_model(images_place_before + images_place_goal)
        reward = self.merge_model([z_m, z_p])

        losses = Losses()

        self.combined = tk.Model(inputs=(images_grasp_before + images_place_before + images_place_goal), outputs=[reward_m, reward_p, reward])
        self.combined.summary()
        self.combined.compile(
            optimizer=tk.optimizers.Adam(learning_rate=2e-4),
            loss=losses.binary_crossentropy,
            loss_weights=[1.0, 1.0, 2.0],
            metrics=[
                losses.binary_crossentropy,
                SplitMeanSquaredError(),
                SplitBinaryAccuracy(),
                SplitPrecision(),
                SplitRecall(),
            ],
        )

        callbacks = [
            tk.callbacks.ModelCheckpoint(
                str(checkpoint_path),
                monitor=f'val_loss',
                verbose=1,
                save_best_only=True
            ),
            tk.callbacks.EarlyStopping(monitor=f'val_loss', patience=60),
            tk.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1, patience=15),
            tf.keras.callbacks.TensorBoard(log_dir=str(self.model_path.parent / 'logs' / f'placing-{time()}')),
        ]

        if load_model:
            self.combined.load_weights(str(self.model_path))
            evaluation = self.combined.evaluate(validation_data, verbose=2)
            callbacks[0].best = evaluation[self.combined.metrics_names.index('loss')]

        self.combined.fit(
            train_data,
            validation_data=validation_data,
            epochs=1000,
            callbacks=callbacks,
            verbose=2,
        )

        self.combined.load_weights(str(checkpoint_path))
        if use_beta_checkpoint_path:
            self.combined.save(str(self.model_path), save_format='h5')

    def define_grasp_model(self, number_primitives: int):
        inputs = [
            tk.Input(shape=self.image_shape['ed'], name='image_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_rc'),
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.42)

        x = tkl.Concatenate()(inputs)

        x = conv_block(x, 32)
        x = conv_block(x, 32, strides=(2, 2))
        x = conv_block(x, 32)

        x = conv_block(x, 48)
        x = conv_block(x, 48)

        x_r = conv_block(x, 64)
        x_r = conv_block(x_r, 64)

        x_r = conv_block(x_r, 64)
        x_r = conv_block(x_r, 48, kernel_size=(2, 2))

        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 96)
        x = conv_block(x, 96, kernel_size=(2, 2))

        reward = tkl.Conv2D(number_primitives, kernel_size=(1, 1), activation='sigmoid', name='reward_grasp')(x_r)
        reward_training = tkl.Reshape((number_primitives,))(reward)

        z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.001), name='z_m0')(x)
        z_training = tkl.Reshape((self.z_size,))(z)
        outputs = [reward_training, z_training]
        return tk.Model(inputs=inputs, outputs=outputs, name='grasp')

    def define_place_model(self):
        inputs = [
            tk.Input(shape=self.image_shape['ed'], name='image_before_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_before_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_before_rc'),
            tk.Input(shape=self.image_shape['ed'], name='image_goal_ed'),
            tk.Input(shape=self.image_shape['rd'], name='image_goal_rd'),
            tk.Input(shape=self.image_shape['rc'], name='image_goal_rc'),
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.42)

        x = tkl.Concatenate()(inputs)

        x = conv_block(x, 32)
        x = conv_block(x, 32)

        x = conv_block(x, 32, dilation_rate=(2, 2))
        x = conv_block(x, 32, dilation_rate=(2, 2))

        x = conv_block(x, 48)
        x = conv_block(x, 48)

        x = conv_block(x, 48, dilation_rate=(2, 2))
        x = conv_block(x, 48, dilation_rate=(2, 2))

        x_r = conv_block(x, 64)
        x_r = conv_block(x_r, 64)

        x_r = conv_block(x_r, 96)
        x_r = conv_block(x_r, 64, kernel_size=(2, 2))

        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 96)
        x = conv_block(x, 96, kernel_size=(2, 2))

        reward = tkl.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='reward_place')(x_r)
        z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.001), name='z_p')(x)

        reward_training = tkl.Reshape((1,))(reward)
        z_training = tkl.Reshape((self.z_size,))(z)
        outputs = [reward_training, z_training]
        return tk.Model(inputs=inputs, outputs=outputs, name='place')

    def define_merge_model(self):
        z_m = tk.Input(shape=(self.z_size), name='z_m')
        z_p = tk.Input(shape=(self.z_size), name='z_p')

        dense_block = dense_block_gen(l2_reg=0.001, dropout_rate=0.35)

        # x = tkl.Concatenate()([z_m, z_p])
        x = z_m - z_p

        x = dense_block(x, 64)
        x = dense_block(x, 64)

        reward = tkl.Dense(1, activation='sigmoid', name='reward_merge')(x)
        return tk.Model(inputs=[z_m, z_p], outputs=[reward], name='merge')

    @staticmethod
    def binary_decision(string: str, p: float) -> bool:
        return float(int(hashlib.sha256(string.encode('utf-8')).hexdigest(), 16) % 2**16) / 2**16 < p

    def assign_set(self, data):
        collection, episode = data
        random_assign = self.binary_decision(episode['id'], self.percent_validation_set)
        episode['is_validation'] = random_assign # or (collection in [])
        episode['collection'] = collection
        return episode

    def split_set(self, data, verbose=1):
        episodes = list(map(self.assign_set, data))

        train_episodes = list(filter(lambda x: not x['is_validation'], episodes))
        validation_episodes = list(filter(lambda x: x['is_validation'], episodes))

        if verbose > 0:
            logger.info(f'Train on {len(train_episodes)} episodes.')
            logger.info(f'Validate on {len(validation_episodes)} episodes.')

        return train_episodes, validation_episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
    parser.add_argument('-d', '--collection', action='append', dest='collection', type=str, default=['placing-screw-1', 'cylinder-screw-3'])
    parser.add_argument('-i', '--image_format', action='store', dest='image_format', type=str, default='png')
    parser.add_argument('-p', '--path', action='store', dest='data_path', type=str, default=None)
    parser.add_argument('--mongo-host', action='store', dest='mongo_host', type=str, default='localhost')

    args = parser.parse_args()

    placing = Placing(
        collections=args.collection,
        mongo_host=args.mongo_host,
        image_format=args.image_format,
        data_path=args.data_path,
    )
