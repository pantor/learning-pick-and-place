import argparse
import copy
import hashlib
from functools import lru_cache
import os
import random
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


class PlacingDataset:
    def __init__(self, episodes, seed=None):
        self.episodes = episodes
        self.episodes_place_success_index = list(map(lambda e: e[0], filter(lambda e: e[1]['actions'][1]['reward'] > 0, enumerate(episodes))))

        self.episodes_different_objects_ids = {
            'wooden': ('2019-12-16-17-01-18-409', '2020-01-22-09-34-02-952'),
            'baby': ('2020-01-22-09-34-02-953', '2020-01-29-23-17-15-032'),
        }

        # Get indexes of episodes between the ids (from above) which have a positive place action
        self.episodes_different_objects_index = {
            k: list(map(lambda e: e[0], filter(lambda e: v[0] <= e[1]['id'] <= v[1] and e[1]['actions'][1]['reward'] > 0, enumerate(episodes))))
            for k, v in self.episodes_different_objects_ids.items()
        }


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

        self.jittered_hindsight_images = 1
        self.jittered_hindsight_x_images = 2  # Only if place reward > 0
        self.jittered_goal_images = 1
        self.different_episodes_images = 1
        self.different_episodes_images_success = 4  # Only if place reward > 0
        self.different_object_images = 4  # Only if place reward > 0
        self.different_jittered_object_images = 0  # Only if place reward > 0

        self.box_distance = 0.281  # [m]

        self.indexer = GraspIndexer([0.05, 0.07, 0.086])  # [m]
        # self.indexer = GraspIndexer([0.025, 0.05, 0.07, 0.086])  # [m]

        self.cameras = ('ed',)
        # self.cameras = ('ed', 'rd', 'rc')

        self.seed = seed
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

    def jitter_pose(self, pose, scale_x=0.05, scale_y=0.05, scale_a=1.5, around=True):
        new_pose = copy.deepcopy(pose)

        if around:
            low = [np.minimum(0.001, scale_x), np.minimum(0.001, scale_y), np.minimum(0.06, scale_a)]
            mode = [np.minimum(0.006, scale_x), np.minimum(0.006, scale_y), np.minimum(0.32, scale_a)]
            high = [scale_x + 1e-6, scale_y + 1e-6, scale_a + 1e-6]
            dx, dy, da = self.random_gen.choice([-1, 1], size=3) * self.random_gen.triangular(low, mode, high, size=3)
        else:
            low = [-scale_x - 1e-6, -scale_y - 1e-6, -scale_a - 1e-6]
            mode = [0.0, 0.0, 0.0]
            high = [scale_x + 1e-6, scale_y + 1e-6, scale_a + 1e-6]
            dx, dy, da = self.random_gen.triangular(low, mode, high, size=3)

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
        grasp_before = self.load_image(collection, episode_id, 0, 'ed-v')
        grasp_before_area = self.area_of_interest(grasp_before, grasp['pose'])
        grasp_index = self.indexer.from_pose(grasp['pose'])

        # Only single grasp
        if len(e['actions']) == 1:
            pass

        place = e['actions'][1]
        place_before = self.load_image(collection, episode_id, 1, 'ed-v')
        place_after = self.load_image(collection, episode_id, 1, 'ed-after')

        # Generate goal has no action_id
        def generate_goal(g_collection, g_episode_id, g_suffix, g_pose, g_suffix_before='v', g_reward=0, g_index=None, g_place_weight=1.0, g_merge_weight=1.0, jitter=None):
            if g_collection == collection and g_episode_id == episode_id and g_suffix == 'v' and g_suffix_before == 'v':
                place_goal_before = place_before
                place_goal = place_before
            elif g_collection == collection and g_episode_id == episode_id and g_suffix == 'v' and g_suffix_before == 'after':
                place_goal_before = place_after
                place_goal = place_before
            elif g_collection == collection and g_episode_id == episode_id and g_suffix == 'after' and g_suffix_before == 'v':
                place_goal_before = place_before
                place_goal = place_after
            elif g_collection == collection and g_episode_id == episode_id and g_suffix == 'after' and g_suffix_before == 'after':
                place_goal_before = place_after
                place_goal = place_after
            else:
                goal_e = self.episodes[g_index]

                g_collection = g_collection if g_collection else goal_e['collection']
                g_episode_id = g_episode_id if g_episode_id else goal_e['id']
                g_pose = g_pose if g_pose else goal_e['actions'][1]['pose']

                place_goal_before = self.load_image(g_collection, g_episode_id, 1, 'ed-' + g_suffix_before)
                place_goal = self.load_image(g_collection, g_episode_id, 1, 'ed-' + g_suffix)

            if isinstance(jitter, dict):
                g_pose = self.jitter_pose(g_pose, **jitter)

            place_before_area = self.area_of_interest(place_goal_before, g_pose)
            place_goal_area = self.area_of_interest(place_goal, g_pose)

            reward_grasp = grasp['reward']
            reward_place = g_reward * grasp['reward'] * place['reward']
            reward_merge = reward_place

            grasp_weight = g_reward
            place_weight = (1.0 + 3.0 * reward_place) * reward_grasp * g_place_weight
            merge_weight = (1.0 + 3.0 * reward_merge) * reward_grasp * g_merge_weight

            return (
                grasp_before_area,
                place_before_area,
                place_goal_area,
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

            if place['reward'] > 0:
                result += [
                    generate_goal(collection, episode_id, 'after', place['pose'], jitter={'scale_x': 0.02, 'scale_y': 0.01, 'scale_a': 0.2})
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
            g_suffix, g_suffix_before = random.choice([('v', 'v'), ('after', 'after'), ('v', 'after')])
            result.append(generate_goal(collection, episode_id, g_suffix, place['pose'], g_suffix_before=g_suffix_before, jitter={'around': False}))

        if self.use_own_goal and 'ed-goal' in place['images']:
            result.append(generate_goal(collection, episode_id, 'goal', place['pose'], g_place_weight=0.2, g_merge_weight=0.7, g_index=index))

            result += [
                generate_goal(collection, episode_id, 'goal', place['pose'], g_index=index, jitter={})
                for _ in range(self.jittered_goal_images)
            ]

        if self.use_different_episodes_as_goals:
            result += [
                generate_goal(None, None, 'after', None, g_index=goal_index, g_place_weight=0.0)
                for goal_index in self.random_gen.choice(self.episodes_place_success_index, size=self.different_episodes_images)
            ]

            if place['reward'] > 0:
                result += [
                    generate_goal(None, None, 'after', None, g_index=goal_index, g_place_weight=0.0)
                    for goal_index in self.random_gen.choice(self.episodes_place_success_index, size=self.different_episodes_images_success)
                ]

                for k, v in self.episodes_different_objects_ids.items():
                    if v[0] <= e['id'] <= v[1]:
                        result += [
                            generate_goal(None, None, 'after', None, g_index=goal_index, g_place_weight=0.0)
                            for goal_index in self.random_gen.choice(self.episodes_different_objects_index[k], size=self.different_object_images)
                        ]

                        # result += [
                        #     generate_goal(None, None, 'after', None, g_index=goal_index, jitter={})
                        #     for goal_index in self.random_gen.choice(self.episodes_different_objects_index[k], size=self.different_jittered_object_images)
                        # ]

        return [np.array(t, dtype=np.float32) for t in zip(*result)]

    def tf_generator(self, index):
        r = tf.py_function(
            self.generator,
            [index],
            (tf.float32,) * 6,
        )
        r[0].set_shape((None, 32, 32, 1))
        r[1].set_shape((None, 32, 32, 1))
        r[2].set_shape((None, 32, 32, 1))
        r[3].set_shape((None, 3))
        r[4].set_shape((None, 3))
        r[5].set_shape((None, 3))
        return (r[0], r[1], r[2]), (r[3], r[4], r[5])

    def get_data(self, shuffle=None):
        data = tf.data.Dataset.range(0, len(self.episodes))
        if shuffle:
            shuffle_number = len(self.episodes) if shuffle == 'all' else int(shuffle)
            data = data.shuffle(shuffle_number, seed=self.seed)
        data = data.map(self.tf_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # data = data.map(self.tf_generator)
        return data.interleave(lambda *x: tf.data.Dataset.from_tensor_slices(x), cycle_length=1)


class Placing:
    def __init__(self, collections, mongo_host='localhost', data_path=None, image_format='png'):
        self.loader = Loader(mongo_host, data_path=data_path, image_format=image_format)
        self.model_path = self.loader.get_model_path(f'placing-3-32-part-type-2')  # [.h5]

        train_batch_size = 64
        validation_batch_size = 512

        self.image_shape = {
            'ed': (None, None, 1),
            'rd': (None, None, 1),
            'rc': (None, None, 3),
        }

        self.z_size = 48

        self.percent_validation_set = 0.2

        number_primitives = 4 if 'screw' in str(self.model_path.stem) else 3

        load_model = False
        use_beta_checkpoint_path = True
        checkpoint_path = self.model_path if not use_beta_checkpoint_path else self.model_path.with_suffix('.beta' + self.model_path.suffix)

        episodes = self.loader.yield_episodes(
            collections,
            query={'$or': [
                # {'actions': {'$size': 1}, 'actions.0.type': 'grasp'},
                {'actions': {'$size': 2}, 'actions.0.type': 'grasp', 'actions.1.type': 'place'},
            ]},
            projection={'_id': 0, 'id': 1, 'actions.pose': 1, 'actions.reward': 1, 'actions.images': 1}
        )
        train_episodes, validation_episodes = self.split_set(episodes)

        train_set = PlacingDataset(train_episodes, seed=42)
        train_data = train_set.get_data(shuffle='all')
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

        image_grasp_before = [
            tk.Input(shape=self.image_shape['ed'], name='image_grasp_before')
        ]
        image_place_before = [
            tk.Input(shape=self.image_shape['ed'], name='image_place_before')
        ]
        image_place_goal = [
            tk.Input(shape=self.image_shape['ed'], name='image_place_goal')
        ]

        reward_m, *z_m = self.grasp_model(image_grasp_before)
        reward_p, z_p = self.place_model(image_place_before + image_place_goal)
        reward = self.merge_model([z_m[0], z_p])

        losses = Losses()

        self.combined = tk.Model(inputs=(image_grasp_before + image_place_before + image_place_goal), outputs=[reward_m, reward_p, reward])
        self.combined.summary()
        self.combined.compile(
            optimizer=tk.optimizers.Adam(learning_rate=1e-4),
            loss=losses.binary_crossentropy,
            loss_weights=[1.0, 1.0, 4.0],
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
            tk.callbacks.ReduceLROnPlateau(factor=0.2, verbose=1, patience=20, min_lr=5e-7),
            tf.keras.callbacks.TensorBoard(log_dir=str(self.model_path.parent / 'logs' / f'placing-{time()}')),
        ]

        if load_model:
            self.combined.load_weights(str(self.model_path))
            evaluation = self.combined.evaluate(validation_data, batch_size=validation_batch_size, verbose=2)
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
            tk.Input(shape=self.image_shape['ed'], name='image')
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.35)
        conv_block_r = conv_block_gen(l2_reg=0.001, dropout_rate=0.5)

        x = conv_block(inputs[0], 32)
        x = conv_block(x, 32, strides=(2, 2))
        x = conv_block(x, 32)

        x_r = conv_block_r(x, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 64)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 48, kernel_size=(2, 2))

        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 96)
        x = conv_block(x, 96)

        x = conv_block(x, 128)
        x = conv_block(x, 128, kernel_size=(2, 2))

        reward = tkl.Conv2D(number_primitives, kernel_size=(1, 1), activation='sigmoid', name='reward_grasp')(x_r)
        reward_training = tkl.Reshape((number_primitives,))(reward)

        z_trainings = []
        for i in range(1):
            z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.0005), name=f'z_m{i}')(x)
            z_training = tkl.Reshape((self.z_size,))(z)
            z_trainings.append(z_training)

        outputs = [reward_training] + z_trainings
        return tk.Model(inputs=inputs, outputs=outputs, name='grasp')

    def define_place_model(self):
        inputs = [
            tk.Input(shape=self.image_shape['ed'], name='image_before'),
            tk.Input(shape=self.image_shape['ed'], name='image_goal'),
        ]

        conv_block = conv_block_gen(l2_reg=0.001, dropout_rate=0.35)
        conv_block_r = conv_block_gen(l2_reg=0.001, dropout_rate=0.5)

        x = tkl.Concatenate()(inputs)

        x = conv_block(x, 32)
        x = conv_block(x, 32)

        x = conv_block(x, 32)
        x = conv_block(x, 32)
        x = conv_block(x, 32)
        x = conv_block(x, 32)

        x_r = conv_block_r(x, 32)
        x_r = conv_block_r(x_r, 32)

        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 48)
        x_r = conv_block_r(x_r, 48)

        x_r = conv_block_r(x_r, 64)
        x_r = conv_block_r(x_r, 48, kernel_size=(2, 2))

        x = conv_block(x, 48)
        x = conv_block(x, 48)

        x = conv_block(x, 64)
        x = conv_block(x, 64)
        x = conv_block(x, 64)
        x = conv_block(x, 64)

        x = conv_block(x, 96)
        x = conv_block(x, 96)

        x = conv_block(x, 128)
        x = conv_block(x, 128, kernel_size=(2, 2))

        reward = tkl.Conv2D(1, kernel_size=(1, 1), activation='sigmoid', name='reward_place')(x_r)
        reward_training = tkl.Reshape((1,))(reward)

        z = tkl.Conv2D(self.z_size, kernel_size=(1, 1), activity_regularizer=tk.regularizers.l2(0.0005), name='z_p')(x)
        z_training = tkl.Reshape((self.z_size,))(z)

        outputs = [reward_training, z_training]
        return tk.Model(inputs=inputs, outputs=outputs, name='place')

    def define_merge_model(self):
        input_shape = (self.z_size)

        z_m = tk.Input(shape=input_shape, name='z_m')
        z_p = tk.Input(shape=input_shape, name='z_p')

        dense_block = dense_block_gen(l2_reg=0.01, dropout_rate=0.2)
        x = z_m - z_p

        x = dense_block(x, 128)
        x = dense_block(x, 128)
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
        episodes = list(map(self.assign_set, data))[-13000:]

        train_episodes = list(filter(lambda x: not x['is_validation'], episodes))
        validation_episodes = list(filter(lambda x: x['is_validation'], episodes))

        if verbose > 0:
            logger.info(f'Train on {len(train_episodes)} episodes.')
            logger.info(f'Validate on {len(validation_episodes)} episodes.')

        return train_episodes, validation_episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline for pick-and-place.')
    parser.add_argument('-d', '--collection', action='append', dest='collection', type=str, default=['placing-3'])
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
