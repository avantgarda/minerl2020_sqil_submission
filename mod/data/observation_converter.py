"""
MIT License

Copyright (c) Preferred Networks, Inc.
"""
import copy
import numpy as np
import cv2
import gym
from collections import OrderedDict


class GrayScaleConverter:
    def __call__(self, observation):
        assert 'pov' in observation
        obs = copy.deepcopy(observation)
        orig_shape = obs['pov'].shape
        if len(obs['pov'].shape) == 3:
            obs['pov'] = cv2.cvtColor(obs['pov'], cv2.COLOR_RGB2GRAY)
        else:
            obs['pov'] = np.array(list(map(
                lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
                obs['pov'].reshape(-1, *orig_shape[-3:]))))
        obs['pov'] = obs['pov'].reshape(*orig_shape[:-1], 1)
        return obs

    def convert_space(self, space):
        new_space_dict = OrderedDict()
        for key in space.spaces:
            if key == 'pov':
                new_space_dict[key] = gym.spaces.Box(
                    low=0, high=255, shape=(*space['pov'].shape[:-1], 1),
                    dtype=np.uint8)
            else:
                new_space_dict[key] = space[key]
        return gym.spaces.Dict(new_space_dict)


class PoVOnlyConverter:
    def __call__(self, observation):
        assert 'pov' in observation
        assert 'vector' in observation
        return observation['pov']

    def convert_space(self, space):
        pov = space['pov']
        return gym.spaces.Box(
            low=0, high=255,
            shape=pov.shape,
            dtype=np.float32)


class VectorCombineConverter:
    def __call__(self, observation):
        assert 'pov' in observation
        assert 'vector' in observation
        scale = 1 / 255
        pov, vector = observation['pov'], observation['vector']
        num_elem = pov.shape[-3] * pov.shape[-2]
        vector_channel = np.tile((vector + 1) / 2, num_elem // vector.shape[-1]).reshape(*pov.shape[:-1], -1)  # noqa
        return np.concatenate([pov, vector_channel / scale], axis=-1)

    def convert_space(self, space):
        pov, vector = space['pov'], space['vector']
        num_new_channel = 1
        return gym.spaces.Box(
            low=0, high=255,
            shape=(*pov.shape[:-1], pov.shape[-1] + num_new_channel),
            dtype=np.float32)


class MoveAxisConverter:
    def __call__(self, observation):
        observation = np.moveaxis(observation, [-3, -2, -1], [-2, -1, -3])
        assert observation.shape[-2:] == (64, 64)
        return observation

    def convert_space(self, space):
        low = self.__call__(space.low)
        high = self.__call__(space.high)
        return gym.spaces.Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype)


class ScaledFloatConverter:
    scale = 1 / 255

    def __call__(self, observation):
        return (observation * self.scale).astype(np.float32)

    def convert_space(self, space):
        low = self.__call__(space.low)
        high = self.__call__(space.high)
        return gym.spaces.Box(
            low=(low * self.scale).astype(np.float32),
            high=(high * self.scale).astype(np.float32),
            shape=low.shape, dtype=np.float32)
