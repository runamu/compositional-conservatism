import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.nets.mlp import MLP
from offlinerlkit.modules.dist_module import TanhNormalWrapper
import einops
import random
import faiss
from scipy.spatial import KDTree
from tqdm import tqdm
import math

class AnchorHandler(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.obs_std = None
        self.observations = None
        self._size = None
        self.anchors = None
        self.store_anchors = False
        self.reuse_anchors = False

        self.sample_size = self.args.closest_obs_sample_size

    def set_dataset(self, observations):
        self.observations = observations
        # self.observations = torch.tensor(observations).to(device=self.device, non_blocking=True)
        # print(self.observations.untyped_storage().nbytes())

        self._size = len(observations)
        if self.args.anchor_mode == 'top_10_d':
            self.set_train_deltas()

    def set_obs_std(self, obs_std_np, obs_std):
        self.obs_std_np = obs_std_np + 1e-3
        self.obs_std = obs_std + 1e-3

    def set_obs_mean(self, obs_mean_np, obs_mean):
        self.obs_mean_np = obs_mean_np
        self.obs_mean = obs_mean

    def set_train_deltas(self):
        if hasattr(self, 'train_deltas'):
            return
        sample_indexes_1 = np.random.choice(self._size, size=self._size, replace=False) # similar to replay buffer sampling
        sample_indexes_2 = np.random.choice(self._size, size=self._size, replace=False) # similar to replay buffer sampling
        self.train_deltas = self.observations[sample_indexes_1] - self.observations[sample_indexes_2]

    # @torch.compile
    def get_rollout_obs(self, obss, horizon_length, anchor_seeking_policy, dynamics):
        if self.reuse_anchors:
            # print('reuse anchors') # debug
            return self.anchors

        anchors = obss #### anchor_seeker_architecture = 'simple'
        for _ in range(horizon_length):
            action = anchor_seeking_policy(anchors)
            next_anchors, _, _, _ = dynamics.step(anchors, action, deterministic=True)
            anchors = next_anchors

        if self.store_anchors:
            # print('store anchors') # debug
            self.anchors = anchors
        return anchors

    def get_random_obs(self, obss):
        sample_indexes = np.random.choice(self._size, size=len(obss), replace=False) # similar to replay buffer sampling
        anchors = self.observations[sample_indexes]
        anchors = torch.tensor(anchors).to(device=self.device, non_blocking=True, dtype=torch.float32)
        assert anchors.shape == obss.shape, f'anchors shape: {anchors.shape}, obss shape: {obss.shape}'

        return anchors

    def get_noised_obs(self, eps, obss):
        B,D = obss.shape
        noises = 2 * eps * self.obs_std * (torch.rand(B, D, device=self.device) - 0.5)
        obss = obss + noises
        return obss

    def normalise_obs(self):
        self.norm_observations = (self.observations - self.obs_mean_np) / (self.obs_std_np)

    @torch.no_grad()
    # @profile
    def get_top_10pct_closest_delta(self, obss):
        '''
            This should be analogous to the method in the Aviv's paper.
            Refer to `choose_goal` in Aviv's code.
        '''
        if self.reuse_anchors:
            return self.anchors

        tensor_flag = False
        if not type(obss) == np.ndarray:
            obss = obss.cpu().detach().numpy()
            tensor_flag = True

        sample_size = self.sample_size
        delta_sample_indexes = np.random.randint(0, self._size, size=sample_size) # similar to replay buffer sampling
        train_delta_sample = self.train_deltas[delta_sample_indexes]
        index = faiss.IndexFlatL2(train_delta_sample.shape[1])
        index.add(train_delta_sample.astype('float32'))

        ############ naive ############
        # import ipdb; ipdb.set_trace()
        # anchors = []
        # for obs in obss:
        #     obs_sample_indexes = np.random.choice(self._size, size=sample_size, replace=False) # similar to replay buffer sampling
        #     curr_deltas = obs - self.observations[obs_sample_indexes]
        #     min_distances, _ = index.search(curr_deltas.astype('float32'), k=1)
        #     closest_idx = random.choice(np.argsort(min_distances)[:(len(curr_deltas)//10)])
        #     anchors.append(self.observations[obs_sample_indexes[closest_idx]])
        # anchors = np.array(anchors)

        ########### batched ############
        batch_size = 32
        if  len(obss) % 32 == 0:
            batch_size = 32
        elif len(obss) % 100 == 0:
            batch_size = 100
        else:
            batch_size = len(obss)
        assert len(obss) % batch_size == 0, f'len(obss): {len(obss)}, batch_size: {batch_size}'
        num_batches = math.ceil(len(obss) / batch_size)

        anchors = []
        obs_sample_indexes = np.random.randint(0, self._size, size=sample_size * num_batches)
        flat_shape = (-1, self.observations.shape[-1])
        # for i in tqdm(range(num_batches)):
        for i in range(num_batches):
            batch_obss = obss[i * batch_size: (i+1) * batch_size]
            batch_deltas = batch_obss[:, np.newaxis] - self.observations[obs_sample_indexes[i * sample_size: (i+1) * sample_size]]
            batch_deltas = batch_deltas.astype('float32').reshape(flat_shape)

            # Note the change from k=1 to k=batch_size
            min_distances, _ = index.search(batch_deltas, k=1)
            min_distances = min_distances.reshape(batch_size, sample_size)
            closest_idxs = np.argmin(min_distances, axis=1)

            batch_anchors = [self.observations[obs_sample_indexes[i * sample_size + idx]] for idx in closest_idxs]
            anchors.extend(batch_anchors)
        anchors = np.array(anchors)

        if tensor_flag:
            anchors = torch.tensor(anchors).to(device=self.device, non_blocking=True, dtype=torch.float32)
        assert anchors.shape == obss.shape, f'anchors shape: {anchors.shape}, obss shape: {obss.shape}'
        if self.store_anchors:
            self.anchors = anchors

        return anchors

    def toggle_on_store(self):
        self.reuse_anchors = False
        self.store_anchors = True

    def toggle_on_reuse(self):
        self.reuse_anchors = True
        self.store_anchors = False

    def toggle_off(self):
        self.reuse_anchors = False
        self.store_anchors = False
        self.anchors = None