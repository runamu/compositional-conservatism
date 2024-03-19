import numpy as np
import torch
from typing import Tuple, Dict
import pickle
import os
from glob import glob

class ReplayBuffer:
    def __init__(
        self,
        args: Dict,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.args = args
        self._ptr = 0
        self._size = 0


        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        if self.args.algo_name == 'rebrac': # rebrac
            self.next_actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
            self.mc_returns = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def reset(self) -> None:
        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=self.action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        if self.args.algo_name == 'rebrac': # rebrac
            self.next_actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
            self.mc_returns = np.zeros((self._max_size, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:

        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size
        if isinstance(obss, torch.Tensor):
            self.observations[indexes] = obss.cpu().detach().numpy().copy()
            self.next_observations[indexes] =next_obss.cpu().detach().numpy().copy()
            self.actions[indexes] = actions.cpu().detach().numpy().copy()
            self.rewards[indexes] = rewards.cpu().detach().numpy().copy()
            self.terminals[indexes] = terminals.cpu().detach().numpy().copy()
        else: #
            self.observations[indexes] = obss.copy()
            self.next_observations[indexes] =next_obss.copy()
            self.actions[indexes] = actions.copy()
            self.rewards[indexes] = rewards.copy()
            self.terminals[indexes] = terminals.copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def save(self, path: str, data: list, filename="reverse_imagination.pkl", key=['observations', 'actions', 'next_observations', 'terminals', 'rewards']) -> None:
        path = os.path.join(path, filename)
        data = {k: data[k] for k in key} if len(key) != 5 else data
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_dataset_from_path(self, path: Dict[str, np.ndarray], filename="reverse_imagination.pkl") -> None:
        path = os.path.join(path, filename)
        with open(path, 'rb') as f:
            loaded_data = pickle.load(f)
        self.load_dataset(loaded_data)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        if self.args.algo_name == 'rebrac': # adroit
            self.next_actions = np.array(dataset["next_actions"], dtype=self.action_dtype)
            self.mc_returns = np.array(dataset["mc_returns"], dtype=np.float32)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        if 'infos/step' in dataset.keys():
            self.steps = np.array(dataset["infos/step"], dtype=np.float32).reshape(-1, 1)

        self._ptr = len(observations)
        self._size = len(observations)

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def normalize_obs_with_params(self, obs_mean, obs_std):
        self.observations = (self.observations - obs_mean) / obs_std
        self.next_observations = (self.next_observations - obs_mean) / obs_std

    def sample(self, batch_size: int, keys=['observations', 'actions', 'next_observations', 'terminals', 'rewards']) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)

        return {
            k: torch.tensor(getattr(self, k)[batch_indexes]).to(device=self.device, non_blocking=True)
            for k in keys
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        dataset = {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy(),
            "infos/step": self.steps[:self._size].copy() if hasattr(self, 'steps') else None,
        }
        return dataset

    def save_replay_buffer(self, path: str, epoch: int, logger):
        # path = '../record/'
        filename = f'replay_buffer_idem_{epoch}.pkl'
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(self, f)
        # logger.log(f"[Saved] [{filename}] replay buffer to path:  {full_path}")

    def load_replay_buffer(self, path: str, epoch: int, logger):
        ## path = '../record/'
        filename = f'replay_buffer_idem_{epoch}.pkl'
        full_path = os.path.join(path, filename)
        with open(full_path, 'rb') as f:
            loaded_instance = pickle.load(f)

        for attr, value in loaded_instance.__dict__.items():
            setattr(self, attr, value)
        logger.log(f"[Loaded] [{filename}] replay buffer from path:  {full_path}")

    def delete_replay_buffer(self, path: str, epoch: int, logger):
        all_files = glob(os.path.join(path, 'replay_buffer_idem_*.pkl'))
        epochs = [int(os.path.basename(file).split('_')[-1].split('.')[0]) for file in all_files]
        largest_epoch = max(epochs, default=None)
        # deleted_file = None
        for file, epoch in zip(all_files, epochs):
            if epoch != largest_epoch:
                os.remove(file)
        #         deleted_file = file
        # logger.log(f"[Deleted] {deleted_file} replay buffer")