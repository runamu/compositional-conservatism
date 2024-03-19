import numpy as np
import torch
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import BasePolicy
from offlinerlkit.utils.set_state_fns import get_set_state_fn
import faiss

class DivergentPolicy(BasePolicy):
    def __init__(self, args, env, dynamics, reverse_dynamics, buffer, action_dim, max_action,
        scale_coef,
        noise_coef,
        device,
        seed=None
    ):
        super().__init__()
        self.args = args
        self.env = env
        self.dynamics = dynamics
        self.reverse_dynamics = reverse_dynamics
        self.buffer = buffer
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)
        self.directions_dict = defaultdict(lambda: None)
        self.scale_coef = scale_coef
        self.noise_coef = noise_coef
        self.action_mode = 'sample'

        self.set_state_fn = get_set_state_fn(task=self.args.task)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def actforward(self,
        obs_batch: torch.Tensor,
        deterministic: bool = False,
        *,
        prev_actions: Union[np.ndarray, torch.Tensor] = None,
        divergent: bool = True,
    ):
        batch_size, _ = obs_batch.shape
        if not divergent:
            assert self.action_mode == 'sample'
            sampled_actions = self.buffer.sample(batch_size, keys=['actions'])['actions']
            return sampled_actions

        # Define directions for each observation in the batch
        directions = torch.zeros((batch_size, self.action_dim), device=self.device)
        if self.action_mode == 'gaussian':
            directions = torch.normal(0, 1, size=(batch_size, self.action_dim), device=self.device)
            directions = directions / torch.norm(directions, dim=1, keepdims=True) * 0.5
        elif self.action_mode in ['sample', 'dissim']:
            if self.action_mode == 'sample':
                sampled_actions = self.buffer.sample(batch_size, keys=['actions'])['actions']
            elif self.action_mode == 'dissim':
                if prev_actions is not None:
                    assert obs_batch.shape[0] == prev_actions.shape[0]
                    sample_size = 10000
                    candidate_actions = self.buffer.sample(sample_size, keys=['actions'])['actions']

                    # faiss method - cosine similarity
                    index = faiss.IndexFlatIP(candidate_actions.shape[1])
                    candidate_actions_np = candidate_actions.cpu().numpy().astype('float32')
                    prev_actions_np = prev_actions.cpu().numpy().astype('float32')
                    faiss.normalize_L2(candidate_actions_np)
                    index.add(candidate_actions_np)
                    _, closest_indices = index.search(-prev_actions_np, k=1)
                    sampled_actions = candidate_actions[closest_indices]

            directions = torch.zeros((batch_size, self.action_dim), device=self.device)
            for i, obs in enumerate(obs_batch):
                obs_data_ptr = obs.data_ptr()
                direction = self.directions_dict[obs_data_ptr]
                if direction is None:
                    # direction = torch.tensor(sampled_actions[i], device=self.device)
                    direction = sampled_actions[i]
                    self.directions_dict[obs_data_ptr] = direction
                directions[i] = direction

        # Sample actions around the directions
        actions = self.sample_actions(directions, deterministic)

        return actions

    def select_action(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            actions = self.actforward(obs_batch, deterministic)
        return actions.cpu().data.numpy()

    def sample_actions(self, directions: torch.Tensor, deterministic: bool) -> np.ndarray:
        batch_size, _ = directions.shape

        if deterministic:
            return directions
        else:
            actions = directions * self.scale_coef + torch.normal(0, self.noise_coef, size=(batch_size, self.action_dim), device=self.device)
            return torch.clamp(actions, -self.max_action, self.max_action)

    def clear_directions(self):
        self.directions_dict = defaultdict(lambda: None)

    def learn(self, batch: Dict) -> Dict[str, float]:
        print("DivergentPolicy does not require learning")

    def save(self, save_path: str, random_states: dict) -> None:
        print("DivergentPolicy does not require saving")

    def load(self, load_path):
        print("DivergentPolicy does not require loading")

    def rollout(
        self,
        init_obss: Union[np.ndarray, torch.Tensor],
        rollout_length: int,
        prev_actions: Union[np.ndarray, torch.Tensor] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        # tensor version only
        num_transitions = 0
        rewards_arr = torch.tensor([], device=self.device)
        rollout_transitions = defaultdict(list)

        # rollout
        next_observations = init_obss
        for _ in range(rollout_length):
            with torch.no_grad():
                actions = self.actforward(init_obss, prev_actions=prev_actions) # always move in the same direction with some noises
            with torch.no_grad():
                observations, rewards, terminals, info = self.reverse_dynamics.step(next_observations, actions) # deterministic=False
            rollout_transitions["obss"].append(observations)
            rollout_transitions["next_obss"].append(next_observations)
            rollout_transitions["actions"].append(actions)
            rollout_transitions["rewards"].append(rewards if self.reverse_dynamics.model._with_reward else torch.zeros(actions.shape[0], 1, device=self.device))
            rollout_transitions["terminals"].append(terminals)

            num_transitions += len(next_observations)
            rewards_arr = torch.cat((rewards_arr, rewards.reshape(-1)), dim=0) if self.reverse_dynamics.model._with_reward else None

            nonterm_mask = (~terminals).reshape(-1)
            if nonterm_mask.sum() == 0:
                break

            next_observations = observations[nonterm_mask]
        self.clear_directions()

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = torch.cat(v, dim=0)

        return rollout_transitions, \
            {"num_transitions": num_transitions,
                "reward_mean": rewards_arr.mean() if self.reverse_dynamics.model._with_reward else 0,
                "reward_std": rewards_arr.std() if self.reverse_dynamics.model._with_reward else 0
                }
