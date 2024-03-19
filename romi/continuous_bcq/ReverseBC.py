'''
From https://github.com/wenzhe-li/romi
'''
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Dict, List, Tuple, Union
from .BCQ import VAE
from collections import defaultdict

from offlinerlkit.policy import BasePolicy

class ReverseBC(BasePolicy):
    def __init__(self, args, dynamics, reverse_dynamics, state_dim, action_dim, max_action, device, entropy_weight=0.5, lr=1e-3):
        super().__init__()
        self.args = args
        latent_dim = action_dim * 2
        self.reverse_dynamics = reverse_dynamics
        self.dynamics = dynamics
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device

        self.entropy_weight = entropy_weight

    def actforward(self, state):
        action = self.vae.decode(state)
        return action

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action = self.vae.decode(state)
        return action.cpu().data.numpy()

    def train(self) -> None:
        self.vae.train()

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # Variational Auto-Encoder Training
        recon, mean, std = self.vae(next_obss, actions)
        recon_loss = F.mse_loss(recon, actions)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()

        # step loss
        rollout_transitions, _ = self.rollout(next_obss, 1) # rollout_transitions = (rollout_len * batch, obj_dim)
        with torch.no_grad():
            anchors, _, _, _ = self.dynamics.step(rollout_transitions['obss'], rollout_transitions['actions']) # normal dynamics
        step_loss = F.mse_loss(anchors, rollout_transitions['next_obss'])


        vae_loss = recon_loss + self.entropy_weight * KL_loss + self.args.step_weight * step_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        return {
            "loss/vae": vae_loss.detach(),
            "loss/recon": recon_loss.detach(),
            "loss/KL": KL_loss.detach(),
            "loss/step": step_loss.detach(),
        }

    def save(self, save_path: str, random_states: dict) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = dict(
            state_dict = self.state_dict(),
            random_states = random_states,
        )
        torch.save(data, os.path.join(save_path, "policy.pth"))

    def load(self, load_path):
        load_dict = torch.load(os.path.join(load_path, "policy.pth"))
        if 'random_states' in load_dict.keys():
            load_dict = load_dict['state_dict']
        self.load_state_dict(load_dict)


    def eval(self) -> None:
        self.vae.eval()

    @torch.no_grad()
    def rollout(
        self,
        init_obss: Union[np.ndarray, torch.Tensor],
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        # numpy version
        if type(init_obss) == np.ndarray:
            num_transitions = 0
            rewards_arr = np.array([])
            rollout_transitions = defaultdict(list)

            # rollout
            next_observations = init_obss
            for _ in range(rollout_length):
                actions = self.select_action(next_observations)
                with torch.no_grad():
                    observations, rewards, terminals, info = self.reverse_dynamics.step(next_observations, actions)
                rollout_transitions["obss"].append(observations)
                rollout_transitions["next_obss"].append(next_observations)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)
                rollout_transitions["terminals"].append(terminals)

                num_transitions += len(next_observations)
                rewards_arr = np.append(rewards_arr, rewards.flatten())

                nonterm_mask = (~terminals).flatten()
                if nonterm_mask.sum() == 0:
                    break

                next_observations = observations[nonterm_mask]

            for k, v in rollout_transitions.items():
                rollout_transitions[k] = np.concatenate(v, axis=0)

            return rollout_transitions, \
                {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean(), "reward_std": rewards_arr.std()}
        # tensor version
        else:
            num_transitions = 0
            rewards_arr = torch.tensor([], device=self.device)
            rollout_transitions = defaultdict(list)

            # rollout
            next_observations = init_obss
            for _ in range(rollout_length):
                with torch.no_grad():
                    actions = self.actforward(next_observations)
                with torch.no_grad():
                    observations, rewards, terminals, info = self.reverse_dynamics.step(next_observations, actions)
                rollout_transitions["obss"].append(observations)
                rollout_transitions["next_obss"].append(next_observations)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)
                rollout_transitions["terminals"].append(terminals)

                num_transitions += len(next_observations)
                rewards_arr = torch.cat((rewards_arr, rewards.reshape(-1)), dim=0)

                nonterm_mask = (~terminals).reshape(-1)
                if nonterm_mask.sum() == 0:
                    break

                next_observations = observations[nonterm_mask]

            for k, v in rollout_transitions.items():
                # print('k:', k)
                # import ipdb; ipdb.set_trace(context=10)
                rollout_transitions[k] = torch.cat(v, dim=0)
            # import ipdb
            # ipdb.set_trace(context=10)
            # print('rewards_arr:', rewards_arr)
            # print('rewards_arr.mean:', rewards_arr.mean())
            return rollout_transitions, \
                {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean().item(), "reward_std": rewards_arr.std().item()}

