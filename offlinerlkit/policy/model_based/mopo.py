import os
import numpy as np
import torch
import torch.nn as nn
import gym
from glob import glob
import re
import random
from typing import Dict, Union, Tuple
from collections import defaultdict
from offlinerlkit.policy import SACPolicy
from offlinerlkit.dynamics import EnsembleDynamics
from collections import deque

class MOPOPolicy(SACPolicy):
    """
    Model-based Offline Policy Optimization <Ref: https://arxiv.org/abs/2005.13239>
    """

    def __init__(
        self,
        args,
        dynamics: EnsembleDynamics,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )
        self.args = args
        self.dynamics = dynamics
        self.device = torch.device(device)

        self.anchor_sharing = True

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
            observations = init_obss
            for _ in range(rollout_length):
                actions = super().select_action(observations)
                with torch.no_grad():
                    next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
                rollout_transitions["obss"].append(observations)
                rollout_transitions["next_obss"].append(next_observations)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)
                rollout_transitions["terminals"].append(terminals)

                num_transitions += len(observations)
                rewards_arr = np.append(rewards_arr, rewards.flatten())

                nonterm_mask = (~terminals).flatten()
                if nonterm_mask.sum() == 0:
                    break

                observations = next_observations[nonterm_mask]

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
            observations = init_obss
            for _ in range(rollout_length):
                # actions = super().select_action(observations)
                actions, _ = super().actforward(observations)

                next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
                rollout_transitions["obss"].append(observations)
                rollout_transitions["next_obss"].append(next_observations)
                rollout_transitions["actions"].append(actions)
                rollout_transitions["rewards"].append(rewards)
                rollout_transitions["terminals"].append(terminals)

                num_transitions += len(observations)
                rewards_arr = torch.cat((rewards_arr, rewards.reshape(-1)), dim=0)

                nonterm_mask = (~terminals).reshape(-1)
                if nonterm_mask.sum() == 0:
                    break

                observations = next_observations[nonterm_mask]

            for k, v in rollout_transitions.items():
                rollout_transitions[k] = torch.cat(v, dim=0)
            return rollout_transitions, \
                {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean().item(), "reward_std": rewards_arr.std().item()}

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        return super().select_action(obs, deterministic)

    def learn(self, batch: Dict) -> Dict[str, float]:
        if len(batch.keys()) == 2: # realbuffer & fakebuffer
            real_batch, fake_batch = batch["real"], batch["fake"]
            mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}
        else: # only realbuffer
            mix_batch = batch["real"]

        return super().learn(mix_batch)

    def save(self, save_path: str, random_states: dict, epoch, logger, lr_scheduler=None, last_10_performance=None) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = dict(
            state_dict = self.state_dict(),
            random_states = random_states,

            actor = self.actor.state_dict(),
            critic1 = self.critic1.state_dict(),
            critic2 = self.critic2.state_dict(),

            alpha = self._alpha,
            log_alpha = self._log_alpha,
            alpha_optim = self.alpha_optim.state_dict(),

            actor_optim = self.actor_optim.state_dict(),
            critic1_optim = self.critic1_optim.state_dict(),
            critic2_optim = self.critic2_optim.state_dict(),

            lr_scheduler = lr_scheduler.state_dict() if lr_scheduler else None,
            epoch=epoch,
            last_10_performance = list(last_10_performance)
        )
        torch.save(data, os.path.join(save_path, f"policy_{epoch:04}.pth"))
        logger.log(f"[Epoch: {epoch}] Saved policy to: {save_path}")

        # remove old file(old epoch)
        ckpt_files = glob(os.path.join(save_path, "policy_*.pth"))
        for file in ckpt_files:
            match = re.search(r'policy_(\d+).pth$', file)
            if match:
                file_epoch = int(match.group(1))
                if file_epoch < epoch:
                    os.remove(file)
                    logger.log(f"[Epoch: {epoch}] Removed policy_{file_epoch:04}.pth")


    def load(self, load_path: str, logger, lr_scheduler, policy_trainer) -> None:
        # Load the entire data dictionary
        data = torch.load(load_path, map_location=self.device)

        # Load state_dict for the main model
        self.load_state_dict(data['state_dict'])

        # Load random states
        random.setstate(data['random_states']["random"])
        np.random.set_state(data['random_states']["np"])
        torch.set_rng_state(torch.ByteTensor(data['random_states']["torch"].cpu()))
        torch.cuda.set_rng_state_all([torch.ByteTensor(t.cpu()) for t in data['random_states']["torch_cuda"]])
        # eval_env.np_random.bit_generator.state = data['random_states']["eval_envs"]

        # Load state_dict for actors and critics
        self.actor.load_state_dict(data['actor'])
        self.critic1.load_state_dict(data['critic1'])
        self.critic2.load_state_dict(data['critic2'])

        # alpha
        self._alpha = data['alpha']
        self._log_alpha = data['log_alpha']
        self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=self.args.alpha_lr)
        self.alpha_optim.load_state_dict(data['alpha_optim'])

        # Load state_dict for optimizers
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critic1_optim.load_state_dict(data['critic1_optim'])
        self.critic2_optim.load_state_dict(data['critic2_optim'])

        # Load epoch
        self._epoch = data['epoch'] + 1

        # Load last performance
        last_10_performance = deque(data['last_10_performance'], maxlen=10)
        logger.log(f"[Epoch: {self._epoch}] Loaded policy from: {load_path}")

        policy_trainer._start_epoch = self._epoch
        policy_trainer.lr_scheduler = lr_scheduler if lr_scheduler is not None else None
        policy_trainer.last_10_performance = last_10_performance

    def set_horizon_length(self, actor_horizon_len, critic_horizon_len):
        self.actor.horizon_length = actor_horizon_len
        self.critic1.horizon_length = critic_horizon_len
        self.critic2.horizon_length = critic_horizon_len
        self.critic1_old.horizon_length = critic_horizon_len
        self.critic2_old.horizon_length = critic_horizon_len

    def set_closest_obs_sample_size(self, closest_obs_sample_size):
        self.actor.closest_obs_sample_size = closest_obs_sample_size
        self.critic1.closest_obs_sample_size = closest_obs_sample_size
        self.critic2.closest_obs_sample_size = closest_obs_sample_size
        self.critic1_old.closest_obs_sample_size = closest_obs_sample_size
        self.critic2_old.closest_obs_sample_size = closest_obs_sample_size

    def anchor_seeking_policy_freeze(self):
        self.actor.anchor_seeking_policy.freeze()
        self.critic1.anchor_seeking_policy.freeze()
        self.critic2.anchor_seeking_policy.freeze()
        self.critic1_old.anchor_seeking_policy.freeze()
        self.critic2_old.anchor_seeking_policy.freeze()

    def anchor_seeking_policy_unfreeze(self):
        self.actor.anchor_seeking_policy.unfreeze()
        self.critic1.anchor_seeking_policy.unfreeze()
        self.critic2.anchor_seeking_policy.unfreeze()
        self.critic1_old.anchor_seeking_policy.unfreeze()
        self.critic2_old.anchor_seeking_policy.unfreeze()


