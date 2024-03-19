import numpy as np
import torch
import torch.nn as nn
import gym
from glob import glob
import re
import random
import os
from typing import Dict, Union, Tuple
from copy import deepcopy
from collections import defaultdict
from offlinerlkit.policy import BasePolicy
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.buffer import ReplayBuffer
from collections import deque


class MOBILEPolicy(BasePolicy):
    """
    Model-Bellman Inconsistancy Penalized Offline Reinforcement Learning
    """

    def __init__(
        self,
        args,
        real_buffer: ReplayBuffer,
        dynamics: EnsembleDynamics,
        actor: nn.Module,
        critics: nn.ModuleList,
        actor_optim: torch.optim.Optimizer,
        critics_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        penalty_coef: float = 1.0,
        num_samples: int = 10,
        deterministic_backup: bool = False,
        max_q_backup: bool = False,
        device: str = "cpu",
    ) -> None:

        super().__init__()
        self.real_buffer = real_buffer
        self.dynamics = dynamics
        self._max_q_backup = max_q_backup
        self.actor = actor
        self.critics = critics
        self.critics_old = deepcopy(critics)
        self.critics_old.eval()
        self.device = torch.device(device)

        for i, module in enumerate(self.critics):
            module.name = f"critic{i+1}"
        for i, module in enumerate(self.critics_old):
            module.name = f"critic{i+1}_old"

        if hasattr(self.actor, 'anchor_handler'):
            for critic_old in self.critics_old:
                if hasattr(critic_old, 'anchor_handler'):
                    del critic_old.anchor_handler
                    critic_old.anchor_handler = self.actor.anchor_handler
            for obj in self.critics + self.critics_old:
                if hasattr(obj, 'anchor_handler'):
                    assert obj.anchor_handler is self.actor.anchor_handler



        self.actor_optim = actor_optim
        self.critics_optim = critics_optim

        self._tau = tau
        self._gamma = gamma
        self.args = args
        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

        self._penalty_coef = penalty_coef
        self._num_samples = num_samples
        self._deteterministic_backup = deterministic_backup

        self.anchor_sharing = True

    def train(self) -> None:
        self.actor.train()
        self.critics.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critics.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critics_old.parameters(), self.critics.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def actforward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)

        return squashed_action, log_prob

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            with torch.no_grad():
                action, _ = self.actforward(obs, deterministic)
                action = action.cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            action, _ = self.actforward(obs, deterministic)
        return action

    #@torch.no_grad()
    def rollout(
        self,
        init_obss: np.ndarray,
        rollout_length: int
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        if type(init_obss) == np.ndarray:
            num_transitions = 0
            rewards_arr = np.array([])
            rollout_transitions = defaultdict(list)

            # rollout
            observations = init_obss
            for _ in range(rollout_length):
                actions = self.select_action(observations)

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
                {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

        # tensor version
        else:
            num_transitions = 0
            rewards_arr = torch.tensor([], device=self.args.device)
            rollout_transitions = defaultdict(list)

            # rollout
            observations = init_obss
            for _ in range(rollout_length):
                # actions = super().select_action(observations)
                with torch.no_grad():
                    actions = self.select_action(observations)
                with torch.no_grad():
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

    @ torch.no_grad()
    def compute_lcb(self, obss: torch.Tensor, actions: torch.Tensor):
        # compute next q std
        pred_next_obss = self.dynamics.sample_next_obss(obss, actions, self._num_samples)
        num_samples, num_ensembles, batch_size, obs_dim = pred_next_obss.shape
        pred_next_obss = pred_next_obss.reshape(-1, obs_dim)
        #if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
        pred_next_actions, _ = self.actforward(pred_next_obss) # pred_next_obss anchor computed
        #if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()

        pred_next_qs =  torch.cat([critic_old(pred_next_obss, pred_next_actions) for critic_old in self.critics_old], 1) # next_obss anchor reused
        #if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
        pred_next_qs = torch.min(pred_next_qs, 1)[0].reshape(num_samples, num_ensembles, batch_size, 1)
        penalty = pred_next_qs.mean(0).std(0)

        return penalty

    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], 0) for k in real_batch.keys()}

        obss, actions, next_obss, rewards, terminals = mix_batch["observations"], mix_batch["actions"], mix_batch["next_observations"], mix_batch["rewards"], mix_batch["terminals"]
        batch_size = obss.shape[0]

        # update critic
        qs = torch.stack([critic(obss, actions) for critic in self.critics], 0)
        with torch.no_grad():
            penalty = self.compute_lcb(obss, actions)
            penalty[:len(real_batch["rewards"])] = 0.0

            if self._max_q_backup:
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, 10, 1) \
                    .view(batch_size * 10, next_obss.shape[-1])
                if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
                tmp_next_qs = torch.cat([critic_old(tmp_next_obss, tmp_next_actions) for critic_old in self.critics_old], 1)
                if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
                tmp_next_qs = tmp_next_qs.view(batch_size, 10, len(self.critics_old)).max(1)[0].view(-1, len(self.critics_old))
                next_q = torch.min(tmp_next_qs, 1)[0].reshape(-1, 1)
            else:
                if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
                next_actions, next_log_probs = self.actforward(next_obss) # next_obss anchor computed
                if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
                next_qs = torch.cat([critic_old(next_obss, next_actions) for critic_old in self.critics_old], 1) # next_obss anchor reused
                if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
                next_q = torch.min(next_qs, 1)[0].reshape(-1, 1)
                if not self._deteterministic_backup:
                    next_q -= self._alpha * next_log_probs
            target_q = (rewards - self._penalty_coef * penalty) + self._gamma * (1 - terminals) * next_q
            target_q = torch.clamp(target_q, 0, None)

        critic_loss = ((qs - target_q) ** 2).mean()
        self.critics_optim.zero_grad()
        critic_loss.backward()
        self.critics_optim.step()

        # update actor
        if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
        a, log_probs = self.actforward(obss) # obss anchor computed
        if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
        qas = torch.cat([critic(obss, a) for critic in self.critics], 1) # obss anchor reused
        if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
        actor_loss = -torch.min(qas, 1)[0].mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self._sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic": critic_loss.item()
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()

        return result

    def set_horizon_length(self, actor_horizon_len, critic_horizon_len):
        self.actor.horizon_length = actor_horizon_len
        for critic in self.critics:
            critic.horizon_length = critic_horizon_len
        for critic_old in self.critics_old:
            critic_old.horizon_length = critic_horizon_len

    def set_closest_obs_sample_size(self, closest_obs_sample_size):
        self.actor.closest_obs_sample_size = closest_obs_sample_size
        for critic in self.critics:
            critic.closest_obs_sample_size = closest_obs_sample_size
        for critic_old in self.critics_old:
            critic_old.closest_obs_sample_size = closest_obs_sample_size

    def anchor_seeking_policy_freeze(self):
        self.actor.anchor_seeking_policy.freeze()
        for critic in self.critics:
            critic.anchor_seeking_policy.freeze()
        for critic_old in self.critics_old:
            critic_old.anchor_seeking_policy.freeze()

    def anchor_seeking_policy_unfreeze(self):
        self.actor.anchor_seeking_policy.unfreeze()
        for critic in self.critics:
            critic.anchor_seeking_policy.unfreeze()
        for critic_old in self.critics_old:
            critic_old.anchor_seeking_policy.unfreeze()

    def save(self, save_path: str, random_states: dict, epoch, logger, lr_scheduler=None, last_10_performance=None) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        assert lr_scheduler is not None, "lr_scheduler should not be None"

        data = dict(
            state_dict = self.state_dict(),
            random_states = random_states,

            actor = self.actor.state_dict(),
            critics = self.critics.state_dict(),

            alpha = self._alpha,
            log_alpha = self._log_alpha,
            alpha_optim = self.alpha_optim.state_dict(),

            actor_optim = self.actor_optim.state_dict(),
            critics_optim = self.critics_optim.state_dict(),

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

        # Load state_dict for actors and critics
        self.actor.load_state_dict(data['actor'])
        self.critics.load_state_dict(data['critics'])

        # alpha
        self._alpha = data['alpha']
        self._log_alpha = data['log_alpha']
        self.alpha_optim = torch.optim.Adam([self._log_alpha], lr=self.args.alpha_lr)
        self.alpha_optim.load_state_dict(data['alpha_optim'])

        # Load state_dict for optimizers
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critics_optim.load_state_dict(data['critics_optim'])

        lr_scheduler.load_state_dict(data['lr_scheduler'])

        # Load epoch
        self._epoch = data['epoch'] + 1

        # Load last performance
        last_10_performance = deque(data['last_10_performance'], maxlen=10)
        logger.log(f"[Epoch: {self._epoch}]Loaded policy from: {load_path}")

        policy_trainer._start_epoch = self._epoch
        policy_trainer.lr_scheduler = lr_scheduler if lr_scheduler is not None else None
        policy_trainer.last_10_performance = last_10_performance

