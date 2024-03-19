import numpy as np
import torch
import torch.nn as nn
import gym
import random
import os
from copy import deepcopy
from typing import Dict
from offlinerlkit.policy import BasePolicy
import re
from glob import glob
from collections import deque


class IQLPolicy(BasePolicy):
    """
    Implicit Q-Learning <Ref: https://arxiv.org/abs/2110.06169>
    """

    def __init__(
        self,
        args,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_q1_optim: torch.optim.Optimizer,
        critic_q2_optim: torch.optim.Optimizer,
        critic_v_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        expectile: float = 0.8,
        temperature: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.args = args
        self.actor = actor
        self.critic_q1, self.critic_q1_old = critic_q1, deepcopy(critic_q1)
        self.critic_q2, self.critic_q2_old = critic_q2, deepcopy(critic_q2)
        self.critic_q1_old.eval()
        self.critic_q2_old.eval()
        self.critic_v = critic_v

        # self.critic_q1.name = "critic_q1"
        # self.critic_q2.name = "critic_q2"
        # self.critic_q1_old.name = "critic_q1_old"
        # self.critic_q2_old.name = "critic_q2_old"
        # self.critic_v.name = "critic_v"

        if hasattr(self.actor, 'anchor_handler'):
            if hasattr(self.critic_q1_old, 'anchor_handler'):
                del self.critic_q1_old.anchor_handler
                self.critic_q1_old.anchor_handler = self.actor.anchor_handler
            if hasattr(self.critic_q2_old, 'anchor_handler'):
                del self.critic_q2_old.anchor_handler
                self.critic_q2_old.anchor_handler = self.actor.anchor_handler
            for obj in [self.critic_q1, self.critic_q2, self.critic_q1_old, self.critic_q2_old, self.critic_v]:
                if hasattr(obj, 'anchor_handler'):
                    assert obj.anchor_handler is self.actor.anchor_handler

        self.actor_optim = actor_optim
        self.critic_q1_optim = critic_q1_optim
        self.critic_q2_optim = critic_q2_optim
        self.critic_v_optim = critic_v_optim

        self.action_space = action_space
        self._tau = tau
        self._gamma = gamma
        self._expectile = expectile
        self._temperature = temperature
        self.device = torch.device(device)

        self.anchor_sharing = True

    def train(self) -> None:
        self.actor.train()
        self.critic_q1.train()
        self.critic_q2.train()
        self.critic_v.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_q1.eval()
        self.critic_q2.eval()
        self.critic_v.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic_q1_old.parameters(), self.critic_q1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic_q2_old.parameters(), self.critic_q2.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                action = dist.mode().cpu().numpy()
            else:
                action = dist.sample().cpu().numpy()
        action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        return action

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self._expectile, (1 - self._expectile))
        return weight * (diff**2)

    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update value net
        with torch.no_grad():
            if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
            q1 = self.critic_q1_old(obss, actions) # obss anchor computed
            if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
            q2 = self.critic_q2_old(obss, actions) # obss anchor reused
            q = torch.min(q1, q2)
        v = self.critic_v(obss) # obss anchor reused
        critic_v_loss = self._expectile_regression(q-v).mean()
        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        self.critic_v_optim.step()

        # update actor (moved for anchor sharing)
        with torch.no_grad():
            q1, q2 = self.critic_q1_old(obss, actions), self.critic_q2_old(obss, actions) # obss anchor reused
            q = torch.min(q1, q2)
            v = self.critic_v(obss) # obss anchor reused
            exp_a = torch.exp((q - v) * self._temperature)
            exp_a = torch.clip(exp_a, None, 100.0)
        dist = self.actor(obss) # obss anchor reused
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_a * log_probs).mean()

        # update critic
        q1, q2 = self.critic_q1(obss, actions), self.critic_q2(obss, actions) # obss anchor reused
        if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
        with torch.no_grad():
            next_v = self.critic_v(next_obss)
            target_q = rewards + self._gamma * (1 - terminals) * next_v

        critic_q1_loss = ((q1 - target_q).pow(2)).mean()
        critic_q2_loss = ((q2 - target_q).pow(2)).mean()

        self.critic_q1_optim.zero_grad()
        critic_q1_loss.backward()
        self.critic_q1_optim.step()

        self.critic_q2_optim.zero_grad()
        critic_q2_loss.backward()
        self.critic_q2_optim.step()

        # update actor (cont'd)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self._sync_weight()

        return {
            "loss/actor": actor_loss.item(),
            "loss/q1": critic_q1_loss.item(),
            "loss/q2": critic_q2_loss.item(),
            "loss/v": critic_v_loss.item()
        }

    def save(self, save_path: str, random_states: dict, epoch, logger, lr_scheduler=None, last_10_performance=None) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data = dict(
            state_dict = self.state_dict(),
            random_states = random_states,

            actor = self.actor.state_dict(),
            critic_q1 = self.critic_q1.state_dict(),
            critic_q2 = self.critic_q2.state_dict(),
            critic_v = self.critic_v.state_dict(),

            actor_optim = self.actor_optim.state_dict(),
            critic_q1_optim = self.critic_q1_optim.state_dict(),
            critic_q2_optim = self.critic_q2_optim.state_dict(),
            critic_v_optim = self.critic_v_optim.state_dict(),

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
        self.critic_q1.load_state_dict(data['critic_q1'])
        self.critic_q2.load_state_dict(data['critic_q2'])
        self.critic_v.load_state_dict(data['critic_v'])

        # Load state_dict for optimizers
        self.actor_optim.load_state_dict(data['actor_optim'])
        self.critic_q1_optim.load_state_dict(data['critic_q1_optim'])
        self.critic_q2_optim.load_state_dict(data['critic_q2_optim'])
        self.critic_v_optim.load_state_dict(data['critic_v_optim'])

        # Load epoch
        self._epoch = data['epoch'] + 1

        last_10_performance = deque(data['last_10_performance'], maxlen=10)
        logger.log(f"[Epoch: {self._epoch}] Loaded policy from: {load_path}")

        policy_trainer._start_epoch = self._epoch
        policy_trainer.lr_scheduler = lr_scheduler if lr_scheduler is not None else None
        policy_trainer.last_10_performance = last_10_performance


