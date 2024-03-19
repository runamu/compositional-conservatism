import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Union, Tuple
from offlinerlkit.policy import BasePolicy

class SACPolicy(BasePolicy):
    """
    Soft Actor Critic <Ref: https://arxiv.org/abs/1801.01290>
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
    ) -> None:
        super().__init__()
        self.trace_grad_norm = False
        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)

        # self.critic1.name = "critic1"
        # self.critic2.name = "critic2"
        # self.critic1_old.name = "critic1_old"
        # self.critic2_old.name = "critic2_old"

        self.critic1_old.eval()
        self.critic2_old.eval()

        self.actor_optim = actor_optim
        self.critic1_optim = critic1_optim
        self.critic2_optim = critic2_optim

        if hasattr(self.actor, 'anchor_handler'):
            if hasattr(self.critic1_old, 'anchor_handler'):
                del self.critic1_old.anchor_handler
                self.critic1_old.anchor_handler = self.actor.anchor_handler
            if hasattr(self.critic2_old, 'anchor_handler'):
                del self.critic2_old.anchor_handler
                self.critic2_old.anchor_handler = self.actor.anchor_handler
            for obj in [self.critic1, self.critic2, self.critic1_old, self.critic2_old]:
                if hasattr(obj, 'anchor_handler'):
                    assert obj.anchor_handler is self.actor.anchor_handler

            self.anchor_sharing = True

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self._alpha = alpha

    def train(self) -> None:
        self.actor.train()
        self.critic1.train()
        self.critic2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()

    def _sync_weight(self) -> None:
        for o, n in zip(self.critic1_old.parameters(), self.critic1.parameters()):
            o.data.copy_(o.data * (1.0 - self._tau) + n.data * self._tau)
        for o, n in zip(self.critic2_old.parameters(), self.critic2.parameters()):
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
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
        with torch.no_grad():
            action, _ = self.actforward(obs, deterministic)
        return action.cpu().numpy()

    def learn(self, batch: Dict) -> Dict[str, float]:
        result =  {}

        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], \
            batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        with torch.no_grad():
            if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
            next_actions, next_log_probs = self.actforward(next_obss) # next_obss anchor computed
            if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
            next_q = torch.min(
                self.critic1_old(next_obss, next_actions), self.critic2_old(next_obss, next_actions) # next_obss anchor reused
            ) - self._alpha * next_log_probs
            target_q = rewards + self._gamma * (1 - terminals) * next_q
            if self.anchor_sharing: self.actor.anchor_handler.toggle_off()

        if self.anchor_sharing: self.actor.anchor_handler.toggle_on_store()
        q1 = self.critic1(obss, actions) # obss anchor computed
        if self.anchor_sharing: self.actor.anchor_handler.toggle_on_reuse()
        q2 = self.critic2(obss, actions) # obss anchor reused

        critic1_loss = ((q1 - target_q).pow(2)).mean()
        result["loss/critic1"] = critic1_loss.detach()
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        critic2_loss = ((q2 - target_q).pow(2)).mean()
        result["loss/critic2"] = critic2_loss.detach()
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        # update actor
        a, log_probs = self.actforward(obss) # obss anchor reused

        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a) # obss anchor reused
        result["train/actor/target_Q"] = torch.min(q1a, q2a).mean().detach() # torch.min(q1a, q2a) should be increased

        if self.anchor_sharing: self.actor.anchor_handler.toggle_off()
        actor_loss = - torch.min(q1a, q2a).mean() + self._alpha * log_probs.mean()
        result["loss/actor"] = actor_loss.detach()

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

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.detach()
            result["alpha"] = self._alpha.detach()

        self._sync_weight()

        return result

