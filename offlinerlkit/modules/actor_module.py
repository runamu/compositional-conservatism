import numpy as np
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.nets.mlp import MLP, pytorch_init, uniform_init
from offlinerlkit.modules.dist_module import TanhNormalWrapper
from offlinerlkit.utils.anchor_handler import AnchorHandler
import einops
from typing import Dict
import torch.nn.functional as F

# for SAC
class ActorProb(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        dist_net: nn.Module,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        self.dist_net = dist_net.to(device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist


# for TD3
class Actor(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim, device=device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions

# for COCOA
class AnchorActor(nn.Module):
    def __init__(
        self,
        args,
        backbone: nn.Module,
        owner=None,
    ) -> None:
        super().__init__()

        self.owner = owner
        output_dim = args.action_dim
        self._max = args.max_action
        self.device = torch.device(args.device)
        latent_dim = getattr(backbone, "output_dim")
        self.backbone = backbone.to(args.device)
        self.last = nn.Linear(latent_dim, output_dim, device=args.device)

    # batchwise operation
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logit = self.backbone(obs)
        action = self._max * torch.tanh(self.last(logit))
        return action

    def save(self, save_path: str, random_states: dict) -> None:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        data = dict(
            state_dict = self.state_dict(),
            random_states = random_states,
        )
        torch.save(data, os.path.join(save_path, "anchor_seeker_pretrain.pth"))

    def pretrain_load(self, path: str) -> None:
        file_path = os.path.join(path, "anchor_seeker_pretrain.pth")
        file_path = os.path.abspath(os.path.join("/", file_path)) # make path to abs path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"'{file_path}' does not exist!")
        self.load_state_dict(torch.load(file_path, map_location=self.device)['state_dict'])

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def reset_actor_hiddenstate(self):
        self.actor_h0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dims, device=self.device)
        self.actor_c0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dims, device=self.device)

    def reset_critic_hiddenstate(self):
        self.critic_h0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dims, device=self.device)
        self.critic_c0 = torch.zeros(self.lstm_num_layers, self.batch_size, self.hidden_dims, device=self.device)


# for COCOA
class TransdActorProb(nn.Module):
    def __init__(
        self,
        args,
        anchor_seeking_policy: AnchorActor,
        anchor_handler: AnchorHandler,
        dynamics: BaseDynamics,
        dist: nn.Module,
        input_dim: int,
        backbone: Union[nn.Module, List[nn.Module]],
        tr_hidden_dims: Union[List[int], Tuple[int]],
        fg_output_dim: int,
        embedding_dim: int,
        action_dim: int,
        activation: nn.Module = nn.ReLU,
        layernorm: bool = False,
        unbounded=False,
        conditioned_sigma=False,
        horizon_length=10,
        max_mu=1.0,
        sigma_min=-5.0,
        sigma_max=2.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.args = args
        self.anchor_seeking_policy = anchor_seeking_policy
        self.anchor_handler = anchor_handler
        self.dynamics = dynamics
        self.dist = dist.to(device)
        self.tr_hidden_dims = tr_hidden_dims
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.horizon_length = horizon_length
        self.device = torch.device(device)
        self.anchor_mode = args.anchor_mode
        self.closest_obs_sample_size = args.closest_obs_sample_size
        self.backbone = backbone.to(device)
        self.transd_dim = tr_hidden_dims[-1]

        # dist
        self._c_sigma = conditioned_sigma
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

        # transduction
        self.f_models = MLP(
                            input_dim=input_dim,
                            hidden_dims=tr_hidden_dims,
                            output_dim=fg_output_dim,
                            activation=activation,
                            dropout_rate=None,
                            layernorm=layernorm,
                        ).to(device)
        self.g_models = MLP(
                            input_dim=input_dim,
                            hidden_dims=tr_hidden_dims,
                            output_dim=fg_output_dim,
                            activation=activation,
                            dropout_rate=None,
                            layernorm=layernorm,
                        ).to(device)


    # # batchwise operation
    # def anchor_seeking_step(self, anchor):
    #     obs = anchor
    #     action = self.anchor_seeking_policy(obs)
    #     next_anchor, _, _, _ = self.dynamics.step(anchor, action, deterministic=True)
    #     return next_anchor

    def fg_model(self, delta, anchor):
        f_outputs = einops.rearrange(self.f_models(delta), 'b (t d) -> b t 1 d', t=self.transd_dim)
        g_outputs = einops.rearrange(self.g_models(anchor), 'b (t d) -> b t d 1', t=self.transd_dim)
        h_outputs = torch.matmul(f_outputs, g_outputs).squeeze(dim=(-2, -1))
        return h_outputs

    def forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)

        if self.anchor_mode == 'rollout':
            anchor = self.anchor_handler.get_rollout_obs(obs, self.horizon_length, self.anchor_seeking_policy, self.dynamics) # for anchor sharing
        elif self.anchor_mode == 'top_10_d':
            anchor = self.anchor_handler.get_top_10pct_closest_delta(obs)
        delta = obs - anchor

        feature = self.fg_model(delta, anchor)
        feature = self.backbone(feature)
        mu_and_sigma = self.dist(feature)
        if 'Tanh' in self.dist.__class__.__name__:
            assert mu_and_sigma.mode()[0].shape == (delta.shape[0], self.action_dim), f"action shape: {mu_and_sigma.mode()[0].shape}"
        else:
            assert mu_and_sigma.mode().shape == (delta.shape[0], self.action_dim), f"action shape: {mu_and_sigma.mode().shape}"
        return mu_and_sigma # already TanhNormalWrapper Class

