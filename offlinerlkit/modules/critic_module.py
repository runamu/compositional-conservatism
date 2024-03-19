import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional, List, Tuple
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.nets import MLP, EnsembleMLP, EnsembleMLP, pytorch_init, uniform_init
from offlinerlkit.modules.actor_module import AnchorActor
import einops
from torch.utils.checkpoint import checkpoint
from offlinerlkit.utils.anchor_handler import AnchorHandler
from typing import Dict
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values

# for COCOA
class TransdCritic(nn.Module):
    def __init__(
            self,
            args,
            anchor_seeking_policy: AnchorActor,
            anchor_handler: AnchorHandler,
            dynamics: BaseDynamics,
            backbone: Union[nn.Module,List[nn.Module]],
            input_dim: int,
            obs_dim: int,
            action_dim: int,
            fg_output_dim: int,
            embedding_dim: int,
            horizon_length: int,
            tr_hidden_dims: Union[List[int], Tuple[int]],
            activation: nn.Module = nn.ReLU,
            layernorm: bool = False,
            device: str = "cpu") -> None:
        super().__init__()
        self.args = args
        self.anchor_seeking_policy = anchor_seeking_policy
        self.anchor_handler = anchor_handler
        self.dynamics = dynamics
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim
        self.horizon_length = horizon_length
        self.tr_hidden_dims = tr_hidden_dims
        self.device = torch.device(device)
        self.anchor_mode = args.anchor_mode
        self.closest_obs_sample_size = args.closest_obs_sample_size

        self.backbone = backbone.to(device)
        self.latent_dim = getattr(self.backbone, "output_dim")
        self.last = nn.Linear(self.latent_dim, 1).to(device)
        self.transd_dim = tr_hidden_dims[-1]

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

        self.name = None

    # # batchwise operation
    # def anchor_seeking_step(self, init_obs, anchor, actions, rollout_length):
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
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)

        if self.anchor_mode == 'rollout':
            anchor = self.anchor_handler.get_rollout_obs(obs, self.horizon_length, self.anchor_seeking_policy, self.dynamics) # for anchor sharing
        elif self.anchor_mode == 'top_10_d':
            anchor = self.anchor_handler.get_top_10pct_closest_delta(obs)
        delta = obs - anchor

        if actions is not None:
            delta = torch.cat([delta, actions], dim=1)
            anchor = torch.cat([anchor, actions], dim=1)
            feature = self.fg_model(delta, anchor)
        else:
            # iql: critic_v
            feature = self.fg_model(delta, anchor)

        feature = self.backbone(feature)
        values = self.last(feature)

        return values
