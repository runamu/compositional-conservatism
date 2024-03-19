import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
import ipdb
import wandb
from torch.utils.checkpoint import checkpoint


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        args,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler

        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.track = args.track
        self.args = args

    # @ torch.no_grad()
    # @torch.compile
    def step(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        "imagine single forward step"
        # numpy mode
        if type(obs) == np.ndarray and type(action) == np.ndarray:
            obs_act = np.concatenate([obs, action], axis=-1)
            obs_act = self.scaler.transform(obs_act)
            mean, logvar = self.model(obs_act)
            mean = mean.cpu().numpy()
            logvar = logvar.cpu().numpy()
            mean[..., :-1] += obs
            std = np.sqrt(np.exp(logvar))

            ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)
            # choose one model from ensemble
            num_models, batch_size, _ = ensemble_samples.shape
            model_idxs = self.model.random_elite_idxs(batch_size)
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]

            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
            terminal = self.terminal_fn(obs, action, next_obs)
            info = {}
            info["raw_reward"] = reward
            info['uncertainty'] = torch.sqrt(mean[..., :-1].var(0, unbiased=False).mean(1))
            if self._penalty_coef:

                if self._uncertainty_mode == "aleatoric":
                    penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
                elif self._uncertainty_mode == "pairwise-diff":
                    next_obses_mean = mean[..., :-1]
                    next_obs_mean = np.mean(next_obses_mean, axis=0)
                    diff = next_obses_mean - next_obs_mean
                    penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
                elif self._uncertainty_mode == "ensemble_std":
                    next_obses_mean = mean[..., :-1]
                    penalty = np.sqrt(next_obses_mean.var(0).mean(1))
                else:
                    raise ValueError
                penalty = np.expand_dims(penalty, 1).astype(np.float32)
                assert penalty.shape == reward.shape
                reward = reward - self._penalty_coef * penalty
                info["penalty"] = penalty

            else: # combo case

                next_obses_mean = mean[..., :-1]
                penalty = np.sqrt(next_obses_mean.var(0).mean(1))
                penalty = np.expand_dims(penalty, 1).astype(np.float32)
                info["penalty"] = penalty


            return next_obs, reward, terminal, info

        # tensor mode
        else:
            obs_act = torch.cat([obs, action], dim=-1)
            obs_act = self.scaler.transform(obs_act) # dont need to change into transform_tensor
            mean, logvar = self.model(obs_act)
            mean = torch.cat((mean[..., :-1] + obs, mean[..., -1:]), dim=-1)
            std = torch.sqrt(torch.exp(logvar))
            if deterministic:

                ensemble_samples = mean.float().to(self.args.device)
                elite_idxs = self.model.elites # sorted by holdout loss, ascending

                # model_idxs = elite_idxs[:5] # choose top 5 models
                samples = torch.index_select(ensemble_samples, dim=0, index=elite_idxs).mean(0)

            else:
                ensemble_samples = (mean + torch.randn(mean.size(), device=self.args.device) * std).float()
                # choose one model from ensemble
                num_models, batch_size, _ = ensemble_samples.size()
                model_idxs = self.model.random_elite_idxs(batch_size)
                samples = ensemble_samples[model_idxs, np.arange(batch_size)]

            next_obs = samples[..., :-1]
            reward = samples[..., -1:]
            terminal = self.terminal_fn(obs, action, next_obs)
            info = {}
            info["raw_reward"] = reward
            info['uncertainty'] = torch.sqrt(mean[..., :-1].var(0, unbiased=False).mean(1))
            if self._penalty_coef is not None:
                if self._penalty_coef:
                    if self._uncertainty_mode == "aleatoric":
                        penalty = torch.max(torch.linalg.norm(std, dim=2), dim=0)[0]
                    elif self._uncertainty_mode == "pairwise-diff":
                        next_obses_mean = mean[..., :-1]
                        next_obs_mean = torch.mean(next_obses_mean, dim=0)
                        diff = next_obses_mean - next_obs_mean
                        penalty = torch.max(torch.linalg.norm(diff, dim=2), dim=0)[0]
                    elif self._uncertainty_mode == "ensemble_std":
                        next_obses_mean = mean[..., :-1]
                        penalty = torch.sqrt(next_obses_mean.var(0, unbiased=False).mean(1))
                    else:
                        raise ValueError
                    penalty = torch.unsqueeze(penalty, 1).float()
                    assert penalty.size() == reward.size()
                    reward = reward - self._penalty_coef * penalty
                    info["penalty"] = penalty

                else: # combo case

                    next_obses_mean = mean[..., :-1]
                    penalty = torch.sqrt(next_obses_mean.var(0, unbiased=False).mean(1))
                    penalty = torch.unsqueeze(penalty, 1).float()
                    info["penalty"] = penalty

            return next_obs, reward, terminal, info

    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]

        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:

        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]
        self.scaler.fit(train_inputs)

        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        best_metric = None
        checkpoint_last = os.path.join(os.path.dirname(logger.checkpoint_dir), "checkpoint_last")
        checkpoint_best = os.path.join(os.path.dirname(logger.checkpoint_dir), "checkpoint_best")

        logger.log("Training dynamics:")
        while True:
            epoch += 1
            logger.log(f'epoch: {epoch}')
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef) # target - mean
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss

            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1

            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

            # save last checkpoint
            indexes = self.select_elites(holdout_losses)
            self.model.set_elites(indexes)
            self.model.load_save()
            self.save(checkpoint_last)

            # save best checkpoint
            if best_metric is None or holdout_loss < best_metric:
                best_metric = holdout_loss
                best_epoch = epoch
                self.save(checkpoint_best)

            if self.track:
                wandb.log({"train/dynamics/epoch": epoch}, commit=False)
                wandb.log({"train/dynamics/best_epoch": best_epoch}, commit=False)
                wandb.log({"train/dynamics/best_metric": best_metric})

        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
        if self.track:
            wandb.log({"holdout_loss": (np.sort(holdout_losses)[:self.model.num_elites]).mean()})

        return {"holdout_loss": (np.sort(holdout_losses)[:self.model.num_elites]).mean()}

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.tensor(targets_batch, device=self.model.device)
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.

            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))# if variance is low, contribution is big to make a more certain prediction.
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)

    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)


    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
