import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Dict, Optional, Union
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger
import math
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from offlinerlkit.utils.termination_fns import termination_fn_dummy
import torch.nn.functional as F
from time import time

class ReverseEnsembleDynamics(EnsembleDynamics):
    def __init__(
        self,
        args,
        dynamics: EnsembleDynamics,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
    ) -> None:
        # super().__init__(model, optim)
        super().__init__(
            args,
            model,
            optim,
            scaler,
            terminal_fn,
            penalty_coef,
            uncertainty_mode,
        )
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.track = args.track
        self.args = args
        self.dynamics = dynamics


    def step(
        self,
        next_obs: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict], Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]]:
        "imagine single backward step"
        # tensor mode only
        next_obs_act = torch.cat([next_obs, action], dim=-1)
        next_obs_act = self.scaler.transform(next_obs_act) # dont need to change into transform_tensor
        mean, logvar = self.model(next_obs_act)
        mean = torch.cat((mean[..., :-1] + next_obs, mean[..., -1:]), dim=-1) if self.model._with_reward else mean + next_obs
        std = torch.sqrt(torch.exp(logvar))
        if deterministic:
            ensemble_samples = mean.float()
            elite_idxs = self.model.elites # sorted by holdout loss, ascending
            # model_idxs = elite_idxs[:5] # choose top 5 models
            samples = torch.index_select(ensemble_samples, dim=0, index=elite_idxs).mean(0)

        else:
            ensemble_samples = (mean + torch.randn(mean.size(), device=self.args.device) * std).float()
            # choose one model from ensemble
            num_models, batch_size, _ = ensemble_samples.size()
            model_idxs = self.model.random_elite_idxs(batch_size)
            samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        obs = samples[..., :-1]  if self.model._with_reward else samples
        reward = samples[..., -1:] if self.model._with_reward else None
        terminal = termination_fn_dummy(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward
        info['uncertainty'] = torch.sqrt(mean[..., :-1].var(0, unbiased=False).mean(1)) if self.model._with_reward else torch.sqrt(mean.var(0, unbiased=False).mean(1))

        return obs, reward, terminal, info

    # from romi
    def format_reverse_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]

        delta_obss =  obss - next_obss
        inputs = np.concatenate((next_obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1) if self.model._with_reward else delta_obss
        return inputs, targets


    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
    ) -> None:

        inputs, targets = self.format_reverse_samples_for_training(data)

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

        logger.log("Training reverse dynamics:")
        while True:
            epoch += 1
            logger.log(f'epoch: {epoch}')
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef) # target - mean
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/reverse_dynamics_train_loss", train_loss)
            logger.logkv("loss/reverse_dynamics_holdout_loss", holdout_loss)
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
                self.save(checkpoint_best)

        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))

        return {"holdout_loss": (np.sort(holdout_losses)[:self.model.num_elites]).mean()}

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
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


    def align(
        self,
        fakebuffer: ReplayBuffer,
        logger: Logger,
        batch_size: int = 10000,
        epoch:int = 50,
    ):

        self.model.train()
        for param in self.dynamics.model.parameters():
            param.requires_grad = False

        ### step loss
        assert not self.model._with_reward
        data = fakebuffer.sample_all()
        temps = torch.tensor(data['observations'], device=self.args.device)
        obss = torch.tensor(data['next_observations'], device=self.args.device)
        actions = torch.tensor(data['actions'], device=self.args.device)

        for e in range(1, epoch+1):
            start_time = time()
            pbar = tqdm(range(math.ceil(fakebuffer._max_size/batch_size)), desc=f"Epoch #{e}/{epoch}")

            loss_temp_sum = 0
            loss_sum = 0
            for i in pbar:
                temp = temps[i*batch_size:(i+1)*batch_size]
                batch_obs = obss[i*batch_size:(i+1)*batch_size] # (batch_size, obs_dim)
                batch_action = actions[i*batch_size:(i+1)*batch_size] # (batch_size, action_dim)
                s_star, _, _, _ = self.step(batch_obs, batch_action, deterministic=True)
                s_hat, _, _, _ = self.dynamics.step(s_star, batch_action, deterministic=True)

                loss_temp = F.mse_loss(temp, s_star) # how deos the reverse_dynamics changed
                loss = F.mse_loss(s_hat, batch_obs)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.item()
                loss_temp_sum += loss_temp.item()

            logger.logkv(f"epoch", e)
            logger.logkv(f"{e} epoch training time", round(time() - start_time, 2))
            logger.logkv(f"loss", (loss_sum/len(pbar)))
            logger.logkv(f"loss_temp", (loss_temp_sum/len(pbar)))
            logger.dumpkvs()

        # save reverse dynamics
        self.save(logger._model_dir, logger)

    def save(self, save_path: str, logger: Logger=None) -> None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if logger:
            torch.save(self.model.state_dict(), os.path.join(save_path, "aligned_reverse_dynamics.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(save_path, "reverse_dynamics.pth"))
        if logger:
            logger.log(f'reverse_dynamics.pth is saved successfully in {save_path}')
        self.scaler.save_scaler(save_path)


    def load(self, load_path: str, mode=None) -> None:
        if mode =='aligned':
            self.model.load_state_dict(torch.load(os.path.join(load_path, "aligned_reverse_dynamics.pth"), map_location=self.model.device))
            self.scaler.load_scaler(load_path)
        else:
            self.model.load_state_dict(torch.load(os.path.join(load_path, "reverse_dynamics.pth"), map_location=self.model.device))
            self.scaler.load_scaler(load_path)


