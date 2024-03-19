import time
import os

import numpy as np
import torch
import gym
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Union
from tqdm import tqdm
from collections import deque
from offlinerlkit.modules.actor_module import AnchorActor
from offlinerlkit.dynamics import BaseDynamics, ReverseEnsembleDynamics
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
from offlinerlkit.utils.util_fns import get_normalized_std_score
from offlinerlkit.policy import BasePolicy, DivergentPolicy
import random
from copy import deepcopy
import math
import pickle

# reverse policy trainer
class AnchorSeekerTrainer:
    def __init__(
        self,
        args,
        anchor_seeking_policy: AnchorActor,
        reverse_dynamics: ReverseEnsembleDynamics,
        dynamics: BaseDynamics,
        policy: Union[BasePolicy, DivergentPolicy],
        eval_env: gym.Env,
        real_buffer: ReplayBuffer,
        fake_buffer: ReplayBuffer,
        logger: Logger,
        rollout_setting: Tuple[int, int, int],
        epoch: int = 1000,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:

        self.args = args
        self.policy = policy
        self.anchor_seeking_policy = anchor_seeking_policy
        self.reverse_dynamics = reverse_dynamics
        self.dynamics = dynamics
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger
        self.track = args.track
        self._rollout_epoch, self._rollout_batch_size, \
            self._rollout_length = rollout_setting

        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler
        self.num_timesteps = None

    def train(
        self,
        max_epochs_since_update: int = 5,
    ) -> Dict[str, float]:
        start_time = time.time()

        self.num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        old_loss = 1e10
        cnt = 0
        checkpoint_last = os.path.join(os.path.dirname(self.logger.checkpoint_dir), "checkpoint_last")

        assert self.args.load_dynamics_path, "load_dynamics_path should be True for evaluation"
        # train loop
        for e in range(1, self._epoch + 1):
            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self._epoch}")
            for it in pbar:
                batch = self.real_buffer.sample(self._batch_size)
                loss = self.policy.learn(batch)
                # pbar.set_postfix(**loss)

                for k, v in loss.items():
                    self.logger.logkv_mean(k, v)

                self.num_timesteps += 1

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            vae_loss = loss["loss/vae"].item()

            # early stopping (chatgpt)
            new_loss = vae_loss
            improvement = (old_loss - new_loss) / old_loss
            old_loss = new_loss  # always update old_loss
            if abs(improvement) > 0.001:  # stop when the loss converges
                cnt = 0
            else:
                cnt += 1
            if cnt >= max_epochs_since_update:
                break

            # save random state
            random_states = {}
            random_states["random"] = random.getstate()
            random_states["np"] = np.random.get_state() # dictionary
            random_states["torch"] = torch.get_rng_state() # Tensor
            random_states["torch_cuda"] = torch.cuda.get_rng_state_all() # List[Tensor]
            random_states["eval_envs"] = self.eval_env.np_random.bit_generator.state

            # save checkpoint
            if e % 10 == 0:
                self.policy.save(checkpoint_last, random_states=random_states)

            self._evaluate()
            # save checkpoint
            torch.save(self.policy.state_dict(), os.path.join(self.logger.checkpoint_dir, "policy.pth"))

        self.logger.log("total time: {:.2f}s".format(time.time() - start_time))
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))

        return {"last_10_performance": np.mean(last_10_performance)}

    @torch.no_grad()
    def _evaluate(self) -> None:
        batch = self.real_buffer.sample(self._batch_size * 100)
        for eval_rollout_length in [1,3,5]:
            loss = 0
            # only test for 1 step is enough I think, no way to test more minutely
            init_obss = batch["observations"]
            rollout_transitions, _ = self.policy.rollout(init_obss, eval_rollout_length) # rollout_transitions = (rollout_len * batch, obj_dim)

            anchors, _, _, _ = self.dynamics.step(rollout_transitions['obss'], rollout_transitions['actions']) # normal dynamics

            loss = F.mse_loss(anchors, rollout_transitions['next_obss']) / eval_rollout_length
            self.logger.log(f"eval/loss/rollout_len_{eval_rollout_length}: {loss:.4f}")

    def generate(self) -> None:
        self.fake_buffer.reset()
        dataset = self.real_buffer.sample_all()# # numpy
        init_obss = torch.tensor(dataset['next_observations'], device=self.args.device)
        prev_actions = torch.tensor(dataset['actions'], device=self.args.device)
        assert len(init_obss) == self.real_buffer._max_size, f'len(init_obss): {len(init_obss)}, self.real_buffer._max_size: {self.real_buffer._max_size}'

        for i in tqdm(range(math.ceil(self.real_buffer._max_size/self._batch_size))):
            init_obs = init_obss[i*self._batch_size : (i+1)*self._batch_size]
            prev_action = prev_actions[i*self._batch_size : (i+1)*self._batch_size]
            rollout_transitions, rollout_info = self.policy.rollout(init_obs, self._rollout_length, prev_action)
            self.fake_buffer.add_batch(**rollout_transitions)
            # self.logger.log(
            #     "num rollout transitions: {}, reward mean: {:.4f}, reward std: {:.4f}".\
            #         format(rollout_info["num_transitions"], rollout_info["reward_mean"], rollout_info["reward_std"]) )
            for _key, _value in rollout_info.items():
                self.logger.logkv_mean("rollout_info/"+_key, _value)
        self.logger.dumpkvs()
        self.fake_buffer.save(path=self.logger.result_dir, data=self.fake_buffer.sample_all())
        self.logger.log(f'Fake buffer saved successfully in {self.logger.result_dir}')
        return self.logger.result_dir

    def anchor_seeker_pretrain_reverse(self, load_reverse_imagination_path, n_epoch, batch_size, lr, logger: Logger, data=None) -> None:
        assert self.anchor_seeking_policy is not None

        # only tensor version
        if load_reverse_imagination_path is not None:
            path = os.path.join(load_reverse_imagination_path, "reverse_imagination.pkl")
            with open(path, 'rb') as f:
                data = pickle.load(f)
        else:
            assert data != None, f"data should be given if load_reverse_imagination_path is {data}"
        observations = torch.tensor(data["observations"],device=self.args.device) # keys = observations(legacy)
        actions = torch.tensor(data["actions"], device=self.args.device)
        sample_num = observations.shape[0]

        self.dynamics.model.eval()
        self.dynamics.model.requires_grad_(False)
        self.anchor_seeking_policy.train()
        optimizer = torch.optim.Adam(self.anchor_seeking_policy.parameters(), lr=lr)
        logger.log("Pretraining anchor seeker")

        old_loss = 1e10
        cnt = 0
        max_epochs_since_update = 5
        for epoch in range(n_epoch):
            accumulated_loss = 0
            for i_batch in range(sample_num // batch_size):
                batch_obs = observations[i_batch * batch_size: (i_batch + 1) * batch_size]
                batch_actions = actions[i_batch * batch_size: (i_batch + 1) * batch_size]
                pred_actions = self.anchor_seeking_policy(batch_obs)
                loss = (((batch_actions - pred_actions) ** 2).mean()) # pred_actions = (batch_size, time_step, action_dim)
                optimizer.zero_grad()
                loss.backward()
                # make_dot(loss).view()
                optimizer.step()

                accumulated_loss += loss.cpu().item()

            logger.logkv(f"train/anchor_seeking_loss",  accumulated_loss/(sample_num // batch_size))
            logger.set_timestep(epoch)
            logger.dumpkvs()

            # early stopping
            new_loss = accumulated_loss
            improvement = (old_loss - new_loss) / old_loss
            old_loss = new_loss  # always update old_loss
            if abs(improvement) > 0.005:  # stop when the loss converges
                cnt = 0
            else:
                cnt += 1
            if cnt >= max_epochs_since_update:
                break

        random_states = {}
        random_states["np"] = np.random.get_state() # dictionary
        random_states["torch"] = torch.get_rng_state() # Tensor
        random_states["torch_cuda"] = torch.cuda.get_rng_state_all() # List[Tensor]

        self.anchor_seeking_policy.save(logger.model_dir, random_states) # "anchor_seeker_pretrain.pth"
        logger.log(f"Saved actor anchor seeker in Dir: {os.path.join(logger.model_dir)}")
