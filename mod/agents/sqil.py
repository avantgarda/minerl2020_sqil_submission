import copy
import collections
import time
import multiprocessing as mp
from logging import getLogger

import torch
import torch.nn.functional as F
import numpy as np

import pfrl
from pfrl import agent
from pfrl.utils.batch_states import batch_states
from pfrl.utils.contexts import evaluating
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.replay_buffer import batch_experiences
from pfrl.replay_buffer import batch_recurrent_experiences
from pfrl.replay_buffer import ReplayUpdater
from pfrl.utils.recurrent import get_recurrent_state_at
from pfrl.utils.recurrent import mask_recurrent_state_at
from pfrl.utils.recurrent import one_step_forward
from pfrl.utils.recurrent import pack_and_forward
from pfrl.utils.recurrent import recurrent_state_as_numpy


def _mean_or_nan(xs):
    """Return its mean a non-empty sequence, numpy.nan for a empty one."""
    return np.mean(xs) if xs else np.nan


def compute_value_loss(y, t, clip_delta=True, batch_accumulator="mean"):
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' or 'sum'. 'mean' will use the mean of
            the loss values in a batch. 'sum' will use the sum.
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        return F.smooth_l1_loss(y, t, reduction=batch_accumulator)
    else:
        return F.mse_loss(y, t, reduction=batch_accumulator) / 2


def compute_weighted_value_loss(
    y, t, weights, clip_delta=True, batch_accumulator="mean"
):
    """Compute a loss for value prediction problem.

    Args:
        y (torch.Tensor): Predicted values.
        t (torch.Tensor): Target values.
        weights (torch.Tensor): Weights for y, t.
        clip_delta (bool): Use the Huber loss function with delta=1 if set True.
        batch_accumulator (str): 'mean' will divide loss by batchsize
    Returns:
        (torch.Tensor) scalar loss
    """
    assert batch_accumulator in ("mean", "sum")
    y = y.reshape(-1, 1)
    t = t.reshape(-1, 1)
    if clip_delta:
        losses = F.smooth_l1_loss(y, t, reduction="none")
    else:
        losses = F.mse_loss(y, t, reduction="none") / 2
    losses = losses.reshape(-1,)
    weights = weights.to(losses.device)
    loss_sum = torch.sum(losses * weights)
    if batch_accumulator == "mean":
        loss = loss_sum / y.shape[0]
    elif batch_accumulator == "sum":
        loss = loss_sum
    return loss


def _batch_reset_recurrent_states_when_episodes_end(
    batch_done, batch_reset, recurrent_states
):
    """Reset recurrent states when episodes end.

    Args:
        batch_done (array-like of bool): True iff episodes are terminal.
        batch_reset (array-like of bool): True iff episodes will be reset.
        recurrent_states (object): Recurrent state.

    Returns:
        object: New recurrent states.
    """
    indices_that_ended = [
        i
        for i, (done, reset) in enumerate(zip(batch_done, batch_reset))
        if done or reset
    ]
    if indices_that_ended:
        return mask_recurrent_state_at(recurrent_states, indices_that_ended)
    else:
        return recurrent_states


def load_experiences_from_demonstrations(
        expert_dataset, batch_size, reward=1):
    if expert_dataset is None:
        raise ValueError("Expert dataset must be provided.")
    ret = []
    for _ in range(batch_size):
        ob, act, _, next_ob, done = expert_dataset.sample()
        ret.append([dict(
            state=ob,
            action=act,
            reward=reward,
            next_state=next_ob,
            next_action=None,
            is_state_terminal=done)])
    return ret


class RewardBasedSampler:
    # Sampling based on proportion of each subtask visited by agent.
    def __init__(self, expert_dataset, reward_boundaries,
                 max_buffer_size=1000, reward=1):
        self.expert_dataset = expert_dataset
        self.reward_boundaries = reward_boundaries
        self.replay_buffers = [
            pfrl.replay_buffers.ReplayBuffer(max_buffer_size, 1)
            for _ in range(len(reward_boundaries) + 1)]
        self.reward_scale = reward
        # Fill replay buffers
        while True:
            update_needed = False
            for rbuf in self.replay_buffers:
                if len(rbuf) < max_buffer_size:
                    update_needed = True
            if update_needed:
                self._update_buffer(max_buffer_size)
            else:
                break

    def _policy_index(self, ob):
        cum_reward = np.array(ob)[-1, 0, 0]
        return np.sum(
            cum_reward > self.reward_boundaries / self.reward_boundaries[-1] - 1e-8)

    def _update_buffer(self, n):
        for _ in range(n):
            ob, act, _, next_ob, done = self.expert_dataset.sample()
            self.replay_buffers[self._policy_index(ob)].append(
                state=ob,
                action=act,
                reward=self.reward_scale,
                next_state=next_ob,
                next_action=None,
                is_state_terminal=done)

    def sample(self, experiences):
        # update_samples
        self._update_buffer(len(experiences))
        n_samples = [0 for _ in range(len(self.reward_boundaries) + 1)]
        for frame in experiences:
            n_samples[self._policy_index(frame[0]['state'])] += 1
        ret = []
        for rbuf, n_sample in zip(self.replay_buffers, n_samples):
            samples = rbuf.sample(n_sample)
            for frame in samples:
                ret.append(frame)
        return ret


class SQIL(agent.AttributeSavingMixin, agent.BatchAgent):
    """Deep Q-Network algorithm.

    Args:
        q_function (StateQFunction): Q-function
        optimizer (Optimizer): Optimizer that is already setup
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        explorer (Explorer): Explorer that specifies an exploration strategy.
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        target_update_interval (int): Target model update interval in step
        clip_delta (bool): Clip delta if set True
        phi (callable): Feature extractor applied to observations
        target_update_method (str): 'hard' or 'soft'.
        soft_update_tau (float): Tau of soft target update.
        n_times_update (int): Number of repetition of update
        batch_accumulator (str): 'mean' or 'sum'
        episodic_update_len (int or None): Subsequences of this length are used
            for update if set int and episodic_update=True
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        recurrent (bool): If set to True, `model` is assumed to implement
            `pfrl.nn.Recurrent` and is updated in a recurrent
            manner.

        Changes from DQN:
            remove recurrent support
            add expert dataset
    """

    saved_attributes = ("model", "target_model", "optimizer")

    def __init__(
        self,
        q_function,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        gpu=None,
        replay_start_size=50000,
        minibatch_size=32,
        update_interval=1,
        target_update_interval=10000,
        clip_delta=True,
        phi=lambda x: x,
        target_update_method="hard",
        soft_update_tau=1e-2,
        n_times_update=1,
        batch_accumulator="mean",
        episodic_update_len=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        expert_dataset=None,
        reward_scale=1.0,
        experience_lambda=1.0,
        recurrent=False,
        reward_boundaries=None,  # specific to options
    ):
        self.expert_dataset = expert_dataset

        self.model = q_function

        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.gamma = gamma
        self.explorer = explorer
        self.gpu = gpu
        self.target_update_interval = target_update_interval
        self.clip_delta = clip_delta
        self.phi = phi
        self.target_update_method = target_update_method
        self.soft_update_tau = soft_update_tau
        self.batch_accumulator = batch_accumulator
        assert batch_accumulator in ("mean", "sum")
        self.logger = logger
        self.batch_states = batch_states
        self.recurrent = recurrent
        if self.recurrent:
            update_func = self.update_from_episodes
        else:
            update_func = self.update
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=update_func,
            batchsize=minibatch_size,
            episodic_update=recurrent,
            episodic_update_len=episodic_update_len,
            n_times_update=n_times_update,
            replay_start_size=replay_start_size,
            update_interval=update_interval,
        )
        self.minibatch_size = minibatch_size
        self.episodic_update_len = episodic_update_len
        self.replay_start_size = replay_start_size
        self.update_interval = update_interval

        assert (
            target_update_interval % update_interval == 0
        ), "target_update_interval should be a multiple of update_interval"

        # For imitation
        self.reward_scale = reward_scale
        self.experience_lambda = experience_lambda

        if reward_boundaries is not None and self.expert_dataset is not None:
            self.reward_based_sampler = RewardBasedSampler(
                self.expert_dataset,
                reward_boundaries,
                reward=reward_scale)
        else:
            self.reward_based_sampler = None

        self.t = 0
        self.optim_t = 0  # Compensate pytorch optim not having `t`
        self._cumulative_steps = 0
        self.last_state = None
        self.last_action = None
        self.target_model = None
        self.sync_target_network()

        # Statistics
        self.q_record = collections.deque(maxlen=1000)
        self.loss_record = collections.deque(maxlen=100)

        # Recurrent states of the model
        self.train_recurrent_states = None
        self.train_prev_recurrent_states = None
        self.test_recurrent_states = None

        # Error checking
        if (
            self.replay_buffer.capacity is not None
            and self.replay_buffer.capacity < self.replay_updater.replay_start_size
        ):
            raise ValueError("Replay start size cannot exceed replay buffer capacity.")

    @property
    def cumulative_steps(self):
        # cumulative_steps counts the overall steps during the training.
        return self._cumulative_steps

    def sync_target_network(self):
        """Synchronize target network with current network."""
        if self.target_model is None:
            self.target_model = copy.deepcopy(self.model)

            def flatten_parameters(mod):
                if isinstance(mod, torch.nn.RNNBase):
                    mod.flatten_parameters()

            # RNNBase.flatten_parameters must be called again after deep-copy.
            # See: https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506  # NOQA
            self.target_model.apply(flatten_parameters)
            # set target n/w to evaluate only.
            self.target_model.eval()
        else:
            synchronize_parameters(
                src=self.model,
                dst=self.target_model,
                method=self.target_update_method,
                tau=self.soft_update_tau,
            )

    def update(self, experiences, errors_out=None):
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None

        Changes from DQN:
            Learned from demonstrations
        """
        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if has_weight:
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []

        if self.reward_based_sampler is not None:
            demo_experiences = self.reward_based_sampler.sample(experiences)
        else:
            demo_experiences = load_experiences_from_demonstrations(
                self.expert_dataset, self.replay_updater.batchsize,
                self.reward_scale)
        demo_batch = batch_experiences(
            demo_experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )

        loss = self._compute_loss(exp_batch, demo_batch, errors_out=errors_out)
        if has_weight:
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optim_t += 1

    def update_from_episodes(self, episodes, errors_out=None):
        assert errors_out is None, "Recurrent DQN does not support PrioritizedBuffer"
        episodes = sorted(episodes, key=len, reverse=True)
        exp_batch = batch_recurrent_experiences(
            episodes,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )

        demo_experiences = load_experiences_from_demonstrations(
            self.expert_dataset, self.replay_updater.batchsize,
            self.reward_scale)
        demo_batch = batch_experiences(
            demo_experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )

        loss = self._compute_loss(exp_batch, demo_batch, errors_out=None)
        self.loss_record.append(float(loss.detach().cpu().numpy()))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optim_t += 1

    def _compute_target_values(self, exp_batch):
        """
        Changes from DQN:
            Consider soft Bellman error
        """
        batch_next_state = exp_batch["next_state"]

        target_next_qout = self.target_model(batch_next_state)

        next_q_max = torch.broadcast_tensors(
            target_next_qout.q_values.max(dim=-1, keepdim=True)[0],
            target_next_qout.q_values)[0]
        next_q_soft = (
            next_q_max[:, 0]
            + (target_next_qout.q_values - next_q_max).exp().sum(dim=-1).log()
        )

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        # return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max
        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_soft

    def _compute_y_and_t(self, exp_batch):
        batch_size = exp_batch["reward"].shape[0]

        # Compute Q-values for current states
        batch_state = exp_batch["state"]

        if self.recurrent:
            qout, _ = pack_and_forward(
                self.model, batch_state, exp_batch["recurrent_state"]
            )
        else:
            qout = self.model(batch_state)

        batch_actions = exp_batch["action"]
        batch_q = torch.reshape(qout.evaluate_actions(batch_actions), (batch_size, 1))

        with torch.no_grad():
            batch_q_target = torch.reshape(
                self._compute_target_values(exp_batch), (batch_size, 1)
            )

        return batch_q, batch_q_target

    def __compute_loss(self, exp_batch, errors_out):
        y, t = self._compute_y_and_t(exp_batch)

        self.q_record.extend(y.detach().cpu().numpy().ravel())

        if errors_out is not None:
            del errors_out[:]
            delta = torch.abs(y - t)
            if delta.ndim == 2:
                delta = torch.sum(delta, dim=1)
            delta = delta.detach().cpu().numpy()
            for e in delta:
                errors_out.append(e)

        if "weights" in exp_batch:
            return compute_weighted_value_loss(
                y,
                t,
                exp_batch["weights"],
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )
        else:
            return compute_value_loss(
                y,
                t,
                clip_delta=self.clip_delta,
                batch_accumulator=self.batch_accumulator,
            )

    def _compute_loss(self, exp_batch, demo_batch, errors_out=None):
        """Compute the Q-learning loss for a batch of experiences


        Args:
          exp_batch (dict): A dict of batched arrays of transitions
        Returns:
          Computed loss from the minibatch of experiences

        Changes from DQN:
            Learned from demonstrations
        """
        exp_loss = self.__compute_loss(exp_batch, errors_out=errors_out)
        demo_loss = self.__compute_loss(demo_batch, errors_out=None)
        return (exp_loss * self.experience_lambda + demo_loss) / 2

    def _evaluate_model_and_update_recurrent_states(self, batch_obs):
        batch_xs = self.batch_states(batch_obs, self.device, self.phi)
        if self.recurrent:
            if self.training:
                self.train_prev_recurrent_states = self.train_recurrent_states
                batch_av, self.train_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.train_recurrent_states
                )
            else:
                batch_av, self.test_recurrent_states = one_step_forward(
                    self.model, batch_xs, self.test_recurrent_states
                )
        else:
            batch_av = self.model(batch_xs)
        return batch_av

    def batch_act(self, batch_obs):
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_argmax = batch_av.greedy_actions.cpu().numpy()
        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t, lambda: batch_argmax[i], action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            # stochastic
            batch_action = [
                self.explorer.select_action(
                    self.t, lambda: batch_argmax[i], action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            # deterministic
            # batch_action = batch_argmax
        return batch_action

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }
                if self.recurrent:
                    transition["recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_prev_recurrent_states, i, detach=True
                        )
                    )
                    transition["next_recurrent_state"] = recurrent_state_as_numpy(
                        get_recurrent_state_at(
                            self.train_recurrent_states, i, detach=True
                        )
                    )
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

        if self.recurrent:
            # Reset recurrent states when episodes end
            self.train_prev_recurrent_states = None
            self.train_recurrent_states = _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                batch_done=batch_done,
                batch_reset=batch_reset,
                recurrent_states=self.train_recurrent_states,
            )

    def _batch_observe_eval(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.recurrent:
            # Reset recurrent states when episodes end
            self.test_recurrent_states = _batch_reset_recurrent_states_when_episodes_end(  # NOQA
                batch_done=batch_done,
                batch_reset=batch_reset,
                recurrent_states=self.test_recurrent_states,
            )

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset
            )
        else:
            return self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset
            )

    def _can_start_replay(self):
        if len(self.replay_buffer) < self.replay_start_size:
            return False
        if self.recurrent and self.replay_buffer.n_episodes < self.minibatch_size:
            return False
        return True

    def stop_episode(self):
        if self.recurrent:
            self.test_recurrent_states = None

    def get_statistics(self):
        return [
            ("average_q", _mean_or_nan(self.q_record)),
            ("average_loss", _mean_or_nan(self.loss_record)),
            ("cumulative_steps", self.cumulative_steps),
            ("n_updates", self.optim_t),
            ("rlen", len(self.replay_buffer)),
        ]
