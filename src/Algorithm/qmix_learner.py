import copy
from datetime import datetime

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from modules.mixer.qmix import QMixer


class QLearner:
    def __init__(self, args, actor, epsilon, save_path, load_path):
        self.args = args
        self.actor = actor
        self.mac = actor
        self.algorithm = "qmix"
        self.epsilon = epsilon
        self.save_path = save_path
        self.load_path = load_path
        self.device = args.device
        self.n_actions = args.action_size
        self.training_episode = 0
        self.memory = []
        self._printed_state_stats = False

        self.mixer = QMixer(args).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(
            self.device
        )
        self.target_mac = self.target_actor
        self.target_mixer = copy.deepcopy(self.mixer).to(
            self.device
        )
        self.params = list(self.actor.parameters()) + list(
            self.mixer.parameters()
        )
        self.writer = SummaryWriter(save_path)

    def SetOptimiser(self):
        optimiser_class = getattr(
            th.optim, self.args.optimiser
        )
        self.optimiser = optimiser_class(
            self.params, **self.args.optimiser_param
        )
        if self.args.load_model:
            self.load_models()

    def append_sample(
        self,
        states,
        obs,
        actions,
        actionmask,
        reward,
        states_next,
        obs_next,
        done,
        actives,
        next_actives,
        next_actionmask,
    ):
        self.memory.append(
            (
                states,
                obs,
                actions,
                actionmask,
                reward,
                states_next,
                obs_next,
                done,
                actives,
                next_actives,
                next_actionmask,
            )
        )

    def memoryClear(self):
        self.memory.clear()

    def GenerateBatch(self):
        if not self.memory:
            raise RuntimeError(
                "QMIX cannot train because episode memory is empty."
            )
        names = (
            "state",
            "obs",
            "actions",
            "avail_actions",
            "reward",
            "state_next",
            "obs_next",
            "terminated",
            "actives",
            "next_actives",
            "next_avail_actions",
        )
        batch = {}
        for index, name in enumerate(names):
            array = np.stack(
                [sample[index] for sample in self.memory], axis=0
            )
            batch[name] = th.as_tensor(
                array, dtype=th.float32, device=self.device
            ).unsqueeze(0)

        batch["actions"] = batch["actions"].long()
        batch["actions_onehot"] = F.one_hot(
            batch["actions"], num_classes=self.n_actions
        ).squeeze(-2).float()
        return batch

    def _build_actor_inputs(self, batch):
        obs_sequence = th.cat(
            [batch["obs"], batch["obs_next"][:, -1:]], dim=1
        )
        if not self.args.use_last_action:
            return obs_sequence

        first_previous_action = th.zeros_like(
            batch["actions_onehot"][:, :1]
        )
        previous_actions = th.cat(
            [first_previous_action, batch["actions_onehot"]],
            dim=1,
        )
        return th.cat(
            [obs_sequence, previous_actions], dim=-1
        )

    @staticmethod
    def _global_state(states):
        return states[:, :, 0, :] if states.dim() == 4 else states

    def train_model(self):
        batch = self.GenerateBatch()
        states = self._global_state(batch["state"])
        next_states = self._global_state(batch["state_next"])

        if not th.isfinite(states).all() or not th.isfinite(
            next_states
        ).all():
            raise FloatingPointError(
                "Non-finite global state entered QMIX batch."
            )
        if not self._printed_state_stats:
            print(
                "[QMIX state] "
                f"min={states.min().item():.4f}, "
                f"max={states.max().item():.4f}, "
                f"max_abs={states.abs().max().item():.4f}, "
                "mixer_layer_norm="
                f"{self.mixer.normalize_state}"
            )
            self._printed_state_stats = True

        actor_inputs = self._build_actor_inputs(batch)
        online_q, _ = self.actor.forward(
            actor_inputs,
            self.actor.init_hidden(),
            softmax=False,
        )
        online_q = online_q.unsqueeze(0)
        with th.no_grad():
            target_q, _ = self.target_actor.forward(
                actor_inputs,
                self.target_actor.init_hidden(),
                softmax=False,
            )
            target_q = target_q.unsqueeze(0)

        chosen_q = th.gather(
            online_q[:, :-1],
            dim=3,
            index=batch["actions"],
        ).squeeze(3)
        chosen_q = chosen_q * batch["actives"].squeeze(-1)

        next_avail = batch["next_avail_actions"]
        next_online_q = online_q[:, 1:].detach().masked_fill(
            next_avail == 0, -1e9
        )
        next_actions = next_online_q.argmax(
            dim=3, keepdim=True
        )
        next_target_q = target_q[:, 1:].masked_fill(
            next_avail == 0, -1e9
        )
        target_taken_q = th.gather(
            next_target_q, dim=3, index=next_actions
        ).squeeze(3)
        target_taken_q = (
            target_taken_q
            * batch["next_actives"].squeeze(-1)
        )

        q_total = self.mixer(chosen_q, states)
        with th.no_grad():
            target_q_total = self.target_mixer(
                target_taken_q, next_states
            )
            targets = (
                batch["reward"]
                + self.args.gamma
                * (1.0 - batch["terminated"])
                * target_q_total
            )

        td_error = q_total - targets
        loss = (td_error ** 2).mean()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip
        )
        self.optimiser.step()

        learn_scheme = {
            "loss": loss.item(),
            "td_error_abs": td_error.detach().abs().mean().item(),
            "q_taken_mean": q_total.detach().mean().item(),
            "target_mean": targets.detach().mean().item(),
            "grad_norm": float(grad_norm),
        }

        self.training_episode += 1
        if (
            self.training_episode
            % self.args.target_update_interval
            == 0
        ):
            self._update_targets()

        self.memoryClear()
        if th.cuda.is_available():
            th.cuda.empty_cache()
        return learn_scheme

    def write_scheme(self, scheme, learn_scheme, args):
        scheme["losses"].append(learn_scheme["loss"])
        scheme["td_error_abs"].append(
            learn_scheme["td_error_abs"]
        )
        scheme["q_taken_mean"].append(
            learn_scheme["q_taken_mean"]
        )
        scheme["target_mean"].append(
            learn_scheme["target_mean"]
        )
        scheme["grad_norm"].append(learn_scheme["grad_norm"])

    def _update_targets(self):
        self.target_actor.load_state_dict(
            self.actor.state_dict()
        )
        self.target_mixer.load_state_dict(
            self.mixer.state_dict()
        )

    def save_model(self):
        print(f"\n... Save Model to {self.save_path}/ckpt ...")
        th.save(
            {
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "mixer": self.mixer.state_dict(),
                "target_mixer": self.target_mixer.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "training_episode": self.training_episode,
            },
            self.save_path + "/ckpt",
        )

    def load_models(self):
        checkpoint = th.load(
            self.load_path + "/ckpt",
            map_location=self.device,
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_actor.load_state_dict(
            checkpoint.get(
                "target_actor", checkpoint["actor"]
            )
        )
        self.target_mixer.load_state_dict(
            checkpoint.get(
                "target_mixer", checkpoint["mixer"]
            )
        )
        self.optimiser.load_state_dict(
            checkpoint["optimiser"]
        )
        self.training_episode = checkpoint.get(
            "training_episode", 0
        )
        print(
            f"... Load Model from "
            f"{self.load_path}/ckpt complete ..."
        )

    def write_summary(self, scheme, step, ENVargs, win_rate):
        info = scheme["EpisodeInfo"]
        episode = scheme["episode"]
        team = scheme["team"]
        episode_lengths = info["episode_length"]
        total_steps = np.sum(episode_lengths)
        total_reward = np.sum(info["scores"])
        current_time = datetime.now().strftime("%m-%d %H:%M:%S")

        if self.args.train_mode:
            mean_loss = (
                np.mean(info["losses"]) if info["losses"] else 0.0
            )
            mean_td = (
                np.mean(info["td_error_abs"])
                if info["td_error_abs"]
                else 0.0
            )
            mean_q = (
                np.mean(info["q_taken_mean"])
                if info["q_taken_mean"]
                else 0.0
            )
            mean_target = (
                np.mean(info["target_mean"])
                if info["target_mean"]
                else 0.0
            )
            mean_grad = (
                np.mean(info["grad_norm"])
                if info["grad_norm"]
                else 0.0
            )
            print(
                f"\n[{current_time}] Episode {episode} team{team} "
                f"(qmix) Summary ({total_steps} step / {step} step)"
            )
            print(
                f"[Reward] r_ex: {total_reward:.4f} | "
                f"win_rate: {100 * win_rate:.3f}%"
            )
            print(
                f"Loss: {mean_loss:.4f} | "
                f"td_error_abs: {mean_td:.4f} | "
                f"Q_taken: {mean_q:.4f} | "
                f"target: {mean_target:.4f} | "
                f"grad_norm: {mean_grad:.4f}\n"
            )
            self.writer.add_scalar(
                "episode/reward", total_reward, episode
            )
            self.writer.add_scalar(
                "episode/episode_length",
                np.mean(episode_lengths)
                if episode_lengths
                else 0.0,
                episode,
            )
            self.writer.add_scalar(
                "episode/win_rate", win_rate * 100, episode
            )
            self.writer.add_scalar(
                "model/loss", mean_loss, episode
            )
            self.writer.add_scalar(
                "model/td_error_abs", mean_td, episode
            )
            self.writer.add_scalar(
                "model/q_taken_mean", mean_q, episode
            )
            self.writer.add_scalar(
                "model/target_mean", mean_target, episode
            )
            self.writer.add_scalar(
                "model/grad_norm", mean_grad, episode
            )
        else:
            print(
                f"\nTestStep {step - ENVargs.run_step} team{team} "
                f"(qmix) Summary ({total_steps} step / {step} step)"
            )
            print(
                f"Episode {episode} | Reward: "
                f"{total_reward:.2f} | "
                f"win_rate: {100 * win_rate:.3f}%"
            )

        for key in info:
            info[key] = []
