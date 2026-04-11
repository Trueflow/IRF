import copy
from datetime import datetime

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from modules.intrinsic.CDS_intrinsic import (
    Combined_Predict_Net,
    Predict_Net,
)
from modules.mixer.mixer import dmaq_Mixer


class CDSagent:
    def __init__(self, args, actor, epsilon, save_path, load_path):
        self.args = args
        self.actor = actor
        self.algorithm = "cds"
        self.writer = SummaryWriter(save_path)
        self.save_path = save_path
        self.load_path = load_path
        self.device = args.device
        self.n_actions = args.action_size
        self.training_episode = 0
        self.memory = []

        self.mixer = dmaq_Mixer(args).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_mac = copy.deepcopy(self.actor).to(self.device)
        self.params = list(self.actor.parameters()) + list(
            self.mixer.parameters()
        )

        self.eval_predict_withoutid = Predict_Net(args).to(self.device)
        self.target_predict_withoutid = copy.deepcopy(
            self.eval_predict_withoutid
        ).to(self.device)
        self.eval_predict_withid = Combined_Predict_Net(args).to(
            self.device
        )
        self.target_predict_withid = copy.deepcopy(
            self.eval_predict_withid
        ).to(self.device)

    def SetOptimiser(self):
        optimiser_class = getattr(th.optim, self.args.optimiser)
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
            )
        )

    def memoryClear(self):
        self.memory.clear()

    def _batch(self):
        if not self.memory:
            raise RuntimeError("CDS cannot train because memory is empty.")
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

    @staticmethod
    def _next_availability(avail_actions):
        final_dummy = th.zeros_like(avail_actions[:, :1])
        final_dummy[..., 0] = 1.0
        return th.cat([avail_actions[:, 1:], final_dummy], dim=1)

    @staticmethod
    def _next_actives(actives):
        return th.cat(
            [actives[:, 1:], th.zeros_like(actives[:, :1])], dim=1
        )

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
            [first_previous_action, batch["actions_onehot"]], dim=1
        )
        return th.cat([obs_sequence, previous_actions], dim=-1)

    def _intrinsic_rewards(
        self,
        batch,
        online_q,
        recurrent_outputs,
        initial_hidden,
    ):
        batch_size, sequence_length, n_agents, _ = batch["obs"].shape

        # h before processing observation t: h0, h_after_0, ..., h_after_t-1.
        previous_outputs = recurrent_outputs[:-2].unsqueeze(0)
        hidden_before = th.cat(
            [initial_hidden.unsqueeze(0), previous_outputs], dim=1
        )

        hidden_agent_major = hidden_before.permute(0, 2, 1, 3)
        obs_agent_major = batch["obs"].permute(0, 2, 1, 3)
        action_agent_major = batch["actions_onehot"].permute(0, 2, 1, 3)
        next_obs_agent_major = batch["obs_next"].permute(0, 2, 1, 3)
        active_agent_major = batch["actives"].permute(0, 2, 1, 3)

        agent_ids = th.eye(n_agents, device=self.device).view(
            1, n_agents, 1, n_agents
        )
        agent_ids = agent_ids.expand(
            batch_size, n_agents, sequence_length, n_agents
        )

        predictor_without_id = th.cat(
            [
                hidden_agent_major,
                obs_agent_major,
                action_agent_major,
            ],
            dim=-1,
        ).detach()
        predictor_with_id = th.cat(
            [predictor_without_id, agent_ids], dim=-1
        )

        flat_without_id = predictor_without_id.reshape(
            -1, predictor_without_id.size(-1)
        )
        flat_with_id = predictor_with_id.reshape(
            -1, predictor_with_id.size(-1)
        )
        flat_next_obs = next_obs_agent_major.reshape(
            -1, next_obs_agent_major.size(-1)
        )
        flat_ids = agent_ids.reshape(-1, n_agents)

        with th.no_grad():
            log_p = self.target_predict_withoutid.get_log_pi(
                flat_without_id, flat_next_obs
            )
            log_q = self.target_predict_withid.get_log_pi(
                flat_with_id, flat_next_obs, flat_ids
            )

            current_q = online_q[:, :-1]
            mean_policy = F.softmax(current_q, dim=-1).mean(
                dim=2, keepdim=True
            )
            individual_policy = F.softmax(
                self.args.beta1 * current_q, dim=-1
            )
            policy_divergence = (
                individual_policy
                * (
                    th.log(individual_policy + 1e-8)
                    - th.log(mean_policy + 1e-8)
                )
            ).sum(dim=-1)
            policy_divergence = policy_divergence.permute(
                0, 2, 1
            ).unsqueeze(-1)

            intrinsic = (
                self.args.beta1 * log_q - log_p
            ).reshape(batch_size, n_agents, sequence_length, 1)
            intrinsic = (
                intrinsic
                + self.args.beta2 * policy_divergence
            )
            intrinsic = intrinsic * active_agent_major

        active_flat = active_agent_major.reshape(-1) > 0
        valid_indices = active_flat.nonzero(
            as_tuple=False
        ).squeeze(1).tolist()
        for indices in BatchSampler(
            SubsetRandomSampler(valid_indices), 256, False
        ):
            self.eval_predict_withoutid.update(
                flat_without_id[indices], flat_next_obs[indices]
            )
            self.eval_predict_withid.update(
                flat_with_id[indices],
                flat_next_obs[indices],
                flat_ids[indices],
            )

        active_count = active_agent_major.sum(dim=1).clamp_min(1.0)
        team_intrinsic = intrinsic.sum(dim=1) / active_count
        return intrinsic, team_intrinsic

    def train_model(self):
        batch = self._batch()
        actor_inputs = self._build_actor_inputs(batch)

        initial_hidden = self.actor.init_hidden()
        online_q, recurrent_outputs = self.actor.cds_forward(
            actor_inputs,
            initial_hidden,
            softmax=False,
        )
        online_q = online_q.unsqueeze(0)

        with th.no_grad():
            target_q, _ = self.target_mac.cds_forward(
                actor_inputs,
                self.target_mac.init_hidden(),
                softmax=False,
            )
            target_q = target_q.unsqueeze(0)

        current_active = batch["actives"].squeeze(-1)
        next_active = self._next_actives(batch["actives"]).squeeze(-1)
        chosen_q = th.gather(
            online_q[:, :-1], dim=3, index=batch["actions"]
        ).squeeze(3)
        chosen_q = chosen_q * current_active

        current_masked_q = online_q[:, :-1].detach().masked_fill(
            batch["avail_actions"] == 0, -1e9
        )
        current_max_q = current_masked_q.max(dim=3).values
        current_max_q = current_max_q * current_active

        next_avail = self._next_availability(batch["avail_actions"])
        next_online_q = online_q[:, 1:].detach().masked_fill(
            next_avail == 0, -1e9
        )
        next_actions = next_online_q.argmax(dim=3, keepdim=True)
        next_target_q = target_q[:, 1:].masked_fill(
            next_avail == 0, -1e9
        )
        target_chosen_q = th.gather(
            next_target_q, dim=3, index=next_actions
        ).squeeze(3)
        target_chosen_q = target_chosen_q * next_active
        target_max_q = next_target_q.max(dim=3).values * next_active

        next_actions_onehot = F.one_hot(
            next_actions.squeeze(-1), num_classes=self.n_actions
        ).float()

        individual_intrinsic, team_intrinsic = self._intrinsic_rewards(
            batch,
            online_q,
            recurrent_outputs,
            initial_hidden,
        )

        value_total, attention_reg, _ = self.mixer(
            chosen_q, batch["state"], is_v=True
        )
        advantage_total, _, _ = self.mixer(
            chosen_q,
            batch["state"],
            actions=batch["actions_onehot"],
            max_q_i=current_max_q,
            is_v=False,
        )
        q_total = value_total + advantage_total

        with th.no_grad():
            target_value, _, _ = self.target_mixer(
                target_chosen_q, batch["state_next"], is_v=True
            )
            target_advantage, _, _ = self.target_mixer(
                target_chosen_q,
                batch["state_next"],
                actions=next_actions_onehot,
                max_q_i=target_max_q,
                is_v=False,
            )
            target_total = target_value + target_advantage
            targets = (
                batch["reward"]
                + self.args.beta * team_intrinsic
                + self.args.gamma
                * (1.0 - batch["terminated"])
                * target_total
            )

        td_error = q_total - targets
        td_loss = (td_error ** 2).mean()
        loss = td_loss + attention_reg

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip
        )
        self.optimiser.step()

        greedy_actions = current_masked_q.argmax(
            dim=3, keepdim=True
        )
        active_denominator = current_active.sum().clamp_min(1.0)
        hit_prob = (
            (
                (greedy_actions == batch["actions"]).float().squeeze(-1)
                * current_active
            ).sum()
            / active_denominator
        )

        agent_intrinsic_mean = (
            individual_intrinsic.sum(dim=2).squeeze(0).squeeze(-1)
            / batch["actives"]
            .permute(0, 2, 1, 3)
            .sum(dim=2)
            .squeeze(0)
            .squeeze(-1)
            .clamp_min(1.0)
        )
        learn_scheme = {
            "agent_r_in": agent_intrinsic_mean.detach().cpu().tolist(),
            "loss": loss.item(),
            "td_loss": td_loss.item(),
            "attention_reg": attention_reg.detach().item(),
            "td_error_abs": td_error.detach().abs().mean().item(),
            "hit_prob": hit_prob.item(),
            "grad_norm": float(grad_norm),
        }

        self.training_episode += 1
        if self.training_episode % self.args.target_update_interval == 0:
            self._update_targets()

        self.memoryClear()
        if th.cuda.is_available():
            th.cuda.empty_cache()
        return learn_scheme

    def write_scheme(self, scheme, learn_scheme, args):
        scheme["losses"].append(learn_scheme["loss"])
        scheme["td_losses"].append(learn_scheme["td_loss"])
        scheme["attention_regs"].append(learn_scheme["attention_reg"])
        scheme["td_error_abs"].append(learn_scheme["td_error_abs"])
        scheme["hit_prob"].append(learn_scheme["hit_prob"])
        scheme["grad_norm"].append(learn_scheme["grad_norm"])
        for agent in range(args.num_agents):
            scheme["intrinsic_rewards"][agent].append(
                learn_scheme["agent_r_in"][agent]
            )

    def _update_targets(self):
        self.target_mac.load_state_dict(self.actor.state_dict())
        self.target_predict_withid.load_state_dict(
            self.eval_predict_withid.state_dict()
        )
        self.target_predict_withoutid.load_state_dict(
            self.eval_predict_withoutid.state_dict()
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        print("CDS updated target networks")

    def save_model(self):
        print(f"\n... Save Model to {self.save_path}/ckpt ...")
        checkpoint = {
            "actor": self.actor.state_dict(),
            "target_actor": self.target_mac.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "eval_predict_withid": self.eval_predict_withid.state_dict(),
            "target_predict_withid": self.target_predict_withid.state_dict(),
            "eval_predict_withoutid": (
                self.eval_predict_withoutid.state_dict()
            ),
            "target_predict_withoutid": (
                self.target_predict_withoutid.state_dict()
            ),
            "predict_withid_optimiser": (
                self.eval_predict_withid.optimiser.state_dict()
            ),
            "predict_withoutid_optimiser": (
                self.eval_predict_withoutid.optimiser.state_dict()
            ),
            "training_episode": self.training_episode,
        }
        th.save(checkpoint, self.save_path + "/ckpt")

    def load_models(self):
        checkpoint = th.load(
            self.load_path + "/ckpt", map_location=self.device
        )
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_mac.load_state_dict(
            checkpoint.get("target_actor", checkpoint["actor"])
        )
        self.mixer.load_state_dict(checkpoint["mixer"])
        self.target_mixer.load_state_dict(
            checkpoint.get("target_mixer", checkpoint["mixer"])
        )
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        self.eval_predict_withid.load_state_dict(
            checkpoint["eval_predict_withid"]
        )
        self.target_predict_withid.load_state_dict(
            checkpoint.get(
                "target_predict_withid",
                checkpoint["eval_predict_withid"],
            )
        )
        self.eval_predict_withoutid.load_state_dict(
            checkpoint["eval_predict_withoutid"]
        )
        self.target_predict_withoutid.load_state_dict(
            checkpoint.get(
                "target_predict_withoutid",
                checkpoint["eval_predict_withoutid"],
            )
        )
        if "predict_withid_optimiser" in checkpoint:
            self.eval_predict_withid.optimiser.load_state_dict(
                checkpoint["predict_withid_optimiser"]
            )
        if "predict_withoutid_optimiser" in checkpoint:
            self.eval_predict_withoutid.optimiser.load_state_dict(
                checkpoint["predict_withoutid_optimiser"]
            )
        self.training_episode = checkpoint.get("training_episode", 0)
        print(f"... Load Model from {self.load_path}/ckpt complete ...")

    def write_summary(self, scheme, step, ENVargs, win_rate):
        info = scheme["EpisodeInfo"]
        episode = scheme["episode"]
        team = scheme["team"]
        total_steps = np.sum(info["episode_length"])
        total_reward = np.sum(info["scores"])
        current_time = datetime.now().strftime("%m-%d %H:%M:%S")

        if self.args.train_mode:
            intrinsic = [
                np.mean(values) if values else 0.0
                for values in info["intrinsic_rewards"]
            ]
            mean_loss = np.mean(info["losses"]) if info["losses"] else 0.0
            mean_td_loss = (
                np.mean(info["td_losses"]) if info["td_losses"] else 0.0
            )
            mean_attention = (
                np.mean(info["attention_regs"])
                if info["attention_regs"]
                else 0.0
            )
            mean_td_error = (
                np.mean(info["td_error_abs"])
                if info["td_error_abs"]
                else 0.0
            )
            mean_hit = (
                np.mean(info["hit_prob"]) if info["hit_prob"] else 0.0
            )
            mean_grad = (
                np.mean(info["grad_norm"]) if info["grad_norm"] else 0.0
            )
            print(
                f"\n[{current_time}] Episode {episode} team{team} "
                f"(cds) Summary ({total_steps} step / {step} step)"
            )
            print(f"[Intrinsic Reward] r_in: {np.round(intrinsic, 5)}")
            print(
                f"[Reward] r_ex: {total_reward:.4f} | "
                f"win_rate: {100 * win_rate:.3f}%"
            )
            print(
                f"Loss: {mean_loss:.4f} | TD: {mean_td_loss:.4f} | "
                f"attention: {mean_attention:.6f} | "
                f"td_error_abs: {mean_td_error:.4f} | "
                f"hit_prob: {mean_hit:.4f} | grad_norm: {mean_grad:.4f}\n"
            )

            self.writer.add_scalar("episode/reward", total_reward, episode)
            self.writer.add_scalar(
                "episode/episode_length",
                np.mean(info["episode_length"]),
                episode,
            )
            self.writer.add_scalar(
                "episode/win_rate", win_rate * 100, episode
            )
            self.writer.add_scalar(
                "model/intrinsic_reward", np.mean(intrinsic), episode
            )
            self.writer.add_scalar("model/loss", mean_loss, episode)
            self.writer.add_scalar("model/td_loss", mean_td_loss, episode)
            self.writer.add_scalar(
                "model/attention_reg", mean_attention, episode
            )
            self.writer.add_scalar(
                "model/td_error_abs", mean_td_error, episode
            )
            self.writer.add_scalar("model/hit_prob", mean_hit, episode)
            self.writer.add_scalar("model/grad_norm", mean_grad, episode)
        else:
            print(
                f"\nTestStep {step - ENVargs.run_step} team{team} "
                f"(cds) Summary ({total_steps} step / {step} step)"
            )
            print(
                f"Episode {episode} | Reward: {total_reward:.2f} | "
                f"win_rate: {100 * win_rate:.3f}%"
            )
            self.writer.add_scalar(
                "Test/episode_length",
                np.mean(info["episode_length"]),
                episode,
            )
            self.writer.add_scalar("Test/win_rate", win_rate * 100, episode)
            self.writer.add_scalar("Test/reward", total_reward, episode)

        for key in info:
            if key == "intrinsic_rewards":
                info[key] = [[] for _ in range(self.args.num_agents)]
            else:
                info[key] = []
