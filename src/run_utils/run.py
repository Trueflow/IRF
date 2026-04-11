import numpy as np
import torch

from modules.action_selector import (
    action_selector_registry as action_selector,
)
from run_utils.mlagents_trans import (
    actions_for_decision_rows,
    align_decision_step,
    merge_next_steps,
    set_action,
)


def agents_run(env, Info):
    agent = Info[-1]
    args = Info[1]
    decision, _ = env.get_steps(agent["behavior"])

    if len(decision.agent_id) == 0:
        agent["memory"]["has_decision"] = False
        return

    state, obs, action_mask, active = align_decision_step(
        decision, args, agent
    )
    agent["memory"]["state"] = state
    agent["memory"]["obs"] = obs
    agent["memory"]["ActionMask"] = action_mask
    agent["memory"]["activeSelf"] = active
    agent["memory"]["has_decision"] = True

    fixed_actions = action_selector[agent["algorithm"]](
        agent, args, args.train_mode
    )
    agent["memory"]["actions"] = fixed_actions
    decision_actions = actions_for_decision_rows(
        fixed_actions, decision, agent
    )
    set_action(decision_actions, env, agent["behavior"])

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def agents_step(env, Info):
    env_type, args, agent = Info[0], Info[1], Info[-1]
    decision, terminal = env.get_steps(agent["behavior"])
    transition = merge_next_steps(
        decision, terminal, args, agent
    )

    agent["memory"]["done"] = transition["episode_done"]
    agent["memory"]["terminated"] = transition["terminated"]
    agent["memory"]["interrupted"] = transition["interrupted"]
    agent["memory"]["reward"] = transition["reward"]
    agent["WriteScheme"]["score"].append(transition["reward"])

    has_decision = agent["memory"].get("has_decision", False)
    if args.train_mode and has_decision:
        obs, actions, action_mask, active = sample_memory(
            agent, args, env_type
        )

        if args.use_next_state:
            if agent["algorithm"] == "qmix":
                agent["agent"].append_sample(
                    agent["memory"]["state"],
                    obs,
                    actions,
                    action_mask,
                    [agent["memory"]["reward"]],
                    transition["state"],
                    transition["obs"],
                    [transition["terminated"]],
                    active,
                    transition["active"],
                    transition["action_mask"],
                )
            else:
                # Existing non-QMIX learners retain their current interface.
                agent["agent"].append_sample(
                    agent["memory"]["state"],
                    obs,
                    actions,
                    action_mask,
                    [agent["memory"]["reward"]],
                    transition["state"],
                    transition["obs"],
                    [transition["terminated"]],
                    active,
                )
        else:
            agent["agent"].append_sample(
                agent["memory"]["state"],
                obs,
                actions,
                action_mask,
                [agent["memory"]["reward"]],
                [transition["terminated"]],
                active,
            )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def agents_write(Info):
    agent, args = Info[-1], Info[1]
    learn_scheme = agent["agent"].train_model()
    scheme = agent["WriteScheme"]["EpisodeInfo"]
    agent["agent"].write_scheme(scheme, learn_scheme, args)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_memory(agent, args, env_type):
    obs = np.zeros(
        (args.num_agents, args.obs_size), dtype=np.float32
    )
    action_mask = np.zeros(
        (args.num_agents, args.action_size), dtype=np.float32
    )
    actions = np.zeros(
        (args.num_agents, 1), dtype=np.int64
    )
    active_agents = np.zeros(
        (args.num_agents, 1), dtype=np.float32
    )

    for slot in range(args.num_agents):
        if float(agent["memory"]["activeSelf"][slot]) == 1.0:
            obs[slot] = agent["memory"]["obs"][slot]
            action_mask[slot] = agent["memory"]["ActionMask"][slot]
            actions[slot] = agent["memory"]["actions"][slot]
            active_agents[slot, 0] = 1.0
    return obs, actions, action_mask, active_agents
