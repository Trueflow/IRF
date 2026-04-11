import numpy as np
from mlagents_envs.base_env import ActionTuple


def process_state(obs, args):
    mask_index = args.vec_obs + args.action_size
    return np.asarray(
        obs[args.VEC_IDX][:, mask_index:-2], dtype=np.float32
    )


def process_obs(obs, args):
    ray_obs = np.asarray(obs[args.RAY_IDX], dtype=np.float32)
    vec_obs = np.asarray(
        obs[args.VEC_IDX][:, : args.vec_obs], dtype=np.float32
    )
    return np.concatenate((ray_obs, vec_obs), axis=-1)


def process_actionmask(obs, args):
    mask_index = args.vec_obs + args.action_size
    return np.asarray(
        obs[args.VEC_IDX][:, args.vec_obs:mask_index],
        dtype=np.float32,
    )


def process_active(obs, args):
    return np.asarray(
        obs[args.VEC_IDX][:, -1], dtype=np.float32
    )


def _ensure_agent_slots(agent, agent_ids, num_agents):
    """Assign ML-Agents IDs to stable recurrent-network slots."""
    mapping = agent.setdefault("agent_id_to_slot", {})
    reverse = agent.setdefault(
        "slot_to_agent_id", [None for _ in range(num_agents)]
    )

    for agent_id_value in agent_ids:
        agent_id = int(agent_id_value)
        if agent_id in mapping:
            continue
        free_slots = [
            index for index, value in enumerate(reverse) if value is None
        ]
        if not free_slots:
            raise RuntimeError(
                f"Received more than {num_agents} agent IDs. "
                f"Known={mapping}, new_id={agent_id}"
            )
        slot = free_slots[0]
        mapping[agent_id] = slot
        reverse[slot] = agent_id
    return mapping


def align_decision_step(decision_steps, args, agent):
    """Convert DecisionSteps rows into fixed [num_agents, ...] arrays."""
    mapping = _ensure_agent_slots(
        agent, decision_steps.agent_id, args.num_agents
    )
    row_obs = process_obs(decision_steps.obs, args)
    row_state = process_state(decision_steps.obs, args)
    row_mask = process_actionmask(decision_steps.obs, args)
    row_active = process_active(decision_steps.obs, args)

    aligned_obs = np.zeros(
        (args.num_agents, args.obs_size), dtype=np.float32
    )
    # A missing/inactive slot receives a safe dummy action.
    aligned_mask = np.zeros(
        (args.num_agents, args.action_size), dtype=np.float32
    )
    aligned_mask[:, 0] = 1.0
    aligned_active = np.zeros((args.num_agents,), dtype=np.float32)

    for row, agent_id_value in enumerate(decision_steps.agent_id):
        slot = mapping[int(agent_id_value)]
        aligned_obs[slot] = row_obs[row]
        aligned_mask[slot] = row_mask[row]
        if aligned_mask[slot].sum() <= 0:
            aligned_mask[slot, 0] = 1.0
        aligned_active[slot] = row_active[row]

    # The Unity environment stores the same global state in every row.
    # Replicating one representative keeps state available even if slot 0 dies.
    aligned_state = np.zeros(
        (args.num_agents, args.state_size), dtype=np.float32
    )
    if len(row_state):
        representative = row_state[0]
        aligned_state[:] = representative
        max_row_difference = float(
            np.max(np.abs(row_state - representative))
        )
        agent["last_global_state_row_difference"] = max_row_difference

    agent["current_decision_ids"] = [
        int(value) for value in decision_steps.agent_id
    ]
    return aligned_state, aligned_obs, aligned_mask, aligned_active


def actions_for_decision_rows(actions, decision_steps, agent):
    """Map fixed-slot actions back to the row order expected by Unity."""
    mapping = agent["agent_id_to_slot"]
    fixed_actions = np.asarray(actions, dtype=np.int32).reshape(-1, 1)
    decision_actions = np.zeros(
        (len(decision_steps.agent_id), 1), dtype=np.int32
    )
    for row, agent_id_value in enumerate(decision_steps.agent_id):
        decision_actions[row] = fixed_actions[
            mapping[int(agent_id_value)]
        ]
    return decision_actions


def set_action(actions, env, behavior):
    action_tuple = ActionTuple()
    action_tuple.add_discrete(
        np.asarray(actions, dtype=np.int32)
    )
    env.set_actions(behavior, action_tuple)


def process_rewards(decision_steps, terminal_steps):
    rewards = []
    if len(decision_steps.agent_id):
        rewards.extend(
            np.asarray(
                decision_steps.group_reward, dtype=np.float32
            ).tolist()
        )
    if len(terminal_steps.agent_id):
        rewards.extend(
            np.asarray(
                terminal_steps.group_reward, dtype=np.float32
            ).tolist()
        )
    return float(np.mean(rewards)) if rewards else 0.0


def merge_next_steps(decision_steps, terminal_steps, args, agent):
    """Merge continuing and terminal rows by agent_id.

    Returns fixed-slot next observations plus two distinct episode signals:
    episode_done controls environment reset/bookkeeping, while terminated is
    the TD terminal flag. A time-limit interruption ends the episode but keeps
    terminated=False so value learning may bootstrap.
    """
    all_ids = list(decision_steps.agent_id) + list(
        terminal_steps.agent_id
    )
    mapping = _ensure_agent_slots(agent, all_ids, args.num_agents)

    next_obs = np.zeros(
        (args.num_agents, args.obs_size), dtype=np.float32
    )
    next_active = np.zeros(
        (args.num_agents, 1), dtype=np.float32
    )
    next_mask = np.zeros(
        (args.num_agents, args.action_size), dtype=np.float32
    )
    next_mask[:, 0] = 1.0

    state_candidates = []

    if len(decision_steps.agent_id):
        dec_obs = process_obs(decision_steps.obs, args)
        dec_state = process_state(decision_steps.obs, args)
        dec_mask = process_actionmask(decision_steps.obs, args)
        dec_active = process_active(decision_steps.obs, args)
        state_candidates.extend(dec_state)
        for row, agent_id_value in enumerate(decision_steps.agent_id):
            slot = mapping[int(agent_id_value)]
            next_obs[slot] = dec_obs[row]
            next_mask[slot] = dec_mask[row]
            if next_mask[slot].sum() <= 0:
                next_mask[slot, 0] = 1.0
            next_active[slot, 0] = dec_active[row]

    terminal_interrupted = {}
    if len(terminal_steps.agent_id):
        term_obs = process_obs(terminal_steps.obs, args)
        term_state = process_state(terminal_steps.obs, args)
        term_mask = process_actionmask(terminal_steps.obs, args)
        term_active = process_active(terminal_steps.obs, args)
        state_candidates.extend(term_state)

        for row, agent_id_value in enumerate(terminal_steps.agent_id):
            agent_id = int(agent_id_value)
            slot = mapping[agent_id]
            interrupted = bool(terminal_steps.interrupted[row])
            terminal_interrupted[agent_id] = interrupted
            next_obs[slot] = term_obs[row]
            next_mask[slot] = term_mask[row]
            if next_mask[slot].sum() <= 0:
                # TerminalSteps has no official action mask. Preserve the
                # previous feasible set for time-limit bootstrapping.
                previous_mask = np.asarray(
                    agent["memory"].get("ActionMask", next_mask),
                    dtype=np.float32,
                )
                if interrupted and previous_mask[slot].sum() > 0:
                    next_mask[slot] = previous_mask[slot]
                else:
                    next_mask[slot, 0] = 1.0
            # A naturally terminated unit has no next utility. An interrupted
            # unit remains bootstrap-active at its final observation.
            next_active[slot, 0] = (
                term_active[row] if interrupted else 0.0
            )

    next_state = np.zeros(
        (args.num_agents, args.state_size), dtype=np.float32
    )
    if state_candidates:
        next_state[:] = np.asarray(state_candidates[0], dtype=np.float32)

    seen_terminal_ids = agent.setdefault(
        "episode_terminal_ids", set()
    )
    seen_terminal_ids.update(
        int(value) for value in terminal_steps.agent_id
    )
    known_ids = set(mapping.keys())
    all_known_terminated = bool(known_ids) and known_ids.issubset(
        seen_terminal_ids
    )
    no_continuing_agents = (
        len(terminal_steps.agent_id) > 0
        and len(decision_steps.agent_id) == 0
    )
    episode_done = all_known_terminated or no_continuing_agents

    if episode_done and terminal_interrupted:
        episode_interrupted = all(terminal_interrupted.values())
    else:
        episode_interrupted = False
    terminated = episode_done and not episode_interrupted

    return {
        "state": next_state,
        "obs": next_obs,
        "active": next_active,
        "action_mask": next_mask,
        "episode_done": episode_done,
        "terminated": terminated,
        "interrupted": episode_interrupted,
        "reward": process_rewards(decision_steps, terminal_steps),
    }


# Compatibility wrappers used by older learners.
def process_done(term):
    return bool(len(term.agent_id))


def process_next_state(dec, term, args):
    source = term.obs if len(term.agent_id) else dec.obs
    return (
        process_state(source, args),
        process_obs(source, args),
        process_active(source, args),
    )
