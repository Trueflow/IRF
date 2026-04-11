import numpy as np
import torch
import torch.nn.functional as F


def _actor_input(AgentInfo, args):
    obs = torch.as_tensor(
        AgentInfo["memory"]["obs"], dtype=torch.float32, device=args.device
    )
    if not args.use_last_action:
        return obs.unsqueeze(0)

    if AgentInfo["last_action"]:
        last_action = AgentInfo["last_action"]
    else:
        last_action = [0 for _ in range(args.num_agents)]
    last_action_tensor = torch.as_tensor(
        last_action, dtype=torch.long, device=args.device
    ).reshape(args.num_agents, 1)
    last_action_onehot = F.one_hot(
        last_action_tensor, num_classes=args.action_size
    ).squeeze(1).float()
    return torch.cat([obs, last_action_onehot], dim=-1).unsqueeze(0)


def single_actor_selector(AgentInfo, args, training=True):
    actions = []
    actionmask = torch.as_tensor(
        AgentInfo["memory"]["ActionMask"],
        dtype=torch.float32,
        device=args.device,
    )
    actor_input = _actor_input(AgentInfo, args)
    pi, AgentInfo["hidden_state"] = AgentInfo["actor"].inference_forward(
        actor_input, AgentInfo["hidden_state"]
    )
    pi = pi.squeeze(0)

    if training:
        epsilon = AgentInfo["epsilon_schedule"].epsilon
        feasible_count = actionmask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pi = (1.0 - epsilon) * pi + epsilon * actionmask / feasible_count

    masked_pi = pi * actionmask
    for agent in range(args.num_agents):
        if actionmask[agent].sum() == 0:
            actions.append(0)
        else:
            actions.append(
                torch.multinomial(masked_pi[agent], num_samples=1).item()
            )

    AgentInfo["last_action"] = actions
    return np.asarray(actions, dtype=object).reshape(args.num_agents, 1)


def qmix_action_selector(AgentInfo, args, training=True):
    """Masked epsilon-greedy selection from raw Q-values."""
    actionmask = torch.as_tensor(
        AgentInfo["memory"]["ActionMask"],
        dtype=torch.float32,
        device=args.device,
    )
    actor_input = _actor_input(AgentInfo, args)
    q_values, AgentInfo["hidden_state"] = (
        AgentInfo["actor"].inference_forward(
            actor_input, AgentInfo["hidden_state"], softmax=False
        )
    )
    q_values = q_values.squeeze(0).masked_fill(actionmask == 0, -1e9)
    epsilon = AgentInfo["epsilon_schedule"].epsilon if training else 0.0

    actions = []
    for agent in range(args.num_agents):
        feasible = actionmask[agent]
        if feasible.sum() == 0:
            action = 0
        elif training and torch.rand((), device=args.device).item() < epsilon:
            action = torch.multinomial(feasible, num_samples=1).item()
        else:
            action = q_values[agent].argmax().item()
        actions.append(action)

    AgentInfo["last_action"] = actions
    return np.asarray(actions, dtype=object).reshape(args.num_agents, 1)


def multi_actor_selector(AgentInfo, args, training=True):
    actions = []
    hidden_states = AgentInfo["hidden_state"]
    obs = torch.as_tensor(
        AgentInfo["memory"]["obs"], dtype=torch.float32, device=args.device
    )
    actionmask = torch.as_tensor(
        AgentInfo["memory"]["ActionMask"],
        dtype=torch.float32,
        device=args.device,
    )

    for agent_idx in range(args.num_agents):
        agent_obs = obs[agent_idx : agent_idx + 1].unsqueeze(0)
        agent_mask = actionmask[agent_idx]
        pi, new_hidden = AgentInfo["actor"][agent_idx].inference_forward(
            agent_obs, hidden_states[agent_idx]
        )
        AgentInfo["hidden_state"][agent_idx] = new_hidden
        pi = pi.squeeze()

        if training:
            epsilon = AgentInfo["epsilon_schedule"].epsilon
            feasible_count = agent_mask.sum().clamp_min(1.0)
            pi = (1.0 - epsilon) * pi + epsilon * agent_mask / feasible_count

        if agent_mask.sum() == 0:
            actions.append(0)
        else:
            actions.append(
                torch.multinomial(pi * agent_mask, num_samples=1).item()
            )

    return np.asarray(actions, dtype=object).reshape(args.num_agents, 1)
