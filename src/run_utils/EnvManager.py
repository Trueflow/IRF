from mlagents_envs.environment import UnityEnvironment, ActionTuple
import numpy as np
import torch

from Algorithm import REGISTRY as A_Registry
from EnvSetting.TeamInfo import WriteSchemeInfo
from Utils.epsilon_schedule import epsilon_schedule
from modules.Actor import REGISTRY as mac_registry


def _reset_agent_episode_state(agent, args):
    """Reset recurrent state and all episode-specific ID bookkeeping."""
    actor = agent.get("actor")
    if isinstance(actor, list):
        agent["hidden_state"] = [
            item.init_hidden_2() for item in actor
        ]
    elif actor is not None:
        agent["hidden_state"] = actor.init_hidden()

    agent["last_action"] = []
    agent["agent_id_to_slot"] = {}
    agent["slot_to_agent_id"] = [
        None for _ in range(args.num_agents)
    ]
    agent["current_decision_ids"] = []
    agent["episode_terminal_ids"] = set()


def InitialSetting(team, info, env, save_path, load_path):
    if info[0] != "marl":
        return

    decision, _ = env.get_steps(info[-1]["behavior"])
    agent = info[-1]
    agent["agents_id"] = [
        int(value) for value in decision.agent_id
    ]
    agent["algorithm"] = agent["algorithm"].lower()
    algorithm = agent["algorithm"]
    args = agent["args"]
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if algorithm == "poca":
        agent["actor"] = [
            mac_registry["rnn"](
                args, args.obs_size, args.action_size
            )
            for _ in range(args.num_agents)
        ]
    elif args.use_last_action:
        agent["actor"] = mac_registry["rnn"](
            args,
            args.obs_size + args.action_size,
            args.action_size,
        )
    else:
        agent["actor"] = mac_registry["rnn"](
            args, args.obs_size, args.action_size
        )

    _reset_agent_episode_state(agent, args)
    agent["WriteScheme"] = WriteSchemeInfo(
        algorithm, args, team
    )
    agent["epsilon_schedule"] = epsilon_schedule(args.eps_greedy)
    agent["epsilon_schedule"].init_schedule(args.train_mode)
    agent["agent"] = A_Registry[algorithm](
        args,
        agent["actor"],
        args.eps_greedy.start,
        save_path + f"/team{team}MARL-{algorithm}",
        load_path,
    )
    agent["agent"].SetOptimiser()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def episodeEnd(step, startStep, TeamInfo, ENVargs, win_count):
    for team, info in TeamInfo.items():
        args = info[1]
        agent = info[-1]
        scheme = agent["WriteScheme"]
        scheme["EpisodeInfo"]["episode_length"].append(
            step - startStep
        )
        scheme["episode"] += 1
        scheme["EpisodeInfo"]["scores"].append(
            np.sum(scheme["score"], axis=0)
        )
        if args.train_mode:
            agent["agent"].epsilon = (
                agent["epsilon_schedule"].update_epsilon(
                    scheme["episode"]
                )
            )

        if scheme["episode"] % ENVargs.print_interval == 0:
            total_wins = sum(win_count.values())
            win_rate = (
                win_count[team] / total_wins if total_wins > 0 else 0
            )
            if ENVargs.training == args.train_mode:
                agent["agent"].write_summary(
                    scheme, step, ENVargs, win_rate
                )

        if (
            args.train_mode
            and scheme["episode"] % ENVargs.save_interval == 0
        ):
            agent["agent"].save_model()


def memoryClear(TeamInfo):
    for _, info in TeamInfo.items():
        args = info[1]
        agent = info[-1]
        agent["WriteScheme"]["score"].clear()
        agent["agent"].memoryClear()
        _reset_agent_episode_state(agent, args)


def calculate_win(env, TeamInfo, win_count, RSAmode):
    for team, info in TeamInfo.items():
        args = info[1]
        _, terminal = env.get_steps(info[-1]["behavior"])
        if len(terminal.agent_id) == 0:
            continue
        is_win = int(terminal.obs[args.VEC_IDX][0, -2])
        if is_win == 1:
            win_count[team] += 1
        if RSAmode and is_win == -1:
            win_count[1] += 1


def LearningEnd(step, ENVargs, TeamInfo):
    for _, info in TeamInfo.items():
        args = info[1]
        agent = info[-1]
        if args.train_mode:
            agent["agent"].save_model()
            agent["WriteScheme"]["score"].clear()
            agent["WriteScheme"]["episode"] = 0
            agent["agent"].memoryClear()
            _reset_agent_episode_state(agent, args)
            args.train_mode = False

    ENVargs.training = False
    print("Test Start")
