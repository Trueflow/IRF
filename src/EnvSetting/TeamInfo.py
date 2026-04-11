def teamInfo(env, config):
    behavior_keys = list(env.behavior_specs.keys())
    print(behavior_keys)

    if config[1].Framework != "marl":
        team_behavior, team_info = {0: []}, {0: []}
        team_behavior[0].append(behavior_keys[0])
        team_num = 1
    else:
        team_behavior, team_info = {0: [], 1: []}, {0: [], 1: []}
        for name in behavior_keys:
            if "team=0" in name:
                team_behavior[0].append(name)
            elif "team=1" in name:
                team_behavior[1].append(name)
        team_num = 2

    for team in range(team_num):
        frame = config[team].Framework
        algorithm = config[team].Algorithm
        args = config[team]
        team_info[team].append(frame)
        team_info[team].append(args)
        if frame == "marl":
            agent_info = create_base_agent_info(
                team_behavior[team][0], algorithm, args
            )
            if algorithm in ("poca_single", "poca"):
                agent_info["memory"]["next_state"] = []
            team_info[team].append(agent_info)
    return team_info


def create_base_agent_info(behavior, algorithm, args):
    return {
        "behavior": behavior,
        "algorithm": algorithm,
        "args": args,
        "agents_id": [],
        "memory": {
            "state": [],
            "obs": [],
            "actions": [],
            "activeSelf": [],
            "reward": [],
            "groupReward": [],
            "done": False,
        },
        "last_action": [],
    }


def create_irf_scheme(args, team):
    return {
        "EpisodeInfo": {
            "scores": [],
            "r_in_list": [],
            "actor_losses": [],
            "intrinsic_losses": [],
            "critic_losses": [],
            "episode_length": [],
        }
    }


def create_coma_scheme(args, team):
    return {
        "EpisodeInfo": {
            "scores": [],
            "actor_losses": [],
            "critic_losses": [],
            "episode_length": [],
        }
    }


def create_poca_scheme(args, team):
    return create_coma_scheme(args, team)


def create_cds_scheme(args, team):
    return {
        "EpisodeInfo": {
            "scores": [],
            "losses": [],
            "td_losses": [],
            "attention_regs": [],
            "td_error_abs": [],
            "hit_prob": [],
            "grad_norm": [],
            "intrinsic_rewards": [
                [] for _ in range(args.num_agents)
            ],
            "episode_length": [],
        }
    }


def create_emc_scheme(args, team):
    return {
        "EpisodeInfo": {
            "scores": [],
            "losses": [],
            "td_losses": [],
            "attention_regs": [],
            "vdn_losses": [],
            "prediction_losses": [],
            "td_error_abs": [],
            "hit_prob": [],
            "intrinsic_rewards": [],
            "grad_norm": [],
            "episode_length": [],
        }
    }


def create_qmix_scheme(args, team):
    return {
        "EpisodeInfo": {
            "scores": [],
            "losses": [],
            "td_error_abs": [],
            "q_taken_mean": [],
            "target_mean": [],
            "grad_norm": [],
            "episode_length": [],
        }
    }


SCHEMA_REGISTRY = {
    "irf": create_irf_scheme,
    "coma": create_coma_scheme,
    "poca": create_poca_scheme,
    "cds": create_cds_scheme,
    "emc": create_emc_scheme,
    "qmix": create_qmix_scheme,
}


def WriteSchemeInfo(algorithm, args, team):
    write_scheme = {
        "score": [],
        "episode": 0,
        "done": False,
        "team": team,
    }
    if algorithm not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown algorithm: {algorithm}")
    write_scheme.update(SCHEMA_REGISTRY[algorithm](args, team))
    return write_scheme
