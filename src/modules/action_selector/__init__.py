from .action_selector import single_actor_selector, qmix_action_selector ,multi_actor_selector

action_selector_registry = {}

action_selector_registry["irf"] = single_actor_selector
action_selector_registry["coma"] = single_actor_selector
action_selector_registry["cds"] = single_actor_selector
action_selector_registry["qmix"] = qmix_action_selector