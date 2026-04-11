from .irf_learner import IRFagent
from .mapoca_learner import POCAagent
from .coma_learner import COMAagent
from .cds_learner import CDSagent
from .emc_learner import EMCagent
from .qmix_learner import QLearner

REGISTRY = {}

REGISTRY["irf"] = IRFagent
REGISTRY["coma"] = COMAagent
REGISTRY["poca"] = POCAagent
REGISTRY["cds"] = CDSagent
REGISTRY["emc"] = EMCagent
REGISTRY["qmix"] = QLearner