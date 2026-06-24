import breaching
import torch
import math

# Redirects logs directly into the jupyter notebook
import logging, sys

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

cfg = breaching.get_config(overrides=["attack=imprint", "case/server=malicious-model-cah"])

device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

cfg.case.user.num_data_points = 64 # How many data points does this user own
cfg.case.server.model_modification.type = 'CuriousAbandonHonesty' # What type of Imprint block will be grafted to the model
cfg.case.server.model_modification.num_bins = 128 # How many bins are in the block

cfg.case.server.model_modification.position = None # '4.0.conv'
cfg.case.server.model_modification.connection = 'addition'

# Unnormalized data:
# cfg.case.data.normalize = False
# cfg.case.server.model_modification.mu = 0
# cfg.case.server.model_modification.sigma = 0.5
# cfg.case.server.model_modification.scale_factor = 1 - 0.990
# cfg.attack.breach_reduction = None # Will be done manually

# Normalized data:
cfg.case.data.normalize = True
cfg.case.server.model_modification.sigma = 0.5 * 0.2260
cfg.case.server.model_modification.mu = -0.4490 * math.sqrt(224**2*3) * 0.5
cfg.case.server.model_modification.scale_factor = -0.9990
cfg.attack.breach_reduction = None # Will be done manually

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)
server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], server.secrets,
                                                      dryrun=cfg.dryrun)
#%%
user.plot(true_user_data)
reconstructed = torch.zeros_like(true_user_data["data"])
for sample in reconstructed_user_data["data"]:
        l2_dists = (sample[None] - true_user_data["data"]).pow(2).mean(dim=[1, 2, 3])
        min_dist, min_idx = l2_dists.min(dim=0)
        if min_dist < 1e-1:
            reconstructed[min_idx] = sample
reconstructed_user_data = dict(data=reconstructed, labels=None)
#%%
metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                    server.model, order_batch=True, compute_full_iip=False,
                                    cfg_case=cfg.case, setup=setup)

user.plot(reconstructed_user_data)