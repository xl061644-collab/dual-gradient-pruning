import breaching
import os
os.chdir("/HARD-DRIVE/XLL/DGP/breaching_ATS/results/")
import torch
import logging, sys
import numpy as np
from test_m import _run_vision_metrics1
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
cfg = breaching.get_config(overrides=["attack=imprint", "case/server=malicious-model-rtf","case=6_large_batch_cifar","case/data=CIFAR10"])
device = torch.device(f'cuda:0')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
import csv
import pandas as pd

from defense import soteria_imagenet,dgp
setup


# cfg.case.model = "lenet_zhu"
# device = torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup
cfg.case.user.num_data_points = 9# How many data points does this user own
cfg.case.server.model_modification.type = 'ImprintBlock' # What type of Imprint block will be grafted to the model
cfg.case.server.model_modification.num_bins = 128 # How many bins are in the block
print(cfg.case.model)

# LeNetZhu
# How does the block interact with the model?
# The block can be placed later in the model given a position such as  '4.0.conv':
cfg.case.server.model_modification.position = None  # None defaults to the first layer
# The block can also be connected in various ways to the other layers:
cfg.case.server.model_modification.connection = 'addition'



# Which linear measurement function should be used?
# We know that the input dataset is already normalized as preprocessing step

# Knowing the distribution relatively well:
cfg.case.server.model_modification.linfunc = 'fourier' # works well for any normalized image data
cfg.case.server.model_modification.mode = 32

# Eyeballing the distribution based on the law of large numbers:
cfg.case.server.model_modification.linfunc = 'randn' # will work decently for anything
cfg.case.server.model_modification.mode = None


user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
# loss_fn=loss_fn+model.loss()
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)
server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)


print(true_user_data)


user.plot3(true_user_data,name="cifar10_true.png")

shared_data['gradients']=dgp(shared_data['gradients'],75)

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], server.secrets,
                                                      dryrun=cfg.dryrun)
m1=_run_vision_metrics1(reconstructed_user_data, true_user_data,setup=setup)
user.plot3(reconstructed_user_data,name="cifar10_reconstruct.png")
