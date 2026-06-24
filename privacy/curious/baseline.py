import breaching
import os
os.chdir("/home2/xl/breaching_ATS/experiments/results/")
import torch
import logging, sys
import math
import numpy as np

from test_m import _run_vision_metrics1
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
cfg = breaching.get_config(overrides=["attack=imprint", "case/server=malicious-model-cah"])
# cfg = breaching.get_config(overrides=["attack=imprint", "case/server=malicious-model-rtf","case=6_large_batch_cifar"])
device = torch.device(f'cuda:0')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
import csv
import pandas as pd

from defense import soteria_imagenet
setup


# cfg.case.model = "lenet_zhu"


cfg.case.user.num_data_points = 16 # How many data points does this user own
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
# device = torch.device('cpu')
# torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
# setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
# setup
# cfg.case.user.num_data_points = 9 # How many data points does this user own
# cfg.case.server.model_modification.type = 'ImprintBlock' # What type of Imprint block will be grafted to the model
# cfg.case.server.model_modification.num_bins = 128 # How many bins are in the block
# print(cfg.case.model)
#
# # LeNetZhu
# # How does the block interact with the model?
# # The block can be placed later in the model given a position such as  '4.0.conv':
# cfg.case.server.model_modification.position = None  # None defaults to the first layer
# # The block can also be connected in various ways to the other layers:
# cfg.case.server.model_modification.connection = 'addition'
#
#
#
# # Which linear measurement function should be used?
# # We know that the input dataset is already normalized as preprocessing step
#
# # Knowing the distribution relatively well:
# cfg.case.server.model_modification.linfunc = 'fourier' # works well for any normalized image data
# cfg.case.server.model_modification.mode = 32
#
# # Eyeballing the distribution based on the law of large numbers:
# cfg.case.server.model_modification.linfunc = 'randn' # will work decently for anything
# cfg.case.server.model_modification.mode = None



dir="/home2/xl/breaching_ATS/experiments/CIFAR100/"
defense="/baseline/"
d="baseline_curious"



user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
# loss_fn=loss_fn+model.loss()
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)
server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)


print(true_user_data)


user.plot3(true_user_data,name=dir+defense+"cifar100_true.png")



reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], server.secrets,
                                                      dryrun=cfg.dryrun)


user.plot3(reconstructed_user_data,name=dir+defense+d+ "reconstruct.png")
metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                    server.model, order_batch=True, compute_full_iip=False,
                                    cfg_case=cfg.case, setup=setup)

m1=_run_vision_metrics1(reconstructed_user_data, true_user_data,setup=setup)
metrics1=[]
metrics1.append(m1)
fileName=dir+defense+"cifar100_metrics.csv"
##保存文件
with open(fileName,"w") as csv_file:
    writer=csv.writer(csv_file)
    for key,value in metrics.items():
        writer.writerow([key,value])

data = pd.read_csv(fileName, encoding='gb18030')
data2 = data.T
data2.to_csv(fileName)


header = ["mse", "psnr", "lpips","min_lpips", "ssim", "max_ssim", 'rpsnr', 'max_rpsnr']
with open(dir+defense +d+ 'cifar100_metrics1.csv', 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(metrics1)


print(metrics1)




# fileName="/home2/xl/breaching_ATS/results/lenet_zhu_modify_metrics_Rob_cifar100_ATS_size=4.csv"
#
#
#
# print("###########modify###########")
# metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
#                                     server.model, order_batch=True, compute_full_iip=False,
#                                     cfg_case=cfg.case, setup=setup)
# with open(fileName,"w") as csv_file:
#     writer=csv.writer(csv_file)
#     for key,value in metrics.items():
#         writer.writerow([key,value])
#
# data = pd.read_csv(fileName, encoding='gb18030')
# data2 = data.T
# data2.to_csv(fileName)
#
#
#
user.plot3(reconstructed_user_data,name=dir+defense+d+ "reconstruct.png")