import numpy as np
import breaching
import torch
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
cfg = breaching.get_config(overrides=["case/server=malicious-fishing", "attack=april_analytic"])


def cupy_topk(input_gradient,k):
    # print(type(input_gradient))
    input_gradient=list(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].cpu().numpy()
        flattened_weights =np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, k)
        # # thresh=0
        # # print("1grad_tensor is",grad_tensor)
        t = np.where(np.abs(grad_tensor) < thresh, 0,grad_tensor)
        # print("2grad_tensor is", grad_tensor)
        input_gradient[i] = torch.Tensor(t).to(device)
    return input_gradient


device = torch.device(f'cuda:2') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

cfg.case.model = "vit_small_april"

cfg.case.data.partition = "unique-class" # This is the worst-case for the attack, as each user owns a unique class
cfg.case.user.num_data_points = 2 # maximum in the validation set. Feel free to load the training set and increase
cfg.case.user.user_idx = 1 # goldfish

cfg.case.user.provide_labels = True # Mostly out of convenience
cfg.case.server.target_cls_idx = 0 # Which class to attack?

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

[shared_data], [server_payload], true_user_data = server.run_protocol(user)
user.plot(true_user_data)
shared_data['gradients']=cupy_topk(shared_data['gradients'],80)
reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data],
                                                      server.secrets, dryrun=cfg.dryrun)
metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload],
                                    server.model, order_batch=True, compute_full_iip=False,
                                    cfg_case=cfg.case, setup=setup)
user.plot(reconstructed_user_data)

from breaching.cases.malicious_modifications.classattack_utils import print_gradients_norm, cal_single_gradients
single_gradients, single_losses = cal_single_gradients(user.model, loss_fn, true_user_data, setup=setup)
print_gradients_norm(single_gradients, single_losses)
