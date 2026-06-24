import numpy as np
import breaching
import torch
import logging, sys
import csv
import pandas as pd



def soteria(input_gradient, model, ground_truth, pruning_rate=80):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device

    gt_data = ground_truth.clone()
    gt_data.requires_grad = True

    # register forward hook to get intermediate layer output
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]

        return hook

    # for ResNet-18
    handle = model.fc.register_forward_hook(get_activation('flatten'))
    out = model(gt_data)

    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:, f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    (feature_graph.data[:, f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:, f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    print('Soteria mask: ', sum(mask))

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)

    handle.remove()

    return gradient


def soteria_imagenet(input_gradient, model, ground_truth, pruning_rate=80):
    """
    Defense proposed in the Soteria paper.
    param:
        - input_gradient: the input_gradient
        - model: the ResNet-18 model
        - ground_truth: the benign image (for learning perturbed representation)
        - pruning_rate: the prune percentage
    Note: This implementation only works for ResNet-18
    """
    device = input_gradient[0].device

    gt_data = ground_truth.clone()
    gt_data.requires_grad = True

    # register forward hook to get intermediate layer output
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = input[0]

        return hook

    # for ResNet-18
    print(model.model)
    handle = model.model.fc.register_forward_hook(get_activation('flatten'))
    model=model.to(device)
    out = model(gt_data)

    feature_graph = activation['flatten']

    deviation_target = torch.zeros_like(feature_graph)
    deviation_x_norm = torch.zeros_like(feature_graph)
    for f in range(deviation_x_norm.size(1)):
        deviation_target[:, f] = 1
        feature_graph.backward(deviation_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
                    (feature_graph.data[:, f]) + 1e-10)
        model.zero_grad()
        gt_data.grad.data.zero_()
        deviation_target[:, f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_x_norm_sum = deviation_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    print('Soteria mask: ', sum(mask))

    gradient = [grad for grad in input_gradient]
    # apply mask
    gradient[-2] = gradient[-2] * torch.Tensor(mask).to(device)

    handle.remove()

    return gradient
def top_k(input_gradient,k=80):
    device = input_gradient[0].device
    input_gradient = list(input_gradient)
    # i=len(input_gradient)-2
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].cpu().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, k)
        t = np.where(np.abs(grad_tensor) < thresh, 0, grad_tensor)
        # thresh = np.percentile(flattened_weights, k)
        # t = np.where(np.abs(grad_tensor) < thresh, 0, grad_tensor)
        # flattened_weights = np.abs(t.flatten())
        # thresh = np.percentile(flattened_weights, k)
        # # thresh=0
        # # print("1grad_tensor is",grad_tensor)
        # t = np.where(np.abs(t) < thresh, 0, t)
        input_gradient[i] = torch.Tensor(t).to(device)
    return input_gradient


def sparse(input_gradient,k=75):
    # print(type(input_gradient))
    # 先去除5%Top后保留的大参数
    device = input_gradient[0].device
    input_gradient = list(input_gradient)
    for i in range(len(input_gradient)):
        grad_tensor = input_gradient[i].cpu()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, k)
        grad_tensor = np.where(np.abs(grad_tensor) < thresh, 0, grad_tensor)
        thresh = np.percentile(flattened_weights, 95)
        grad_tensor = np.where(np.abs(grad_tensor) > thresh, 0, grad_tensor)

        input_gradient[i] = torch.Tensor(grad_tensor).to(device)
    return input_gradient
