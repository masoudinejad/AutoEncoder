import torch.nn as nn


def get_activation_fcn(act_fcn):
    match act_fcn:
        case "relu":
            return nn.ReLU(inplace=True)
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()