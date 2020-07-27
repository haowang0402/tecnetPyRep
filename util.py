import torch.nn as nn
import torch

def identity(x):
    return x
def activation(fn_name):
    fn = None
    if fn_name == 'relu':
        fn = nn.ReLU()
    elif fn_name == 'elu':
        fn = nn.ELU()
    elif fn_name == 'leaky_relu':
        fn = nn.LeakyReLU(0.1)
    else:
        fn = identity()
    return fn
