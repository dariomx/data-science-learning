#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch import sigmoid, relu


class Highway(nn.Module):
    def __init__(self, w_embed_size: int):
        super(Highway, self).__init__()
        self.proj = nn.Linear(w_embed_size, w_embed_size, bias=True)
        self.gate = nn.Linear(w_embed_size, w_embed_size, bias=True)

    def forward(self, x_conv_out):
        # (w_batch_size, w_embed_size)
        x_proj = relu(self.proj(x_conv_out))
        # (w_batch_size, w_embed_size)
        x_gate = sigmoid(self.gate(x_conv_out))
        # (w_batch_size, w_embed_size)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        # (w_batch_size, w_embed_size)
        return x_highway
