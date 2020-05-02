#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,
                 c_embed_size: int,
                 w_embed_size: int,
                 kernel_size: int = 5):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(c_embed_size,
                              w_embed_size,
                              kernel_size=kernel_size,
                              bias=True)

    def forward(self, x_reshaped):
        # (w_batch_size, c_embed_size, max_word_length)
        x_conv = self.conv(x_reshaped)
        # (w_batch_size, w_embed_size, max_word_length - k + 1)
        return x_conv