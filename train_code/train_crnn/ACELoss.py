# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
class ACE(nn.Module):

    def __init__(self):
        super(ACE, self).__init__()
        self.softmax = None;
        self.label = None;

    def forward(self, input, label):

        self.bs, self.h, self.w, _ = input.size()
        T_ = self.h * self.w

        input = input.view(self.bs, T_, -1)
        input = input + 1e-10

        self.softmax = input

        self.label = label

        # ACE Implementation (four fundamental formulas)
        input = torch.sum(input, 1)
        input = input / T_
        label = label / T_
        loss = (-torch.sum(torch.log(input) * label)) / self.bs

        return loss